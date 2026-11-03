"""
Training loss with diagnostic terms for SAM 2 fine-tuning.

:class:`DiagnosticMultiStepLoss` is the upstream :class:`training.loss_fns.MultiStepMultiMasksAndIous`
(identical gradients, identical ``core_loss``) plus detached diagnostics that the trainer logs next to the
losses (``Losses/{phase}_{key}_<name>`` in ``train_stats.json`` / ``val_stats.json`` and TensorBoard):

Loss bookkeeping
    ``w_loss_mask``, ``w_loss_dice``, ``w_loss_iou``, ``w_loss_class``
        weighted contribution of each term to ``core_loss`` (weight × loss)
    ``loss_mask_per_step``, ``loss_dice_per_step``
        losses divided by the number of prediction steps (1 initial prompt + correction clicks per frame)
    ``n_steps``
        prediction steps per frame

Mask quality (hard masks, logits > 0, objects present in the target only)
    ``iou_first`` / ``iou_last``
        IoU of the best of the candidate masks after the initial prompt / after the last correction click
    ``iou_first_point`` / ``iou_first_box``
        ``iou_first`` split by prompt type (logged only in batches with that prompt)
    ``iou_corr_gain``
        ``iou_last − iou_first``: what the correction clicks add
    ``area_bias_last``
        (pred area − GT area) / GT area of the last mask: > 0 over-, < 0 under-segmentation

IoU head (what the automatic mask generator relies on: it keeps the mask with the highest predicted IoU and
filters by ``pred_iou_thresh``)
    ``iou_pick_first``
        IoU of the candidate the IoU head would pick at the initial prompt
    ``iou_pick_regret``
        ``iou_first − iou_pick_first`` (0 = the head always picks the best candidate)
    ``iou_pick_acc``
        fraction of objects where the picked candidate is the best one
    ``iou_pred_err``
        mean |predicted IoU − actual IoU| over all candidates and steps
    ``best_idx_0`` …
        how often each multimask output is the best one at the initial prompt (a collapsed output shows up as ~0)

Object-score head
    ``obj_prob``
        mean predicted object probability; ``obj_neg_frac``: fraction of target slots without an object.
        ``obj_neg_frac = 0`` means ``loss_class`` has no negatives to learn from.

Use it in a training config with::

    loss:
      all:
        _target_: sam2.addons.train_losses.DiagnosticMultiStepLoss
        weight_dict: {loss_mask: 20, loss_dice: 1, loss_iou: 1, loss_class: 1}
        ...                          # all other MultiStepMultiMasksAndIous arguments unchanged
"""

from collections import defaultdict

import torch

from training.loss_fns import MultiStepMultiMasksAndIous


def _hard_iou(masks, gt):
    """IoU and areas of logit masks [N, M, H, W] vs. a binary target [N, 1, H, W] → (N, M) each."""
    pred = masks > 0
    g = (gt > 0).expand_as(pred)
    inter = (pred & g).flatten(2).sum(-1).float()
    union = (pred | g).flatten(2).sum(-1).float()
    return inter / union.clamp(min=1.0), pred.flatten(2).sum(-1).float(), g.flatten(2).sum(-1).float()


def _prompt_type(point_inputs):
    if point_inputs is None:
        return "mask"
    labels = point_inputs["point_labels"]
    return "box" if bool(((labels == 2) | (labels == 3)).any()) else "point"


class DiagnosticMultiStepLoss(MultiStepMultiMasksAndIous):
    """:class:`MultiStepMultiMasksAndIous` plus detached diagnostics (see module docstring)."""

    def __init__(self, weight_dict, diagnostics=True, **kwargs):
        super().__init__(weight_dict, **kwargs)
        self.diagnostics = diagnostics

    def forward(self, outs_batch, targets_batch):
        losses = super().forward(outs_batch, targets_batch)
        if self.diagnostics:
            with torch.no_grad():
                losses.update(self._diagnostics(outs_batch, targets_batch, losses))
        return losses

    def _diagnostics(self, outs_batch, targets_batch, losses):
        acc = defaultdict(list)
        n_steps_total = 0
        for outs, targets in zip(outs_batch, targets_batch):
            gt = targets.unsqueeze(1).float()                                   # [N, 1, H, W]
            present = (gt > 0).flatten(1).any(-1)                               # [N]
            masks_steps = outs["multistep_pred_multimasks_high_res"]
            ious_steps = outs["multistep_pred_ious"]
            obj_steps = outs["multistep_object_score_logits"]
            n_steps_total += len(masks_steps)
            acc["n_steps"].append(float(len(masks_steps)))
            acc["obj_neg_frac"].append(1.0 - present.float().mean().item())
            acc["obj_prob"].append(torch.cat([o.sigmoid().flatten() for o in obj_steps]).mean().item())

            pred_err = []
            for masks, pious in zip(masks_steps, ious_steps):
                iou, _, _ = _hard_iou(masks, gt)
                pred_err.append((pious.float() - iou).abs()[present].flatten())
            if present.any():
                acc["iou_pred_err"].append(torch.cat(pred_err).mean().item())
            if not present.any():
                continue

            # initial prompt: candidate masks, best (oracle) vs. picked by the IoU head
            iou0, _, _ = _hard_iou(masks_steps[0], gt)
            iou0, piou0 = iou0[present], ious_steps[0].float()[present]
            best = iou0.argmax(-1)
            pick = piou0.argmax(-1)
            rows = torch.arange(len(iou0), device=iou0.device)
            iou_first = iou0[rows, best]
            iou_pick = iou0[rows, pick]
            acc["iou_first"].append(iou_first.mean().item())
            acc[f"iou_first_{_prompt_type(outs['multistep_point_inputs'][0])}"].append(iou_first.mean().item())
            acc["iou_pick_first"].append(iou_pick.mean().item())
            acc["iou_pick_regret"].append((iou_first - iou_pick).mean().item())
            if iou0.shape[1] > 1:
                acc["iou_pick_acc"].append((pick == best).float().mean().item())
                for k in range(iou0.shape[1]):
                    acc[f"best_idx_{k}"].append((best == k).float().mean().item())

            # after the last correction click
            iouL, areaP, areaG = _hard_iou(masks_steps[-1], gt)
            iouL, areaP, areaG = iouL[present], areaP[present], areaG[present]
            bestL = iouL.argmax(-1)
            rows = torch.arange(len(iouL), device=iouL.device)
            acc["iou_last"].append(iouL[rows, bestL].mean().item())
            acc["iou_corr_gain"].append((iouL[rows, bestL] - iou_first).mean().item())
            acc["area_bias_last"].append(((areaP[rows, bestL] - areaG[rows, 0]) / areaG[rows, 0]).mean().item())

        device = losses["loss_mask"].device if torch.is_tensor(losses["loss_mask"]) else None
        out = {k: torch.tensor(sum(v) / len(v), device=device) for k, v in acc.items()}
        for k, w in self.weight_dict.items():
            v = losses[k]
            out[f"w_{k}"] = (v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=device)) * w
        steps = max(n_steps_total, 1)
        for k in ("loss_mask", "loss_dice"):
            v = losses[k]
            out[f"{k}_per_step"] = (v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=device)) / steps
        return out
