import torch


def onehot_from_labelmap(
    label: torch.Tensor,
    n_classes: int | None = None,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Convert an integer label map to one-hot binary channels.

    Parameters
    ----------
    label : torch.Tensor
        Shape ``[B, H, W]``, integer class ids.
    n_classes : int, optional
        Number of classes.  Inferred from ``label.max()`` when omitted.
    ignore_index : int
        Class id to exclude from the output (default 0 = background).

    Returns
    -------
    onehot : torch.Tensor
        Shape ``[B, C, H, W]``, float32, one channel per class
        (background excluded when ``ignore_index=0``).
    class_ids : list[int]
        Class ids corresponding to each channel.
    """
    ids = sorted(
        int(v) for v in label.unique().tolist() if int(v) != ignore_index
    )
    if n_classes is not None:
        ids = [i for i in range(1, n_classes + 1) if i != ignore_index]

    onehot = torch.stack(
        [(label == c).to(torch.float32) for c in ids], dim=1
    )  # (B, C, H, W)
    return onehot, ids


def _s2_bhw(x: torch.Tensor, radial: bool, eps: float):
    """Core S2 computation for (B, H, W) float32 tensor."""
    B, H, W = x.shape
    F = torch.fft.fft2(x, dim=(-2, -1))
    S2 = torch.real(torch.fft.ifft2(F * torch.conj(F), dim=(-2, -1))) / (H * W)
    S2 = torch.fft.fftshift(S2, dim=(-2, -1))

    if not radial:
        return S2, None

    device = x.device
    yy, zz = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    r = torch.sqrt((yy - (H - 1) / 2.0) ** 2 + (zz - (W - 1) / 2.0) ** 2)
    r_int = r.round().to(torch.int64)
    rmax = int(r_int.max().item())

    vals = S2.reshape(B, -1)
    bins = r_int.reshape(-1)
    prof = torch.zeros((B, rmax + 1), device=device, dtype=vals.dtype)
    prof.scatter_add_(1, bins.unsqueeze(0).expand(B, -1), vals)
    cnt = torch.zeros(rmax + 1, device=device, dtype=vals.dtype)
    cnt.scatter_add_(0, bins, torch.ones_like(bins, dtype=vals.dtype))
    prof = prof / cnt.clamp(min=eps)

    return S2, prof  # (B, H, W), (B, R)


@torch.no_grad()
def s2_descriptor(patches: torch.Tensor, radial: bool = False, eps: float = 1e-12):
    """
    Compute two-point correlation S2.

    Parameters
    ----------
    patches : torch.Tensor
        ``[B, H, W]``   — single binary phase per sample, or
        ``[B, C, H, W]``— C one-hot channels (e.g. from ``onehot_from_labelmap``).

    Returns
    -------
    S2 : torch.Tensor
        ``[B, H, W]`` or ``[B, C, H, W]``
    S2r : torch.Tensor  (only when ``radial=True``)
        ``[B, R]`` or ``[B, C, R]``
    """
    if patches.ndim == 3:
        x = patches.to(torch.float32)
        S2, prof = _s2_bhw(x, radial, eps)
        return (S2, prof) if radial else S2

    assert patches.ndim == 4, "Expected [B, H, W] or [B, C, H, W]"
    B, C, H, W = patches.shape
    x = patches.to(torch.float32).reshape(B * C, H, W)
    S2_flat, prof_flat = _s2_bhw(x, radial, eps)
    S2 = S2_flat.reshape(B, C, H, W)

    if radial:
        return S2, prof_flat.reshape(B, C, -1)   # (B, C, R)
    return S2


@torch.no_grad()
def phi_descriptor(patches: torch.Tensor) -> torch.Tensor:
    """
    Volume fraction per sample (and per channel for multi-phase input).

    Parameters
    ----------
    patches : torch.Tensor
        ``[B, H, W]`` or ``[B, C, H, W]``

    Returns
    -------
    phi : torch.Tensor
        ``[B]`` or ``[B, C]``
    """
    x = patches.to(torch.float32)
    if x.ndim == 3:
        return x.mean(dim=(1, 2))          # (B,)
    assert x.ndim == 4
    return x.mean(dim=(2, 3))             # (B, C)


@torch.no_grad()
def corr_length_halfheight(S2r: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Half-height correlation length per patch, using r = 1 as the peak reference.
    S2r: [B, R], phi: [B] in [0,1]
    Returns: zeta [B] (int64), first r>=1 where C(r) <= 0.5*C(1); R-1 if never drops.
    """
    B, R = S2r.shape
    base = (phi ** 2).unsqueeze(1)      # [B,1]
    C = S2r - base                      # [B,R]

    # Use C at index 1 as reference
    C1 = C[:, 1].clamp_min(1e-12)       # [B]
    thresh = 0.5 * C1                   # [B]

    # Condition only for indices >= 1
    cond = C[:, 1:] <= thresh[:, None]  # [B, R-1]

    # Find first such index (relative to r=1)
    has_hit = cond.any(dim=1)
    idx_rel = torch.argmax(cond.to(torch.int32), dim=1)  # range: 0..R-2

    # Build final absolute indices
    idx = torch.full((B,), R-1, device=S2r.device, dtype=torch.int64)
    idx[has_hit] = idx_rel[has_hit] + 1   # shift back by +1 to real r-index

    return idx

@torch.no_grad()
def integral_range_from_S2r(S2r: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Same as before, batched. Returns A_int [B] in pixel^2.
    """
    B, R = S2r.shape
    base = (phi ** 2).unsqueeze(1)
    C = S2r - base
    Cpos = torch.clamp(C, min=0.0)
    r = torch.arange(R, device=S2r.device, dtype=S2r.dtype)  # [R]
    return (2.0 * torch.pi * (Cpos * r).sum(dim=1))          # [B]

@torch.no_grad()
def rve_size_from_integral_range(phi_mean: torch.Tensor, A_int_mean: torch.Tensor, cv_target: float) -> torch.Tensor:
    return torch.sqrt((phi_mean * (1.0 - phi_mean)) * A_int_mean / (cv_target ** 2))