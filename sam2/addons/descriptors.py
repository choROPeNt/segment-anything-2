import torch



@torch.no_grad()
def s2_descriptor(patches: torch.Tensor, radial: bool = False, eps: float = 1e-12):
    """
    Compute two-point correlation S2 for binary patches.

    Parameters
    ----------
    patches : torch.Tensor
        Shape [B, H, W], values in {0,1} (float/bool is fine). On CPU/GPU/MPS.

    Returns
    -------
    result : dict
        - 'S2':   (B,H,W)      centered two-point correlation
    """
    assert patches.ndim == 3, "Expected [B,H,W]"
    x = patches.to(torch.float32)

    B, H, W = x.shape


    # ---- S2 via FFT (autocorrelation normalized by N) ----
    F = torch.fft.fft2(x, dim=(-2, -1))
    S2 = torch.real(torch.fft.ifft2(F * torch.conj(F), dim=(-2, -1))) / (H * W)
    S2 = torch.fft.fftshift(S2, dim=(-2, -1))  # center
    
    if radial:
        # Precompute radius bins (shared for batch)
        device = x.device
        y = torch.arange(H, device=device)
        z = torch.arange(W, device=device)
        yy, zz = torch.meshgrid(y, z, indexing="ij")
        cy, cz = (H - 1) / 2.0, (W - 1) / 2.0
        r = torch.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
        r_int = r.round().to(torch.int64)
        rmax = int(r_int.max().item())

        vals = S2.reshape(B, -1)
        bins = r_int.reshape(-1)
        prof = torch.zeros((B, rmax + 1), device=device, dtype=vals.dtype)
        prof.scatter_add_(1, bins.unsqueeze(0).expand(B, -1), vals)

        # counts per radius
        cnt = torch.zeros(rmax + 1, device=device, dtype=vals.dtype)
        cnt.scatter_add_(0, bins, torch.ones_like(bins, dtype=vals.dtype, device=device))
        prof = prof / torch.clamp(cnt, min=eps)

        return S2, prof  # (B, R)
    
    return S2


@torch.no_grad()
def phi_descriptor(patches: torch.Tensor):
    """
    Compute volume fraction (phi) for binary patches.

    Parameters
    ----------
    patches : torch.Tensor
        Shape [B, H, W], values in {0,1} (float/bool is fine). On CPU/GPU/MPS.
    Returns
    -------
    result : dict
        - 'phi':  (B,)         volume fraction
    """
    assert patches.ndim == 3, "Expected [B,H,W]"
    x = patches.to(torch.float32)

    B, H, W = x.shape

    # ---- phi ----
    phi = x.mean(dim=(1, 2))  # (B,)
    return phi


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