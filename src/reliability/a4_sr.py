"""Through-plane super-resolution, self-supervised per case (§2.4, Eq. 3–4).

Text: "adapted from the principle introduced by SMORE"; "reduced residual convolutional architecture
derived from VDSR"; "10 convolutional residual layers with 48 feature channels per layer"; "48 × 48
pixels"; "batch size of 16"; "Adam optimizer, an initial learning rate of 1e-3"; "L1 reconstruction
objective" (Eq. 4); "2500 iterations" (rate and iterations selected by HEBO -> a4_config.SR_LR/SR_ITERS);
degradation I_deg = U_k[D_k(G * I)] with Gaussian FWHM equal
to k (Eq. 3); "the trained network was applied to planes containing the z-axis... The reconstructed
orthogonal sections were subsequently combined to form the volumetric super-resolved CT" (the
TWO families of planes containing z, Y–Z and X–Z, combined by the mean); baseline = "resampled to
the same target voxel grid... without learned intensity correction" (trilinear); "All case-specific
network training and inference were performed on an NVIDIA GeForce GTX 1660 Ti GPU".

The network (`VDSRlite`) is imported from `mr_superres.py` unchanged (architecture of the text). The degradation
`degrade` is its own (decimation; linear U_k) and differs from the old `degrade_axis` (area + bicubic).
The training loop follows the SAME ORDER of calls to the random generator as the previous code, but the draws
differ (val_frac = 0 changes the population of `np.random.choice`); old and new runs are NOT comparable
weight by weight. No PSNR/SSIM.
No slice is held out (SR_VAL_FRAC = 0; §2.4 "patches extracted from the available volume" and §2.8.2
"432 image slices"). Declared assumptions: raw HU input; seed 0
for the evaluated network; degradation axis drawn at random; D_k = decimation with the mean of the two central
pixels for even
k; linear U_k.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import ndimage as ndi

from reliability import a4_config as C
from mr_superres import VDSRlite  # §2.4: architecture, unchanged


# ----------------------------------------------------------------------------- pick_device
def pick_device(require_cuda: bool = True) -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # run-to-run determinism
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            "§2.4 requires a GPU (GTX 1660 Ti); CUDA unavailable. Use require_cuda=False only in the smoke test."
        )
    return torch.device("cpu")


# ----------------------------------------------------------------------------- degradation (Eq. 3)
def degrade(img2d: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Eq. 3: I_deg = U_k[D_k(G * I)]. G = Gaussian with FWHM = k pixels along the degraded axis;
 D_k = subsampling with spacing k (decimation, aligned to the block centers, index (k-1)/2 + jk;
 for even k the center falls at a half-integer and the mean of the two central pixels is taken, which is the
 linear interpolation at that coordinate);
 U_k = LINEAR reinterpolation to the original grid (the same kernel as the trilinear baseline), with the
 centers aligned as in interpolate_trilinear_z (align_corners=False)."""
    sigma = k / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    s = [0.0, 0.0]
    s[axis] = sigma
    blurred = ndi.gaussian_filter(img2d, sigma=s)  # G
    H, W = img2d.shape
    L = H if axis == 0 else W
    n = -(-L // k)  # ceil(L/k): the period of D_k/U_k is EXACTLY k
    t = torch.from_numpy(blurred).float()[
        None, None
    ]  # (case of L not a multiple of k,
    if (
        C.SR_DEGRADATION_DK == "decimation"
    ):  # L//k samples reinterpolated to L would give period L/n != k)
        idx = torch.clamp(
            torch.arange(n) * k + (k - 1) // 2, max=L - 1
        )  # block center (last block: clamp at the border)
        if k % 2 == 0:  # even k: center between two pixels -> mean of the two
            idx2 = torch.clamp(idx + 1, max=L - 1)
            low = (
                0.5 * (t[:, :, idx, :] + t[:, :, idx2, :])
                if axis == 0
                else 0.5 * (t[:, :, :, idx] + t[:, :, :, idx2])
            )
        else:
            low = t[:, :, idx, :] if axis == 0 else t[:, :, :, idx]
    elif C.SR_DEGRADATION_DK == "area":
        if L % k:
            pad = n * k - L
            t = torch.nn.functional.pad(
                t, (0, 0, 0, pad) if axis == 0 else (0, pad, 0, 0), mode="replicate"
            )
        size = (n, W) if axis == 0 else (H, n)
        low = torch.nn.functional.interpolate(t, size=size, mode="area")
    else:
        raise ValueError(C.SR_DEGRADATION_DK)
    mode = "bilinear" if C.SR_DEGRADATION_UK == "linear" else "bicubic"
    size_up = (n * k, W) if axis == 0 else (H, n * k)
    up = torch.nn.functional.interpolate(
        low, size=size_up, mode=mode, align_corners=False
    )  # U_k, period k
    up = (
        up[:, :, :H, :] if axis == 0 else up[:, :, :, :W]
    )  # crop to the patch size
    return up[0, 0].numpy()


# ----------------------------------------------------------------------------- normalization
def normalize(thick: np.ndarray):
    """Intensity scale of the network input: "hu" = no scaling (§2.4 does not describe
 normalization; lo = 0, esc = 1); "minmax" = extremes of the thick volume, without clipping; "percentiles" = previous
 reading
 (percentiles 0.5/99.9 with clipping to [0, 1]). Returns (scaled volume, lo, esc) with HU = scaled * esc + lo."""
    mode = C.SR_NORMALIZATION
    if mode == "hu":
        return np.asarray(thick, np.float32).copy(), 0.0, 1.0
    if mode == "minmax":
        lo, hi = float(thick.min()), float(thick.max())
        esc = hi - lo
        return ((thick - lo) / (esc + 1e-6)).astype(np.float32), lo, esc
    lo, hi = (
        np.percentile(thick, C.SR_NORM_PERCENTILES[0]),
        np.percentile(thick, C.SR_NORM_PERCENTILES[1]),
    )
    esc = float(hi - lo)
    return np.clip((thick - lo) / (esc + 1e-6), 0, 1).astype(np.float32), float(lo), esc


def denormalize(vol_norm: np.ndarray, lo: float, esc: float) -> np.ndarray:
    return vol_norm * esc + lo


# ----------------------------------------------------------------------------- training (Eq. 3–4)
def train(
    slices,
    k: int,
    iters: int,
    dev: torch.device,
    seed: int,
    dropout: float = 0.0,
    patch: int = C.SR_PATCH,
    batch: int = C.SR_BATCH,
    lr: float = C.SR_LR,
    val_frac: float = C.SR_VAL_FRAC,
    verbose: bool = True,
):
    """slices: list of 2D arrays (in-plane slices of the THICK volume, normalized). Returns (network, info).
 Same order of calls to the generator as mr_superres.train (seed + shuffle + choice/randint), with val_frac = 0."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(slices)
    idx = np.arange(n)
    np.random.shuffle(idx)  # shuffle kept (same random sequence)
    nval = int(n * val_frac)
    tr_ids = idx[nval:]  # val_frac = 0: all slices are used for training
    effective_batches = []
    network = VDSRlite(ch=C.SR_CHANNELS, n=C.SR_BODY_N, dropout=dropout).to(dev)
    assert (
        sum(isinstance(m, nn.Conv2d) for m in network.modules()) == C.SR_CONV_LAYERS
    )  # §2.4: 10 convolutions
    opt = torch.optim.Adam(
        network.parameters(), lr=lr
    )  # §2.4 Adam (rate: a4_config.SR_LR)
    l1_loss = nn.L1Loss()  # Eq. 4
    t0 = time.time()
    hist = []
    for it in range(1, iters + 1):
        xb, yb = [], []
        for _ in range(batch):
            sl = slices[np.random.choice(tr_ids)]
            H, W = sl.shape
            if H <= patch or W <= patch:
                continue
            y0 = np.random.randint(0, H - patch)
            x0 = np.random.randint(0, W - patch)
            hr = sl[y0 : y0 + patch, x0 : x0 + patch]
            ax = np.random.randint(
                0, 2
            )  # degradation axis drawn between the two of the slice
            xb.append(degrade(hr, k, ax)[None])
            yb.append(hr[None])  # Eq. 3 (G, D_k, U_k)
        if not xb:
            continue
        effective_batches.append(len(xb))
        x = torch.from_numpy(np.stack(xb)).float().to(dev)
        y = torch.from_numpy(np.stack(yb)).float().to(dev)
        opt.zero_grad()
        loss = l1_loss(network(x), y)
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == 1:
            hist.append((it, float(loss)))
            if verbose:
                print(
                    "    iter %5d/%d  L1=%.4f  (%.0fs)"
                    % (it, iters, float(loss), time.time() - t0),
                    flush=True,
                )
    network.eval()
    info = {
        "iters": iters,
        "lr": float(lr),
        "seed": seed,
        "dropout": dropout,
        "n_slices": n,
        "n_training": len(tr_ids),
        "n_reserved": int(nval),
        "pick_device": str(dev),
        "seconds": round(time.time() - t0, 1),
        "loss_history": hist,
        "effective_batch_min": int(min(effective_batches)) if effective_batches else 0,
        "effective_batch_mode": int(np.bincount(effective_batches).argmax())
        if effective_batches
        else 0,
        "n_iterations_run": len(effective_batches),
    }
    return network, info


# ----------------------------------------------------------------------------- inference (two families of planes)
def _enable_dropout(network):
    for m in network.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()


def apply(
    network,
    vol_norm: np.ndarray,
    k: int,
    mc_dropout: bool = False,
    orientations=C.SR_ORIENTATIONS,
    chunk: int | None = None,
) -> np.ndarray:
    """vol_norm (Zt, Y, X) normalized thick -> super-resolved volume (Zt·k, Y, X).
 Network input = trilinear resampling to the target grid (the same as the baseline, §2.4); the network
 estimates the residual; applied to the Y–Z planes (sweeping x) and X–Z planes (sweeping y), with the z axis along the
 plane width in both (as in training, the degraded axis is an axis of the slice); the two
 reconstructions are combined by the mean."""
    from reliability.a4_data import interpolate_trilinear_z

    dev = next(network.parameters()).device
    up = torch.from_numpy(interpolate_trilinear_z(vol_norm, k))  # (Zup, Y, X), CPU
    Zup, Y, X = up.shape
    network.eval()
    if mc_dropout:
        _enable_dropout(network)
    outputs = []
    with torch.no_grad():
        for ori in orientations:
            out = torch.empty_like(up)
            if ori == "yz":
                width = X
                alt = Y
            elif ori == "xz":
                width = Y
                alt = X
            else:
                raise ValueError(ori)
            ch = (
                chunk
                if chunk is not None
                else (
                    1
                    if dev.type == "cpu"
                    else max(1, min(32, int(200e6 / (48.0 * alt * Zup * 4.0))))
                )
            )
            i0 = 0
            while i0 < width:
                i1 = min(width, i0 + ch)
                if ori == "yz":
                    planes = (
                        up[:, :, i0:i1].permute(2, 1, 0).contiguous().unsqueeze(1)
                    )  # [B,1,Y,Zup]
                else:
                    planes = (
                        up[:, i0:i1, :].permute(1, 2, 0).contiguous().unsqueeze(1)
                    )  # [B,1,X,Zup]
                try:
                    r = network(planes.to(dev))[:, 0].cpu()
                except torch.cuda.OutOfMemoryError:
                    if ch == 1:
                        raise
                    torch.cuda.empty_cache()
                    ch = max(1, ch // 2)
                    continue
                if ori == "yz":
                    out[:, :, i0:i1] = r.permute(2, 1, 0)
                else:
                    out[:, i0:i1, :] = r.permute(2, 0, 1)
                i0 = i1
            outputs.append(out)
    if mc_dropout:
        network.eval()
    return torch.stack(outputs).mean(dim=0).numpy()


def in_plane_slices(thick_norm: np.ndarray):
    """§2.4: "Self-supervised training pairs were generated from image planes with higher native
 spatial resolution" = the in-plane (axial) slices of the thick volume."""
    return [thick_norm[z] for z in range(thick_norm.shape[0])]
