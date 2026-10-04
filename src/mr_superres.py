"""Self-supervised THROUGH-PLANE super-resolution of MRI (SMORE style).

Idea (Zhao et al., "SMORE: A Self-Supervised Anti-aliasing and Super-Resolution
Algorithm for MRI Using Deep Learning", IEEE TMI 2021, DOI 10.1109/TMI.2020.3037187
-- CHECK DOI): the IN-PLANE slices have high resolution (0.197 mm); the THROUGH-plane
resolution is low (3.5 mm slice / 4.2 mm spacing). A 2D network is trained to
restore high resolution from degraded versions of the high-resolution slices THEMSELVES
(simulating the slice thickness), and this network is applied in the low-resolution
direction. It uses NO external data nor ground truth.

Own implementation (author). Network: VDSR-lite (residual). Reports real PSNR/SSIM
on a validation set (held-out slices, degraded) -- an honest metric of
how much the network recovers from the SIMULATED degradation (it does not prove recovery on the real one).

Usage:
 python src/mr_superres.py --in data/interim/exam001_t1_short.nii.gz \
 --out data/interim/exam001_t1_short_SR.nii.gz --iters 3000
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# --------------------------- pick_device (CPU/GPU) ---------------------------
def get_device(pref=None):
    """Compute device. Order: argument > env SR_DEVICE > auto (cuda if
 available). The protocol GEOMETRY does not change with the pick_device -- only the
 floating-point arithmetic of the convolutions; therefore the pick_device is
 recorded in the metrics of each run."""
    name = (pref or os.environ.get("SR_DEVICE", "auto")).lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("  [warning] CUDA unavailable -> falling back to CPU")
        name = "cpu"
    return torch.device(name)


DEVICE = get_device()
# convolution algorithm autotune: valid for the fixed sizes of the training
CUDNN_BENCHMARK = os.environ.get("SR_CUDNN_BENCHMARK", "0") == "1"  # default: deterministic
if DEVICE.type == "cuda":
    if CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    else:  # run-to-run reproducibility on the GPU; small cost
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# --------------------------- network (VDSR-lite) --------------------------------
class VDSRlite(nn.Module):
    def __init__(self, ch=48, n=8, dropout=0.0):
        """dropout: p of the Dropout2d after each conv+relu block of the body.
 dropout=0.0 (default) reproduces EXACTLY the network used in the previous
 results (no layer is inserted). >0 enables stochastic
 sampling (MC-dropout / Dropsembles) without changing the rest."""
        super().__init__()
        body = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n):
            body += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
            if dropout > 0:
                body += [nn.Dropout2d(dropout)]
        body += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.body = nn.Sequential(*body)
        self.dropout_p = float(dropout)

    def forward(self, x):
        return x + self.body(x)      # learns the residual (high frequency)


# --------------------------- simulated degradation -----------------------------
def _fwhm_to_sigma(fwhm_vox):
    return fwhm_vox / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def degrade_axis(img2d, scale, axis):
    """Simulates the thick-slice PSF along 'axis': blurs (Gaussian, FWHM=
 scale) + subsamples by 'scale' + reinterpolates to the original size (bicubic)
 -> same dimension, but with the blur typical of low resolution."""
    sigma = _fwhm_to_sigma(scale)
    s = [0, 0]; s[axis] = sigma
    blur = ndi.gaussian_filter(img2d, sigma=s)
    # subsample and back (nearest down, cubic up) via torch to stay smooth
    t = torch.from_numpy(blur).float()[None, None]
    H, W = img2d.shape
    if axis == 1:
        down = torch.nn.functional.interpolate(t, size=(H, max(1, W // scale)),
                                               mode="area")
        up = torch.nn.functional.interpolate(down, size=(H, W), mode="bicubic",
                                             align_corners=False)
    else:
        down = torch.nn.functional.interpolate(t, size=(max(1, H // scale), W),
                                               mode="area")
        up = torch.nn.functional.interpolate(down, size=(H, W), mode="bicubic",
                                             align_corners=False)
    return up[0, 0].numpy()


# --------------------------- geometry-oriented loss ---------------------
def _grad_mag(y):
    """Magnitude of the 2D gradient (central difference), same size as y [B,1,H,W].
 The bone border = where |grad| is high = where the mesh surface is born."""
    gy = torch.zeros_like(y); gx = torch.zeros_like(y)
    gy[:, :, 1:-1, :] = (y[:, :, 2:, :] - y[:, :, :-2, :]) * 0.5
    gx[:, :, :, 1:-1] = (y[:, :, :, 2:] - y[:, :, :, :-2]) * 0.5
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _edge_weighted_l1(out, y, lam):
    """Border-weighted L1: w = 1 + lam * |grad y| / mean(|grad y|).
 lam=0 falls back EXACTLY to the mean L1 (same value as nn.L1Loss)."""
    g = _grad_mag(y)
    w = 1.0 + lam * g / (g.mean() + 1e-12)
    return (w * (out - y).abs()).mean()


# --------------------------- training ------------------------------------------
def train(hr_slices, scale, iters, patch=48, bs=16, lr=1e-3, seed=0, val_frac=0.15,
          edge_weight=0.0, device=None, dropout=0.0):
    dev = device if device is not None else DEVICE
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(hr_slices)
    idx = np.arange(n); np.random.shuffle(idx)
    nval = max(1, int(n * val_frac))
    val_ids, tr_ids = idx[:nval], idx[nval:]
    net = VDSRlite(dropout=dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.L1Loss()                       # used when edge_weight == 0
    hist = []
    t0 = time.time()
    for it in range(1, iters + 1):
        xb, yb = [], []
        for _ in range(bs):
            sl = hr_slices[np.random.choice(tr_ids)]
            H, W = sl.shape
            if H <= patch or W <= patch:
                continue
            y0 = np.random.randint(0, H - patch); x0 = np.random.randint(0, W - patch)
            hr = sl[y0:y0 + patch, x0:x0 + patch]
            ax = np.random.randint(0, 2)              # degrades in X or Y (robust)
            lr_img = degrade_axis(hr, scale, ax)
            xb.append(lr_img[None]); yb.append(hr[None])
        if not xb:
            continue
        xb = torch.from_numpy(np.stack(xb)).float().to(dev)
        yb = torch.from_numpy(np.stack(yb)).float().to(dev)
        opt.zero_grad()
        out = net(xb)
        # edge_weight==0 -> pure L1 (identical to the previous); >0 -> border-weighted L1
        loss = lossf(out, yb) if edge_weight == 0.0 \
            else _edge_weighted_l1(out, yb, edge_weight)
        loss.backward(); opt.step()
        if it % 200 == 0 or it == 1:
            hist.append((it, float(loss)))
            print(f"  iter {it:5d}/{iters}  L1={float(loss):.4f}  "
                  f"({time.time()-t0:.0f}s)")
    # validation: PSNR/SSIM on held-out (degraded) slices, restored
    net.eval(); ps, ss, ps0, ss0 = [], [], [], []
    with torch.no_grad():
        for vi in val_ids:
            hr = hr_slices[vi]
            lr_img = degrade_axis(hr, scale, 1)       # reference axis
            rng = float(hr.max() - hr.min()) or 1.0
            out = net(torch.from_numpy(lr_img[None, None]).float().to(dev)
                      )[0, 0].cpu().numpy()
            ps.append(psnr(hr, out, data_range=rng))
            ss.append(ssim(hr, out, data_range=rng))
            ps0.append(psnr(hr, lr_img, data_range=rng))   # baseline: bicubic
            ss0.append(ssim(hr, lr_img, data_range=rng))
    metrics = {
        "scale": int(scale), "iters": int(iters), "edge_weight": float(edge_weight),
        "dropout": float(dropout),
        "device": str(dev),
        "cudnn_benchmark": bool(dev.type == "cuda" and CUDNN_BENCHMARK),
        "n_train": int(len(tr_ids)),
        "n_val": int(len(val_ids)), "train_seconds": round(time.time() - t0, 1),
        "PSNR_bicubic": round(float(np.mean(ps0)), 3),
        "PSNR_network": round(float(np.mean(ps)), 3),
        "SSIM_bicubic": round(float(np.mean(ss0)), 4),
        "SSIM_network": round(float(np.mean(ss)), 4),
        "loss_hist": hist,
    }
    return net, metrics


# --------------------------- application to the volume -----------------------------
def apply_through_plane(net, vol, scale, chunk=None, mc_dropout=False):
    """Applies the network in the Z direction (low resolution). vol: [Z,Y,X].
 Upsamples Z by 'scale' (bicubic) and refines with the network, slicing into planes that
 contain Z. The planes go in a BATCH to the network's pick_device -- the network is purely
 convolutional (no batch normalization), so batch and plane-by-plane give the
 same result; only the number of calls changes."""
    dev = next(net.parameters()).device
    Z, Y, X = vol.shape
    up = torch.nn.functional.interpolate(
        torch.from_numpy(vol[None, None]).float(),
        size=(Z * scale, Y, X), mode="trilinear", align_corners=False)[0, 0]
    Zup = up.shape[0]
    if chunk is None:                       # ~200 MB per intermediate activation
        budget = 200e6 / (48.0 * Y * Zup * 4.0)
        chunk = 1 if dev.type == "cpu" else max(1, min(32, int(budget)))
    net.eval()
    if mc_dropout:                # MC-dropout: only the dropout layers become
        for _m in net.modules():  # stochastic again (the rest stays in eval)
            if isinstance(_m, nn.Dropout2d):
                _m.train()
    out = torch.empty_like(up)
    with torch.no_grad():
        x0 = 0
        while x0 < X:
            x1 = min(X, x0 + chunk)
            # [Zup,Y,B] -> [B,1,Y,Zup]
            planes = up[:, :, x0:x1].permute(2, 1, 0).contiguous().unsqueeze(1)
            try:
                r = net(planes.to(dev))[:, 0].cpu()
            except torch.cuda.OutOfMemoryError:
                if chunk == 1:
                    raise
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
                print(f"  [warning] OOM -> chunk={chunk}")
                continue
            out[:, :, x0:x1] = r.permute(2, 1, 0)
            x0 = x1
    return out.numpy()


def main():
    ap = argparse.ArgumentParser(description="Through-plane super-resolution (SMORE-like)")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--scale", type=int, default=0, help="0=auto (sz/sx)")
    ap.add_argument("--device", default=None, help="cuda | cpu | auto (default)")
    args = ap.parse_args()
    dev = get_device(args.device)

    img = sitk.ReadImage(args.inp)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)   # [Z,Y,X]
    sx, sy, sz = img.GetSpacing()
    scale = args.scale or int(round(sz / ((sx + sy) / 2)))
    print(f"volume {vol.shape} spacing=({sx:.3f},{sy:.3f},{sz:.3f})  scale={scale}")

    # normalize to [0,1]
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    voln = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)
    hr_slices = [voln[z] for z in range(voln.shape[0])]     # in-plane slices (HR)

    print(f"training (self-supervised,{args.iters} iters, {dev})...")
    net, metrics = train(hr_slices, scale, args.iters, device=dev)
    print("validation metrics:", json.dumps(metrics, ensure_ascii=False,
          indent=2, default=str))

    print("applying along Z...")
    sr = apply_through_plane(net, voln, scale)
    sr = sr * (hi - lo) + lo                                 # undo normalization

    out_img = sitk.GetImageFromArray(sr.astype(np.float32))
    out_img.SetSpacing((sx, sy, sz / scale))
    out_img.SetOrigin(img.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, -(scale - 1) / (2.0 * scale))))  # sub-slice 0 at -(k-1)/(2k) of the thick index
    out_img.SetDirection(img.GetDirection())
    sitk.WriteImage(out_img, args.out)
    with open(os.path.splitext(args.out)[0] + "_metrics.json", "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False, default=str)
    torch.save(net.state_dict(), os.path.splitext(args.out)[0] + "_net.pt")
    print(f"OK -> {args.out}  (Z: {vol.shape[0]} -> {sr.shape[0]}, "
          f"spacing Z {sz:.2f} -> {sz/scale:.3f} mm)")


if __name__ == "__main__":
    main()
