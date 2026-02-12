# flake8: noqa
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import drct.archs
import drct.data
import drct.models

from basicsr.utils.options import parse_options
from basicsr.utils import get_root_logger, mkdir_and_rename
from basicsr.data import build_dataset, build_dataloader
from basicsr.models import build_model
from basicsr.metrics import calculate_psnr, calculate_ssim


# -------------------------
# Image helpers
# -------------------------
def tensor2img01(t: torch.Tensor) -> np.ndarray:
    """CHW/1CHW torch -> HWC float32 [0,1]."""
    if t.ndim == 4:
        t = t[0]
    t = t.detach().float().cpu().clamp_(0, 1)
    return t.permute(1, 2, 0).numpy()


def rgb2y(img: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> Y [0,1], BT.601."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0
    return np.clip(y, 0, 1)


def bicubic_upsample(lq: torch.Tensor, scale: int) -> torch.Tensor:
    """NCHW -> NCHW"""
    return F.interpolate(lq, scale_factor=scale, mode="bicubic", align_corners=False)


# -------------------------
# DI (Gini-based)
# -------------------------
def gini_coefficient(x: np.ndarray) -> float:
    x = x.astype(np.float64).flatten()
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


def diffusion_index_from_map(m: np.ndarray) -> float:
    """DI = 1 - gini (diffused -> larger DI)."""
    m = np.clip(m.astype(np.float64), 0, None)
    g = gini_coefficient(m.flatten())
    return float(1.0 - g)


# -------------------------
# Edge-based patch selection ONLY
# -------------------------
def sobel_edge_strength(gray01: np.ndarray) -> np.ndarray:
    """gray01: HxW in [0,1] -> HxW edge magnitude (normalized)."""
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float64)
    Ky = np.array([[1,  2,  1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=np.float64)

    g = gray01.astype(np.float64)
    g = np.pad(g, ((1, 1), (1, 1)), mode="reflect")

    H, W = gray01.shape
    gx = np.zeros((H, W), dtype=np.float64)
    gy = np.zeros((H, W), dtype=np.float64)

    # (속도 필요하면 scipy/torch conv로 바꾸면 됨. 지금은 “최종 동작 보장” 위주)
    for i in range(H):
        for j in range(W):
            patch = g[i:i+3, j:j+3]
            gx[i, j] = np.sum(patch * Kx)
            gy[i, j] = np.sum(patch * Ky)

    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-8)
    return mag


def find_patch_by_score(score_map: np.ndarray, patch=64, stride=8):
    """score_map: HxW -> best (y,x,mean_score) maximizing patch mean score."""
    H, W = score_map.shape
    patch = min(patch, H, W)

    S = np.pad(score_map, ((1, 0), (1, 0)), mode="constant")
    S = np.cumsum(np.cumsum(S, axis=0), axis=1)

    def mean(y0, x0):
        y1, x1 = y0 + patch, x0 + patch
        s = S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]
        return s / (patch * patch)

    best_y, best_x, best_score = 0, 0, -1.0
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            m = mean(y, x)
            if m > best_score:
                best_y, best_x, best_score = y, x, float(m)
    return best_y, best_x, best_score


def save_and_show_edge_patch(best, out_png, patch=64, stride=8):
    """
    ✅ 박스 위치는 오직 '경계선(edge) 강한 곳' 기준
    ✅ DI는 BI-HR 맵 기반으로 표시
    """
    hr = best["hr"]  # HWC [0,1]
    bi = best["bi"]
    sr = best["sr"]

    # edge score on HR (gray)
    gray = 0.299 * hr[..., 0] + 0.587 * hr[..., 1] + 0.114 * hr[..., 2]
    edge = sobel_edge_strength(gray)

    y, x, score_val = find_patch_by_score(edge, patch=patch, stride=stride)

    # heatmap (시각화용): SR-HR
    diff_sr = np.mean(np.abs(sr - hr), axis=2)
    diff_sr = diff_sr / (diff_sr.max() + 1e-8)
    diff_sr = np.clip(diff_sr, 0, 1)

    # DI (표시용): BI-HR 기반
    bihr = np.mean(np.abs(bi - hr), axis=2)
    bihr_n = bihr / (bihr.max() + 1e-8)
    di = diffusion_index_from_map(bihr_n)

    # crops
    hr_p = hr[y:y + patch, x:x + patch, :]
    bi_p = bi[y:y + patch, x:x + patch, :]
    sr_p = sr[y:y + patch, x:x + patch, :]
    heat_p = diff_sr[y:y + patch, x:x + patch]

    plt.figure(figsize=(16, 7))

    ax0 = plt.subplot(2, 4, 1)
    ax0.imshow(hr); ax0.set_title("HR (GT) w/ box"); ax0.axis("off")
    ax0.add_patch(Rectangle((x, y), patch, patch, fill=False, edgecolor="red", linewidth=2))

    ax1 = plt.subplot(2, 4, 2)
    ax1.imshow(bi)
    ax1.set_title(f"Bicubic\nPSNR {best['psnr_bi']:.2f} / SSIM {best['ssim_bi']:.4f}")
    ax1.axis("off")
    ax1.add_patch(Rectangle((x, y), patch, patch, fill=False, edgecolor="red", linewidth=2))

    ax2 = plt.subplot(2, 4, 3)
    ax2.imshow(sr)
    ax2.set_title(f"Student(SR)\nPSNR {best['psnr_sr']:.2f} / SSIM {best['ssim_sr']:.4f}")
    ax2.axis("off")
    ax2.add_patch(Rectangle((x, y), patch, patch, fill=False, edgecolor="red", linewidth=2))

    ax3 = plt.subplot(2, 4, 4)
    ax3.imshow(diff_sr, cmap="hot")
    ax3.set_title(f"Heatmap |SR-HR|\nDI (from BI-HR) = {di:.4f}")
    ax3.axis("off")
    ax3.add_patch(Rectangle((x, y), patch, patch, fill=False, edgecolor="red", linewidth=2))

    ax4 = plt.subplot(2, 4, 5); ax4.imshow(hr_p); ax4.set_title("HR patch"); ax4.axis("off")
    ax5 = plt.subplot(2, 4, 6); ax5.imshow(bi_p); ax5.set_title("Bicubic patch"); ax5.axis("off")
    ax6 = plt.subplot(2, 4, 7); ax6.imshow(sr_p); ax6.set_title("SR patch"); ax6.axis("off")
    ax7 = plt.subplot(2, 4, 8); ax7.imshow(heat_p, cmap="hot"); ax7.set_title("Heatmap patch"); ax7.axis("off")

    plt.suptitle(
        f"BEST PSNR: {best['img_name']} (dataset={best['dataset']}) | EDGE patch@({x},{y}) score={score_val:.4e} | patch={patch}",
        y=1.02
    )
    plt.tight_layout()
    os.makedirs(osp.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"[BEST] dataset={best['dataset']} img={best['img_name']}")
    print(f"  Bicubic: PSNR {best['psnr_bi']:.4f}, SSIM {best['ssim_bi']:.6f}")
    print(f"  Student: PSNR {best['psnr_sr']:.4f}, SSIM {best['ssim_sr']:.6f}")
    print(f"[Saved] {out_png}")


# -------------------------
# Main pipeline
# -------------------------
@torch.no_grad()
def test_best_edge_DI_pipeline(root_path: str):
    opt, _ = parse_options(root_path, is_train=False)

    exp_name = opt.get("name", "test")
    results_root = opt["path"].get("results_root", osp.join(root_path, "results"))
    log_root = opt["path"].get("log", osp.join(root_path, "logs"))

    mkdir_and_rename(log_root)
    logger = get_root_logger(
        logger_name="basicsr",
        log_level="INFO",
        log_file=osp.join(log_root, f"{exp_name}_best_edge_DI.log"),
    )

    scale = opt.get("scale", 2)
    metric_opt = opt["val"]["metrics"]
    crop_border = metric_opt["psnr"].get("crop_border", scale)
    test_y = metric_opt["psnr"].get("test_y_channel", True)

    # ✅ vis patch size: 네 네트워크 img_size로 자동 맞춤
    vis_patch = int(opt.get("network_g", {}).get("img_size", 64))

    logger.info(f"Exp: {exp_name}")
    logger.info(f"scale={scale}, crop_border={crop_border}, test_y_channel={test_y}")
    logger.info(f"vis_patch(network_g.img_size)={vis_patch}")
    logger.info(f"results_root={results_root}")

    model = build_model(opt)
    model.net_g.eval()

    best_global = None

    for phase, dataset_opt in opt["datasets"].items():
        if not phase.startswith("test"):
            continue

        dataset_name = dataset_opt.get("name", phase)
        dataset = build_dataset(dataset_opt)
        dataloader = build_dataloader(
            dataset,
            dataset_opt,
            num_gpu=opt.get("num_gpu", 1),
            dist=False,
            sampler=None,
        )

        logger.info(f"Testing {dataset_name} (#images={len(dataset)})")

        for idx, data in enumerate(dataloader):
            model.feed_data(data)
            model.test()
            visuals = model.get_current_visuals()

            sr_t = visuals.get("result", None)
            gt_t = visuals.get("gt", None)
            lq_t = visuals.get("lq", None)
            if sr_t is None or gt_t is None or lq_t is None:
                raise RuntimeError(
                    f"visuals keys={list(visuals.keys())}\n"
                    "result/gt/lq 키가 없으면 여기 키를 네 환경에 맞게 바꿔야 함."
                )

            sr = tensor2img01(sr_t)
            gt = tensor2img01(gt_t)
            bi = tensor2img01(bicubic_upsample(lq_t, scale))

            # size align
            h = min(sr.shape[0], gt.shape[0], bi.shape[0])
            w = min(sr.shape[1], gt.shape[1], bi.shape[1])
            sr = sr[:h, :w, :]
            gt = gt[:h, :w, :]
            bi = bi[:h, :w, :]

            # metrics (0~255 기준으로 맞춤)
            if test_y:
                sr_eval = rgb2y(sr)
                gt_eval = rgb2y(gt)
                bi_eval = rgb2y(bi)
            else:
                sr_eval, gt_eval, bi_eval = sr, gt, bi

            sr_eval = (sr_eval * 255.0).astype(np.float64)
            gt_eval = (gt_eval * 255.0).astype(np.float64)
            bi_eval = (bi_eval * 255.0).astype(np.float64)

            psnr_sr = calculate_psnr(sr_eval, gt_eval, crop_border=crop_border, input_order="HWC", test_y_channel=False)
            ssim_sr = calculate_ssim(sr_eval, gt_eval, crop_border=crop_border, input_order="HWC", test_y_channel=False)

            psnr_bi = calculate_psnr(bi_eval, gt_eval, crop_border=crop_border, input_order="HWC", test_y_channel=False)
            ssim_bi = calculate_ssim(bi_eval, gt_eval, crop_border=crop_border, input_order="HWC", test_y_channel=False)

            # name
            img_path = None
            for k in ["gt_path", "lq_path", "path"]:
                if k in data:
                    v = data[k]
                    img_path = v[0] if isinstance(v, (list, tuple)) else v
                    break
            img_name = osp.basename(img_path) if img_path else f"{dataset_name}_{idx:06d}"

            cand = {
                "dataset": dataset_name,
                "img_name": img_name,
                "psnr_sr": float(psnr_sr),
                "ssim_sr": float(ssim_sr),
                "psnr_bi": float(psnr_bi),
                "ssim_bi": float(ssim_bi),
                "hr": gt,
                "sr": sr,
                "bi": bi,
            }

            # best sample selection is still by SR PSNR (이건 “최고 PSNR 샘플” 요구사항)
            if (best_global is None) or (cand["psnr_sr"] > best_global["psnr_sr"]):
                best_global = cand

    assert best_global is not None, "유효한 샘플을 찾지 못했습니다(경로/파일명 매칭 확인)."

    out_dir = osp.join(results_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_png = osp.join(out_dir, "BEST_PSNR_EDGE_patch_DI.png")

    # ✅ 박스 위치=EDGE only, DI 표시, patch 크기=network_g.img_size
    save_and_show_edge_patch(best_global, out_png, patch=vis_patch, stride=8)


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_best_edge_DI_pipeline(root_path)
