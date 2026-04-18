import cv2
import numpy as np


def color_transfer(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LAB 色彩迁移（ROI 优化版）"""
    # ROI bounding box
    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) < 100:
        return source

    pad = 10
    y_min = max(0, y_idx.min() - pad)
    y_max = min(source.shape[0], y_idx.max() + 1 + pad)
    x_min = max(0, x_idx.min() - pad)
    x_max = min(source.shape[1], x_idx.max() + 1 + pad)

    src_roi = source[y_min:y_max, x_min:x_max]
    tgt_roi = target[y_min:y_max, x_min:x_max]
    mask_roi = mask[y_min:y_max, x_min:x_max]

    kernel_stat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_stat = cv2.erode(mask_roi, kernel_stat, iterations=2)

    if np.count_nonzero(mask_stat) < 100:
        return source

    src_lab = cv2.cvtColor(src_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    s_mean, s_std = cv2.meanStdDev(src_lab, mask=mask_stat)
    t_mean, t_std = cv2.meanStdDev(tgt_lab, mask=mask_stat)
    s_mean, s_std = s_mean.flatten(), s_std.flatten()
    t_mean, t_std = t_mean.flatten(), t_std.flatten()

    res_lab = src_lab.copy()

    l_scale = np.clip(t_std[0] / (s_std[0] + 1e-5), 0.8, 1.2)
    res_lab[:, :, 0] = (src_lab[:, :, 0] - s_mean[0]) * l_scale + t_mean[0]

    for i in range(1, 3):
        ab_scale = np.clip(t_std[i] / (s_std[i] + 1e-5), 0.5, 1.5)
        transferred = (src_lab[:, :, i] - s_mean[i]) * ab_scale + t_mean[i]
        res_lab[:, :, i] = transferred * 0.7 + src_lab[:, :, i] * 0.3

    res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
    transfer_roi = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

    result = source.copy()
    roi_mask = mask_roi > 0
    result[y_min:y_max, x_min:x_max][roi_mask] = transfer_roi[roi_mask]
    return result
