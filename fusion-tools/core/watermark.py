import cv2
import numpy as np
from config import WATERMARK_TEXT, WATERMARK_ALPHA, WATERMARK_ANGLE


def apply_watermark(
    img_bgr: np.ndarray,
    text: str = WATERMARK_TEXT,
    alpha: float = WATERMARK_ALPHA,
    angle: float = WATERMARK_ANGLE,
) -> np.ndarray:
    """在图像上叠加斜向平铺水印"""
    h, w = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(max(w, h) / 1600.0, 0.8)
    thickness = max(int(max(w, h) / 900), 1)

    # 放大画布用于旋转
    scale_canvas = 1.6
    big_h, big_w = int(h * scale_canvas), int(w * scale_canvas)
    overlay = np.zeros((big_h, big_w, 3), dtype=img_bgr.dtype)

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    tw, th = text_size
    step_x = max(int(tw * 1.2), 80)
    step_y = max(int(th * 2.2), 60)

    for y in range(0, big_h + step_y, step_y):
        for x in range(-big_w // 2, big_w + step_x, step_x):
            cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    center = (big_w // 2, big_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    overlay = cv2.warpAffine(overlay, rot_mat, (big_w, big_h))

    # 裁剪回原尺寸
    y0, x0 = (big_h - h) // 2, (big_w - w) // 2
    overlay = overlay[y0 : y0 + h, x0 : x0 + w]

    # 只在水印文字区域叠加，不影响无文字区域的原始亮度
    text_mask = (overlay.max(axis=2) > 0).astype(np.float32)[:, :, np.newaxis]
    result = img_bgr.astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    result = result * (1.0 - text_mask * alpha) + overlay_f * (text_mask * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)
