import cv2
import numpy as np
from config import SIFT_FEATURES, SIFT_CONTRAST_THRESHOLD, SIFT_MATCH_RATIO, MIN_GOOD_MATCHES, ALIGNMENT_MAX_PX


def _downscale_for_matching(img: np.ndarray, max_px: int):
    """将图像缩小到 max_px 以加速 SIFT 计算，返回 (缩小后图像, 缩放比例)"""
    h, w = img.shape[:2]
    if max(h, w) <= max_px:
        return img, 1.0
    scale = max_px / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def compute_alignment(img_src: np.ndarray, img_dst: np.ndarray):
    """
    计算 img_src → img_dst 的相似变换矩阵（2x3 仿射）。

    优化点：
    1. 在缩小图上做 SIFT + FLANN 匹配，然后将变换矩阵还原到原图尺度
    2. nfeatures 从 8000 降到 4000
    3. 直接返回 2x3 仿射矩阵，不升维到 3x3

    返回: (M_affine_2x3, error_msg)
    - 成功: (ndarray, None)
    - 失败: (None, error_string)
    """
    # 缩小图像用于特征匹配
    src_small, scale_src = _downscale_for_matching(img_src, ALIGNMENT_MAX_PX)
    dst_small, scale_dst = _downscale_for_matching(img_dst, ALIGNMENT_MAX_PX)

    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=SIFT_CONTRAST_THRESHOLD)
    kp1, des1 = sift.detectAndCompute(src_small, None)
    kp2, des2 = sift.detectAndCompute(dst_small, None)

    if des1 is None or des2 is None:
        return None, "特征描述符提取失败，图像可能缺乏纹理。"

    # FLANN 匹配
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),  # FLANN_INDEX_KDTREE
        dict(checks=50),
    )
    matches = flann.knnMatch(des1, des2, k=2)

    good = [m for match in matches if len(match) == 2 for m, n in [match] if m.distance < SIFT_MATCH_RATIO * n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        return None, f"有效匹配点太少 ({len(good)})，无法精准对齐。"

    # 获取匹配点（在缩小图坐标系中）
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 还原到原图坐标
    src_pts /= scale_src
    dst_pts /= scale_dst

    M_affine, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M_affine is None:
        return None, "无法计算对齐矩阵，请确保图片内容足够接近。"

    # 检查缩放比例是否异常
    scale_x = np.sqrt(M_affine[0, 0] ** 2 + M_affine[0, 1] ** 2)
    if scale_x < 0.1 or scale_x > 10.0:
        return None, f"检测到异常缩放比例 ({scale_x:.2f})，对齐可能失效。"

    return M_affine, None


def warp_image(img: np.ndarray, M_affine: np.ndarray, target_size: tuple) -> np.ndarray:
    """使用仿射矩阵变换图像。target_size = (width, height)"""
    return cv2.warpAffine(img, M_affine, target_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def warp_mask(mask: np.ndarray, M_affine: np.ndarray, target_size: tuple) -> np.ndarray:
    """使用仿射矩阵变换 mask（使用最近邻插值保持二值性）"""
    return cv2.warpAffine(mask, M_affine, target_size, flags=cv2.INTER_NEAREST)
