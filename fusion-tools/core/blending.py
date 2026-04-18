import cv2
import numpy as np


def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """构建高斯金字塔（逐层下采样模糊）"""
    pyramid = [img.astype(np.float32)]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _build_laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """构建拉普拉斯金字塔（每层保存该频段细节）"""
    gaussian = _build_gaussian_pyramid(img, levels)
    laplacian = []
    for i in range(levels):
        h, w = gaussian[i].shape[:2]
        upsampled = cv2.pyrUp(gaussian[i + 1], dstsize=(w, h))
        laplacian.append(gaussian[i] - upsampled)
    laplacian.append(gaussian[-1])
    return laplacian


def _build_mask_pyramid(
    mask: np.ndarray, levels: int, face_mask: np.ndarray | None = None
) -> list[np.ndarray]:
    """
    构建面部感知的多频段融合 mask 金字塔。

    频段策略：
    - 低频层（顶部 2 层）：色调/光照 → mask 高度模糊，让背景光照渗入面部
      面部区域在低频层的 mask 要适度降低，让背景的亮度和色温影响面部，
      否则面部会保持原图光照导致对比度异常。
    - 中频层（中间层）：皮肤纹理 → 正常高斯金字塔过渡
    - 高频层（底部 2 层）：毛发/五官边缘 → 面部区域适度提升 mask，
      但不能太激进（原来 1.2x 太高），否则会带入前景的局部对比度。
    """
    mask_f = mask.astype(np.float32) / 255.0
    if mask_f.ndim == 2:
        mask_f = np.stack([mask_f] * 3, axis=2)

    # 高斯金字塔 + 每层用原始 alpha 下采样做 clamp
    # 防止 pyrDown 的高斯扩散把 mask 的非零区域"泄漏"到 alpha=0 的边缘外
    # 这是消除分频融合边缘色差晕染的关键
    pyramid = [mask_f]
    clamp = mask_f.copy()
    for _ in range(levels):
        raw_down = cv2.pyrDown(pyramid[-1])
        clamp = cv2.pyrDown(clamp)
        # clamp: 每层 mask 不能超过该层原始 alpha 的下采样值
        # 这截断了高斯扩散，确保 alpha=0 的区域在所有频段都严格为 0
        clamped = np.minimum(raw_down, clamp)
        pyramid.append(clamped)

    if face_mask is not None:
        face_f = face_mask.astype(np.float32) / 255.0
        if face_f.ndim == 2:
            face_f = np.stack([face_f] * 3, axis=2)

        # 轻度膨胀面部区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        for c in range(3):
            face_f[:, :, c] = cv2.dilate(face_f[:, :, c], kernel, iterations=1)

        face_pyr = [face_f]
        for _ in range(levels):
            face_pyr.append(cv2.pyrDown(face_pyr[-1]))

        for i in range(len(pyramid)):
            face_weight = np.clip(face_pyr[i], 0, 1)
            if i <= 1:
                # 高频层：温和提升面部纹理保留
                pyramid[i] = np.clip(pyramid[i] + face_weight * 0.1, 0, 1)
            elif i == levels:
                # 仅最顶层（最低频，即全局色调）：轻微降低面部 mask
                # 让背景的整体色温微量渗入，但不影响面部不透明度
                pyramid[i] = pyramid[i] * (1.0 - face_weight * 0.1)

    return pyramid


def _reconstruct_from_laplacian(pyramid: list[np.ndarray]) -> np.ndarray:
    """从拉普拉斯金字塔重建完整图像"""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[:2]
        img = cv2.pyrUp(img, dstsize=(w, h)) + pyramid[i]
    return img


def laplacian_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    face_mask: np.ndarray | None = None,
    levels: int = 5,
) -> np.ndarray:
    """
    拉普拉斯金字塔分频融合。

    将图像分解到不同频段：
    - 低频（色调/光照）→ 背景主导，自然消除色差
    - 中频（皮肤纹理）→ 平滑过渡
    - 高频（毛发/锐边）→ 前景主导，保护面部细节

    一步完成色调匹配 + 边缘过渡 + 面部保护。
    """
    h, w = background.shape[:2]
    if foreground.shape[:2] != (h, w):
        foreground = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # padding 确保尺寸能被 2^levels 整除
    factor = 2 ** levels
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        foreground = cv2.copyMakeBorder(foreground, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        background = cv2.copyMakeBorder(background, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        if face_mask is not None:
            face_mask = cv2.copyMakeBorder(face_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    fg_lap = _build_laplacian_pyramid(foreground, levels)
    bg_lap = _build_laplacian_pyramid(background, levels)
    mask_pyr = _build_mask_pyramid(mask, levels, face_mask)

    blended_lap = []
    for fg_l, bg_l, m_l in zip(fg_lap, bg_lap, mask_pyr):
        if m_l.ndim == 2 and fg_l.ndim == 3:
            m_l = np.stack([m_l] * 3, axis=2)
        elif m_l.shape[:2] != fg_l.shape[:2]:
            m_l = cv2.resize(m_l, (fg_l.shape[1], fg_l.shape[0]))
            if m_l.ndim == 2:
                m_l = np.stack([m_l] * 3, axis=2)
        blended_lap.append(fg_l * m_l + bg_l * (1.0 - m_l))

    result = np.clip(_reconstruct_from_laplacian(blended_lap), 0, 255).astype(np.uint8)

    if pad_h > 0 or pad_w > 0:
        result = result[:h, :w]

    return result


# ============================================================
# 泊松融合（原方案）
# ============================================================

def poisson_blend(
    warped_src: np.ndarray,
    target: np.ndarray,
    mask_binary: np.ndarray,
    face_mask_warped: np.ndarray | None,
    mode: str = "poisson_normal",
) -> np.ndarray:
    """泊松融合 + 面部感知 mask + 大图缩图优化"""
    from config import POISSON_MAX_PX
    clone_flag = cv2.NORMAL_CLONE if mode == "poisson_normal" else cv2.MIXED_CLONE

    # 构建 mask：收缩 + 面部保护
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask_binary, kernel, iterations=2)
    if face_mask_warped is not None:
        _, face_bin = cv2.threshold(face_mask_warped, 128, 255, cv2.THRESH_BINARY)
        k_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        face_bin = cv2.dilate(face_bin, k_face, iterations=1)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(mask_binary, face_bin))

    mask[:10, :] = 0
    mask[-10:, :] = 0
    mask[:, :10] = 0
    mask[:, -10:] = 0

    y_idx, x_idx = np.where(mask > 0)
    if len(x_idx) == 0:
        return target

    def _do_clone(src, dst, m):
        yi, xi = np.where(m > 0)
        if len(xi) == 0:
            return dst
        c = ((xi.min() + xi.max()) // 2, (yi.min() + yi.max()) // 2)
        try:
            return cv2.seamlessClone(src, dst, m, c, clone_flag)
        except Exception:
            mi = cv2.bitwise_not(m)
            return cv2.add(cv2.bitwise_and(dst, dst, mask=mi), cv2.bitwise_and(src, src, mask=m))

    h, w = target.shape[:2]
    if max(h, w) > POISSON_MAX_PX:
        # 缩图做泊松，再上采样混合回原图
        scale = POISSON_MAX_PX / max(h, w)
        sw, sh = int(w * scale), int(h * scale)
        src_s = cv2.resize(warped_src, (sw, sh), interpolation=cv2.INTER_AREA)
        dst_s = cv2.resize(target, (sw, sh), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        blended_s = _do_clone(src_s, dst_s, mask_s)
        blended_up = cv2.resize(blended_s, (w, h), interpolation=cv2.INTER_LANCZOS4)
        # 只在 mask 区域使用泊松结果，非融合区域保持原图清晰度
        mask_f = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
        mask_3 = mask_f[:, :, np.newaxis]
        return np.clip(blended_up.astype(np.float32) * mask_3 + target.astype(np.float32) * (1 - mask_3), 0, 255).astype(np.uint8)

    return _do_clone(warped_src, target, mask)


def final_composite(
    warped_src: np.ndarray,
    seamless: np.ndarray,
    target: np.ndarray,
    mask_binary: np.ndarray,
    face_mask_warped: np.ndarray | None,
) -> np.ndarray:
    """原方案最终合成：边缘羽化 + 面部保护"""
    h, w = target.shape[:2]

    # 边缘羽化
    kernel_feather = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_main = cv2.erode(mask_binary, kernel_feather, iterations=3)

    feather_amount = max(int(min(w, h) * 0.008), 11)
    if feather_amount % 2 == 0:
        feather_amount += 1
    mask_feathered = cv2.GaussianBlur(mask_main, (feather_amount, feather_amount), 0)

    # 核心保护
    kernel_size = max(int(min(w, h) * 0.02), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask_eroded = cv2.erode(mask_main, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

    if face_mask_warped is not None:
        _, face_bin = cv2.threshold(face_mask_warped, 128, 255, cv2.THRESH_BINARY)
        k_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        face_bin = cv2.dilate(face_bin, k_face, iterations=1)
        mask_eroded = cv2.bitwise_or(mask_eroded, face_bin)

    blur_r = max(int(min(w, h) * 0.08), 3)
    if blur_r % 2 == 0:
        blur_r += 1
    mask_protect = cv2.GaussianBlur(mask_eroded, (blur_r, blur_r), 0)

    alpha_edge = (mask_feathered.astype(np.float32) / 255.0)[:, :, np.newaxis]
    alpha_protect = (mask_protect.astype(np.float32) / 255.0)[:, :, np.newaxis]
    core = (alpha_protect >= 0.6).astype(np.float32)
    alpha_protect = alpha_protect * (1.0 - core) + core

    src_f = warped_src.astype(np.float32)
    sea_f = seamless.astype(np.float32)
    tgt_f = target.astype(np.float32)

    inner = src_f * alpha_protect + sea_f * (1.0 - alpha_protect)
    result = inner * alpha_edge + tgt_f * (1.0 - alpha_edge)
    return np.clip(result, 0, 255).astype(np.uint8)
