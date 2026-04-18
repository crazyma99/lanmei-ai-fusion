import os
import uuid
import tempfile
import time
import threading
import cv2
import numpy as np
import httpx
from dataclasses import dataclass

from config import MAX_RESOLUTION, URL_DOWNLOAD_TIMEOUT, URL_MAX_SIZE_MB, TEMP_FILE_MAX_AGE_HOURS
from core.segmentation import (
    bisenet_parse, bisenet_face_mask, bisenet_person_mask,
    modnet_alpha_matte, inpaint_region,
)
from core.alignment import compute_alignment, warp_image, warp_mask
from core.color_transfer import color_transfer
from core.blending import laplacian_blend, poisson_blend, final_composite
from core.watermark import apply_watermark


@dataclass
class FusionResult:
    success: bool
    message: str
    images: dict  # {key: {"bgr": ndarray, "rgb": ndarray, "png": path, "jpg": path, "watermark": path}}


def _validate_url(url: str):
    """SSRF 防护：只允许 http/https，禁止内网地址"""
    from urllib.parse import urlparse
    import ipaddress
    import socket

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"不支持的协议: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("无效的 URL")
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("禁止访问内网地址")
    except socket.gaierror:
        raise ValueError(f"无法解析域名: {hostname}")


def download_image_from_url(url: str) -> np.ndarray:
    _validate_url(url)
    max_bytes = URL_MAX_SIZE_MB * 1024 * 1024
    with httpx.Client(timeout=URL_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        # 流式下载，限制大小
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            # 先检查 Content-Length
            cl = resp.headers.get("content-length")
            if cl and int(cl) > max_bytes:
                raise ValueError(f"图片大小 ({int(cl) // 1024 // 1024}MB) 超过限制")
            chunks = []
            total = 0
            for chunk in resp.iter_bytes(8192):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"图片下载超过 {URL_MAX_SIZE_MB}MB 限制")
                chunks.append(chunk)
        data = b"".join(chunks)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片")
    return img


def _resize_to_max(img: np.ndarray, max_px: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_px:
        return img
    scale = max_px / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


def _save_results(img_bgr: np.ndarray, tag: str) -> dict:
    temp_dir = tempfile.gettempdir()
    uid = uuid.uuid4().hex[:8]

    png_path = os.path.join(temp_dir, f"fusion_{tag}_{uid}.png")
    cv2.imwrite(png_path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    jpg_path = os.path.join(temp_dir, f"fusion_{tag}_{uid}.jpg")
    quality = 95
    while quality > 50:
        cv2.imwrite(jpg_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if os.path.getsize(jpg_path) < 10 * 1024 * 1024:
            break
        quality -= 5

    wm_img = apply_watermark(img_bgr)
    wm_path = os.path.join(temp_dir, f"fusion_{tag}_{uid}_wm.jpg")
    cv2.imwrite(wm_path, wm_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {"png": png_path, "jpg": jpg_path, "watermark": wm_path}


def _sharpen_warped(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.addWeighted(img, 0.8, sharpened, 0.2, 0)


def _clean_mask(mask_warped: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(mask_warped, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask


# ============================================================
# 方案 A：BiSeNet + 泊松融合（原方案 - 主方案）
# ============================================================

def _run_bisenet_pipeline(
    img1_bgr: np.ndarray, img2: np.ndarray,
    M_affine: np.ndarray, target_size: tuple,
    progress_cb,
) -> tuple[np.ndarray | None, str]:
    """原方案：BiSeNet 分割 + 色彩迁移 + 泊松融合 + 边缘羽化"""
    # BiSeNet 分割图1
    progress_cb(0.35, "BiSeNet 分割...")
    parsing1 = bisenet_parse(img1_bgr)
    face_mask = bisenet_face_mask(parsing1)
    person_mask = bisenet_person_mask(parsing1)

    # 面部保护并入 person_mask
    if face_mask is not None and np.count_nonzero(face_mask) > 0:
        k_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        face_safe = cv2.dilate(face_mask, k_safe, iterations=2)
        person_mask = cv2.bitwise_or(person_mask, face_safe)

    # 底图 inpaint
    parsing2 = bisenet_parse(img2)
    bg_face = bisenet_face_mask(parsing2)
    img2_clean, _ = inpaint_region(img2, bg_face)

    # Warp
    progress_cb(0.45, "原方案变换...")
    warped_bgr = _sharpen_warped(warp_image(img1_bgr, M_affine, target_size))
    warped_alpha = warp_mask(person_mask, M_affine, target_size)
    mask_binary = _clean_mask(warped_alpha)

    face_mask_warped = None
    if face_mask is not None:
        face_mask_warped = warp_mask(face_mask, M_affine, target_size)

    if np.count_nonzero(mask_binary > 0) == 0:
        return None, "对齐后图像在视野外"

    # 色彩迁移
    progress_cb(0.55, "色彩迁移...")
    warped_transferred = color_transfer(warped_bgr, img2_clean, mask_binary)

    # 泊松融合
    progress_cb(0.65, "泊松融合...")
    seamless = poisson_blend(warped_transferred, img2_clean, mask_binary, face_mask_warped)

    # 最终合成
    progress_cb(0.75, "边缘羽化...")
    final = final_composite(warped_transferred, seamless, img2_clean, mask_binary, face_mask_warped)

    return final, "泊松融合完成"


# ============================================================
# 方案 B：MODNet + 分频融合（新方案 - 备选）
# ============================================================

def _edge_color_match(src: np.ndarray, tgt: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    在融合边缘的环带区域采样图2的色调，把图1整体色调对齐过去。

    原理：
    - 在 alpha 0.05~0.4 的环带区域（融合过渡带），同时采样 src 和 tgt 的 LAB 统计值
    - 用这个环带的统计差异来匹配 src 的整个融合区域
    - 环带是两张图"交汇"的地方，在这里匹配色调最有效
    """
    # 构建边缘环带 mask
    alpha_f = alpha.astype(np.float32) / 255.0
    edge_band = ((alpha_f >= 0.05) & (alpha_f <= 0.4)).astype(np.uint8)
    fg_region = (alpha_f > 0.05).astype(np.uint8)

    edge_count = np.count_nonzero(edge_band)
    fg_count = np.count_nonzero(fg_region)
    if edge_count < 200 or fg_count < 200:
        return src

    # ROI bounding box
    y_idx, x_idx = np.where(fg_region > 0)
    pad = 20
    y_min = max(0, y_idx.min() - pad)
    y_max = min(src.shape[0], y_idx.max() + 1 + pad)
    x_min = max(0, x_idx.min() - pad)
    x_max = min(src.shape[1], x_idx.max() + 1 + pad)

    src_roi = src[y_min:y_max, x_min:x_max]
    tgt_roi = tgt[y_min:y_max, x_min:x_max]
    edge_roi = edge_band[y_min:y_max, x_min:x_max]
    fg_roi = fg_region[y_min:y_max, x_min:x_max]

    src_lab = cv2.cvtColor(src_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 在环带区域采样两边的统计值
    s_mean, s_std = cv2.meanStdDev(src_lab, mask=edge_roi)
    t_mean, t_std = cv2.meanStdDev(tgt_lab, mask=edge_roi)
    s_mean, s_std = s_mean.flatten(), s_std.flatten()
    t_mean, t_std = t_mean.flatten(), t_std.flatten()

    # LAB 匹配（在整个前景区域应用）
    res_lab = src_lab.copy()
    for i in range(3):
        scale = np.clip(t_std[i] / (s_std[i] + 1e-5), 0.8, 1.2)
        shifted = (src_lab[:, :, i] - s_mean[i]) * scale + t_mean[i]
        # 保守混合：50% 迁移
        res_lab[:, :, i] = shifted * 0.5 + src_lab[:, :, i] * 0.5

    res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
    matched_roi = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

    # 用 alpha 渐变应用（alpha 越高的区域匹配越强）
    alpha_roi = alpha[y_min:y_max, x_min:x_max].astype(np.float32) / 255.0
    alpha_roi_3 = alpha_roi[:, :, np.newaxis]

    result = src.copy()
    roi = result[y_min:y_max, x_min:x_max]
    blended = (matched_roi.astype(np.float32) * alpha_roi_3 +
               roi.astype(np.float32) * (1.0 - alpha_roi_3))
    result[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _build_radial_mask(
    face_mask_warped: np.ndarray,
    warped_alpha: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    从面部中心构建径向渐变 alpha mask。

    面部中心 alpha=1，向外平滑衰减到 0，形成圆形过渡。
    比垂直切线自然得多，不会在肩膀处产生水平色差带。

    公式：radial_alpha = smoothstep(1 - dist / radius)
    其中 dist 是到面部中心的欧氏距离，radius 是融合半径。
    """
    # 从 warped face_mask 找面部中心
    y_idx, x_idx = np.where(face_mask_warped > 128)
    if len(y_idx) == 0:
        # 没检测到面部，回退到 warped_alpha 的重心
        y_idx, x_idx = np.where(warped_alpha > 50)
        if len(y_idx) == 0:
            return warped_alpha

    cy = int(y_idx.mean())
    cx = int(x_idx.mean())

    # 面部区域的大小决定融合半径
    face_h = y_idx.max() - y_idx.min()
    face_w = x_idx.max() - x_idx.min()
    face_size = max(face_h, face_w)

    # 融合半径 = 面部大小的 1.8 倍（覆盖面部 + 头发 + 脖子上部）
    radius = face_size * 1.8
    if radius < 50:
        radius = 50

    # 构建距离场
    yy, xx = np.mgrid[:target_h, :target_w]
    dist = np.sqrt((yy - cy).astype(np.float32) ** 2 + (xx - cx).astype(np.float32) ** 2)

    # 归一化到 [0, 1]，距离越近值越大
    t = np.clip(1.0 - dist / radius, 0, 1)
    # smoothstep: 3t² - 2t³，让过渡更柔和
    radial = t * t * (3.0 - 2.0 * t)

    # 与 MODNet alpha 取交集：只在人像区域内生效
    alpha_norm = warped_alpha.astype(np.float32) / 255.0
    fusion_alpha = radial * alpha_norm

    return (np.clip(fusion_alpha, 0, 1) * 255).astype(np.uint8)


def _run_modnet_pipeline(
    img1_bgr: np.ndarray, img2: np.ndarray,
    M_affine: np.ndarray, target_size: tuple,
    progress_cb,
) -> tuple[np.ndarray | None, str]:
    """新方案：MODNet alpha + 面部径向渐变 + 分频融合"""
    h2, w2 = target_size[1], target_size[0]

    # MODNet 人像分割
    progress_cb(0.35, "MODNet 分割...")
    alpha1 = modnet_alpha_matte(img1_bgr)

    # BiSeNet 检测面部中心（仅用于定位，不用于 mask）
    progress_cb(0.4, "检测面部位置...")
    parsing1 = bisenet_parse(img1_bgr)
    face_mask1 = bisenet_face_mask(parsing1)

    # Warp
    progress_cb(0.5, "新方案变换...")
    warped_bgr = _sharpen_warped(warp_image(img1_bgr, M_affine, target_size))
    warped_alpha = cv2.warpAffine(alpha1, M_affine, target_size, flags=cv2.INTER_LINEAR)
    # 内缩 2px 去除 warp 边缘的黑线（边缘 alpha > 0 但对应像素是黑色填充）
    erode_k = np.ones((5, 5), np.uint8)
    warped_alpha = cv2.erode(warped_alpha, erode_k, iterations=1)
    face_mask_warped = warp_mask(face_mask1, M_affine, target_size) if face_mask1 is not None else None

    if np.count_nonzero(warped_alpha > 10) == 0:
        return None, "对齐后图像在视野外"

    # 构建径向融合 mask
    progress_cb(0.6, "构建径向融合区域...")
    if face_mask_warped is not None and np.count_nonzero(face_mask_warped > 128) > 0:
        fusion_alpha = _build_radial_mask(face_mask_warped, warped_alpha, h2, w2)
    else:
        fusion_alpha = warped_alpha

    # 色调预匹配
    progress_cb(0.65, "色调预匹配...")
    warped_matched = _edge_color_match(warped_bgr, img2, fusion_alpha)

    # 直接 alpha blending — MODNet 的连续 alpha + 径向渐变已经足够平滑
    # 不走拉普拉斯金字塔，避免金字塔分解/重建在各频段引入色调偏移
    progress_cb(0.7, "Alpha 融合...")
    alpha_f = fusion_alpha.astype(np.float32) / 255.0
    alpha_3 = alpha_f[:, :, np.newaxis]
    final = (warped_matched.astype(np.float32) * alpha_3 +
             img2.astype(np.float32) * (1.0 - alpha_3))
    final = np.clip(final, 0, 255).astype(np.uint8)

    return final, "Alpha融合(径向)"


# ============================================================
# 主流水线
# ============================================================

def process_fusion(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    progress_callback=None,
) -> FusionResult:
    """双算法人像融合流水线，同时输出两种结果"""

    def _progress(val, desc=""):
        if progress_callback:
            progress_callback(val, desc=desc)

    _progress(0, "正在初始化...")

    # 底图分辨率限制
    img2 = _resize_to_max(img2_bgr, MAX_RESOLUTION)
    h2, w2 = img2.shape[:2]
    target_size = (w2, h2)

    # SIFT 对齐（两个方案共用）
    _progress(0.1, "SIFT 特征对齐...")
    M_affine, err = compute_alignment(img1_bgr, img2)
    if M_affine is None:
        return FusionResult(success=False, message=err, images={})

    results = {}
    msgs = []

    # 方案 A：BiSeNet + 泊松（主方案）
    _progress(0.3, "运行主方案...")
    try:
        final_a, msg_a = _run_bisenet_pipeline(img1_bgr, img2, M_affine, target_size, _progress)
        if final_a is not None:
            out_h, out_w = final_a.shape[:2]
            msgs.append(f"方案A({msg_a}) {out_w}x{out_h}")
            paths = _save_results(final_a, "bisenet")
            results["algo_a"] = {"bgr": final_a, "rgb": cv2.cvtColor(final_a, cv2.COLOR_BGR2RGB), **paths}
        else:
            msgs.append(f"方案A失败: {msg_a}")
    except Exception as e:
        msgs.append(f"方案A异常: {e}")

    # 方案 B：MODNet + 分频（备选）
    _progress(0.8, "运行备选方案...")
    try:
        final_b, msg_b = _run_modnet_pipeline(img1_bgr, img2, M_affine, target_size, _progress)
        if final_b is not None:
            out_h, out_w = final_b.shape[:2]
            msgs.append(f"方案B({msg_b}) {out_w}x{out_h}")
            paths = _save_results(final_b, "modnet")
            results["algo_b"] = {"bgr": final_b, "rgb": cv2.cvtColor(final_b, cv2.COLOR_BGR2RGB), **paths}
        else:
            msgs.append(f"方案B失败: {msg_b}")
    except Exception as e:
        msgs.append(f"方案B异常: {e}")

    _progress(1.0, "全部完成！")

    if not results:
        return FusionResult(success=False, message=" | ".join(msgs), images={})

    return FusionResult(success=True, message=" | ".join(msgs), images=results)


# === 临时文件清理 ===
def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    cutoff = time.time() - TEMP_FILE_MAX_AGE_HOURS * 3600
    count = 0
    for f in os.listdir(temp_dir):
        if f.startswith("fusion_") and (f.endswith(".png") or f.endswith(".jpg")):
            path = os.path.join(temp_dir, f)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    count += 1
            except OSError:
                pass
    if count > 0:
        print(f"已清理 {count} 个过期临时文件")


def start_cleanup_timer():
    def _loop():
        while True:
            time.sleep(3600)
            cleanup_temp_files()
    threading.Thread(target=_loop, daemon=True).start()
