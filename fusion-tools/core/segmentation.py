import cv2
import numpy as np
from config import FACE_IDS, PERSON_IDS, MODNET_REF_SIZE
from core.model_manager import ModelManager

# ============================================================
# BiSeNet 路径（原方案）
# ============================================================

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_FACE_IDS = np.array(FACE_IDS, dtype=np.uint8)
_PERSON_IDS = np.array(PERSON_IDS, dtype=np.uint8)


def bisenet_parse(img_bgr: np.ndarray) -> np.ndarray:
    """BiSeNet 单次推理，返回原尺寸 parsing map"""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_norm = (img_resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    img_chw = img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    manager = ModelManager()
    out = manager.bisenet.infer(img_chw)
    if out.ndim == 4:
        parsing_512 = out[0].argmax(0).astype(np.uint8)
    else:
        parsing_512 = out.argmax(0).astype(np.uint8)
    return cv2.resize(parsing_512, (w, h), interpolation=cv2.INTER_NEAREST)


def bisenet_face_mask(parsing: np.ndarray) -> np.ndarray:
    """从 parsing 提取面部 mask"""
    mask = np.isin(parsing, _FACE_IDS).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    return cv2.dilate(mask, kernel, iterations=2)


def bisenet_person_mask(parsing: np.ndarray) -> np.ndarray:
    """从 parsing 提取人体前景 mask"""
    mask = np.isin(parsing, _PERSON_IDS).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


# ============================================================
# MODNet 路径（新方案）
# ============================================================

def _modnet_scale_factor(im_h: int, im_w: int, ref_size: int):
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh, im_rw = im_h, im_w
    im_rw = max(im_rw - im_rw % 32, 32)
    im_rh = max(im_rh - im_rh % 32, 32)
    return im_rw / im_w, im_rh / im_h


def modnet_alpha_matte(img_bgr: np.ndarray) -> np.ndarray:
    """MODNet 人像抠图，返回 (H, W) uint8 alpha [0, 255]"""
    im_h, im_w = img_bgr.shape[:2]
    im = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    im = (im - 127.5) / 127.5

    x_scale, y_scale = _modnet_scale_factor(im_h, im_w, MODNET_REF_SIZE)
    im_resized = cv2.resize(im, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA)
    im_tensor = im_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    manager = ModelManager()
    matte = np.squeeze(manager.modnet.infer(im_tensor))
    matte = cv2.resize(matte, (im_w, im_h), interpolation=cv2.INTER_AREA)
    return (np.clip(matte, 0, 1) * 255).astype(np.uint8)


# ============================================================
# 通用工具
# ============================================================

def inpaint_region(img_bgr: np.ndarray, mask: np.ndarray):
    """用 mask 对图像做内容填充"""
    if mask is None or np.count_nonzero(mask) == 0:
        return img_bgr, None
    binary = (mask > 50).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    binary = cv2.dilate(binary, kernel, iterations=1)
    filled = cv2.inpaint(img_bgr, binary, 3, cv2.INPAINT_TELEA)
    return filled, binary
