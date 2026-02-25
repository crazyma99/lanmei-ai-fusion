import gradio as gr
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import onnxruntime as ort

# 尝试导入硬件加速库
try:
    import torch
    # 兼容性检测：优先检测 CUDA (ModelScope 常用)，再检测 MPS (本地 Mac)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("ModelScope: 检测到 CUDA，启用 GPU 加速")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Local: 检测到 MPS，启用 Mac GPU 加速")
    else:
        DEVICE = torch.device("cpu")
        print("未检测到兼容 GPU，使用 CPU 处理")
except ImportError:
    torch = None
    class MockDevice:
        def __init__(self):
            self.type = "cpu"
    DEVICE = MockDevice()
    print("未安装 torch，默认使用 CPU 处理")

device_type = getattr(DEVICE, "type", "cpu")
DEVICE_LABEL = {
    "cuda": "CUDA(GPU)",
    "mps": "MPS(GPU)",
    "cpu": "CPU"
}.get(device_type, str(device_type).upper())

FACE_PARSER = None
BISE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../", "models", "bisenet")
BISE_MODEL_PATH = os.path.join(BISE_MODEL_DIR, "bisenet_face.pth")
BISE_ONNX_PATH = os.path.join(BISE_MODEL_DIR, "resnet18.onnx")
FACE_ONNX_SESSION = None

def get_face_onnx_session():
    global FACE_ONNX_SESSION
    if FACE_ONNX_SESSION is not None:
        return FACE_ONNX_SESSION
    if not os.path.exists(BISE_ONNX_PATH):
        return None
    providers = ["CPUExecutionProvider"]
    if device_type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(BISE_ONNX_PATH, providers=providers)
    except Exception as e:
        print(f"加载 BiSeNet ONNX 模型失败: {e}")
        return None
    FACE_ONNX_SESSION = sess
    print(f"已从本地 ONNX 加载 BiSeNet: {BISE_ONNX_PATH}")
    return sess

def get_face_parser():
    global FACE_PARSER
    if FACE_PARSER is not None:
        return FACE_PARSER
    if torch is None:
        return None
    model = None
    if os.path.exists(BISE_MODEL_PATH):
        try:
            model = torch.hub.load("valgur/face-parsing.PyTorch", "bisenet", pretrained=False)
        except Exception as e:
            print(f"加载本地 BiSeNet 结构失败（构建网络时出错）: {e}")
            return None
        try:
            state = torch.load(BISE_MODEL_PATH, map_location="cpu")
            model.load_state_dict(state)
            print(f"已从本地权重加载 BiSeNet: {BISE_MODEL_PATH}")
        except Exception as e:
            print(f"加载本地 BiSeNet 权重失败: {e}")
            return None
    else:
        try:
            model = torch.hub.load("valgur/face-parsing.PyTorch", "bisenet", pretrained=True)
        except Exception as e:
            print(f"从远程加载 BiSeNet 面部分割模型失败: {e}")
            return None
        try:
            os.makedirs(BISE_MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), BISE_MODEL_PATH)
            print(f"BiSeNet 权重已保存到本地: {BISE_MODEL_PATH}")
        except Exception as e:
            print(f"保存 BiSeNet 权重到本地失败: {e}")
    model.to(DEVICE)
    model.eval()
    FACE_PARSER = model
    return model

def get_face_mask_bisenet(img_bgr):
    parsing = get_bisenet_parsing(img_bgr)
    if parsing is None:
        return None
    face_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17], dtype=np.uint8)
    mask = np.isin(parsing, face_ids).astype("uint8") * 255
    face_pixels = int(np.count_nonzero(mask))
    total_pixels = mask.size
    print(f"BiSeNet face mask coverage: {face_pixels}/{total_pixels} ({face_pixels / max(total_pixels,1):.4f})")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def get_person_mask_bisenet(img_bgr):
    parsing = get_bisenet_parsing(img_bgr)
    if parsing is None:
        return None
    keep_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 18], dtype=np.uint8)
    mask = np.isin(parsing, keep_ids).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def inpaint_face_region(img_bgr):
    h, w = img_bgr.shape[:2]
    face_mask = get_face_mask_bisenet(img_bgr)
    if face_mask is None or np.count_nonzero(face_mask) == 0:
        print("背景人脸填充：未检测到有效面部区域，跳过 inpaint")
        return img_bgr, None
    mask = (face_mask > 0).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpaint_mask = mask.astype("uint8")
    print(f"背景人脸填充：mask 覆盖像素 {int(np.count_nonzero(inpaint_mask))}/{h*w}")
    filled = cv2.inpaint(img_bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
    return filled, inpaint_mask

def get_bisenet_parsing(img_bgr):
    h, w = img_bgr.shape[:2]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    img_norm = (img_norm - mean) / std
    img_chw = img_norm.transpose(2, 0, 1)

    parsing = None

    sess = get_face_onnx_session()
    if sess is not None:
        try:
            inp = img_chw[None, ...].astype("float32")
            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: inp})[0]
            if out.ndim == 4:
                logits = out[0]
                parsing = logits.argmax(0).astype(np.uint8)
                unique_labels, counts = np.unique(parsing, return_counts=True)
                print("BiSeNet ONNX parsing labels:", dict(zip(unique_labels.tolist(), counts.tolist())))
        except Exception as e:
            print(f"BiSeNet ONNX 推理失败: {e}")

    if parsing is None:
        parser = get_face_parser()
        if parser is None or torch is None:
            return None
        with torch.no_grad():
            tensor = torch.from_numpy(img_chw).unsqueeze(0).to(DEVICE)
            out = parser(tensor)[0]
            parsing = out.argmax(0).cpu().numpy().astype(np.uint8)
        unique_labels, counts = np.unique(parsing, return_counts=True)
        print("BiSeNet Torch parsing labels:", dict(zip(unique_labels.tolist(), counts.tolist())))
    parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
    return parsing

def color_transfer(source, target, mask):
    """
    高级色彩协调 (Harmonization)：模仿 PS 神经网络滤镜效果。
    除了 LAB 均值标准差匹配外，引入了色彩衰减控制，防止过饱和或偏色。
    """
    # 转换到 LAB 空间
    source_lab = cv2.cvtColor(source.astype("uint8"), cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target.astype("uint8"), cv2.COLOR_BGR2LAB).astype("float32")

    # 获取有效区域统计信息
    # 使用稍微收缩的 mask 以获得更纯净的颜色统计
    kernel_stat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_stat = cv2.erode(mask, kernel_stat, iterations=2)
    
    source_pixels = source_lab[mask_stat > 0]
    target_pixels = target_lab[mask_stat > 0]

    if len(source_pixels) < 100 or len(target_pixels) < 100:
        return source

    # 计算均值和标准差
    s_mean, s_std = cv2.meanStdDev(source_lab, mask=mask_stat)
    t_mean, t_std = cv2.meanStdDev(target_lab, mask=mask_stat)
    
    s_mean = s_mean.flatten()
    s_std = s_std.flatten()
    t_mean = t_mean.flatten()
    t_std = t_std.flatten()

    # 色彩协调核心逻辑
    res_lab = source_lab.copy()
    
    # 1. 亮度通道 (L) 匹配：限制对比度剧烈变化，保持柔和
    l_scale = np.clip(t_std[0] / (s_std[0] + 1e-5), 0.8, 1.2)
    res_lab[:,:,0] = (source_lab[:,:,0] - s_mean[0]) * l_scale + t_mean[0]
    
    # 2. 色彩通道 (A, B) 匹配：模仿 Harmonization 的协调感
    # 减弱色彩迁移的强度 (0.7)，让人物保留一部分原始肤色，同时融入背景色调
    for i in range(1, 3):
        ab_scale = np.clip(t_std[i] / (s_std[i] + 1e-5), 0.5, 1.5)
        res_lab[:,:,i] = (source_lab[:,:,i] - s_mean[i]) * ab_scale + t_mean[i]
        # 与原色轻微混合，增加自然感
        res_lab[:,:,i] = res_lab[:,:,i] * 0.7 + source_lab[:,:,i] * 0.3

    res_lab = np.clip(res_lab, 0, 255).astype("uint8")
    transfer = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
    
    # 平滑应用：只在 mask 区域应用
    result = source.copy()
    result[mask > 0] = transfer[mask > 0]
    return result

def blend_images_smart(img_source_nobg, img_target, progress=None, mode="poisson_normal", use_color=True, face_mask=None):
    """
    核心处理流程优化：
    1. SIFT 特征对齐
    2. 图像透视变换
    3. 颜色迁移 (Color Transfer) - 增强稳定性
    4. 智能融合 (结合泊松与 Alpha 混合) - 解决黑影
    5. 核心区域保护 - 解决面部丢失
    """
    if progress: progress(0.4, desc="正在进行特征对齐...")
    # 转换格式
    img1_src_bgr = img_source_nobg[:, :, :3]
    img1_mask = img_source_nobg[:, :, 3] # Alpha通道
    img2_target_bgr = img_target
    
    # 保存原始未对齐的前景，用于最后可能的细节恢复
    img1_orig_bgr = img1_src_bgr.copy()

    # ===========================
    # 1. SIFT 特征对齐 (增强版)
    # ===========================
    print("正在计算特征点 (FLANN 高精度模式)...")
    if progress: progress(0.45, desc="正在计算特征点 (FLANN 高精度模式)...")
    # 增加 nfeatures 并优化对比度阈值
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.03)
    
    # 检测关键点
    kp1, des1 = sift.detectAndCompute(img1_src_bgr, None)
    kp2, des2 = sift.detectAndCompute(img2_target_bgr, None)
    
    if des1 is None or des2 is None:
        return None, "特征描述符提取失败，图像可能缺乏纹理。"

    # 使用 FLANN 匹配器替代 BFMatcher，速度更快且在特征点多时更稳定
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        # 使用 Lowe's ratio test，更加严格的 0.65 过滤
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 20: 
        return None, "有效匹配点太少，无法精准对齐。请检查图片是否属于同一场景。"

    # 获取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    print("正在计算相似变换矩阵 (防止变形)...")
    if progress: progress(0.5, desc="正在计算相似变换矩阵 (防止变形)...")
    M_affine, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if M_affine is None:
        return None, "无法计算对齐矩阵，请确保图片内容足够接近。"

    # 将 2x3 仿射矩阵转为 3x3 矩阵，以便后续 warpPerspective 使用（或者改用 warpAffine）
    M = np.vstack([M_affine, [0, 0, 1]])

    # 检查缩放比例是否异常 (防止缩放过大或过小)
    scale_x = np.sqrt(M[0,0]**2 + M[0,1]**2)
    if scale_x < 0.1 or scale_x > 10.0:
        return None, f"检测到异常缩放比例 ({scale_x:.2f})，对齐可能失效。"

    # ===========================
    # 2. 图像变换 (Warp)
    # ===========================
    h_bg, w_bg = img2_target_bgr.shape[:2]
    print(f"目标分辨率: {w_bg}x{h_bg}")
    if progress: progress(0.55, desc=f"正在进行图像变换 ({w_bg}x{h_bg})...")
    
    # 使用 INTER_LANCZOS4 替代 INTER_CUBIC，它是更高质量的插值算法，适合缩放/旋转
    img1_warped = cv2.warpPerspective(img1_src_bgr, M, (w_bg, h_bg), flags=cv2.INTER_LANCZOS4)
    mask_warped = cv2.warpPerspective(img1_mask, M, (w_bg, h_bg), flags=cv2.INTER_LANCZOS4)
    
    # 锐化处理：补偿插值带来的微量模糊
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img1_warped_sharpened = cv2.filter2D(img1_warped, -1, kernel_sharpen)
    # 仅轻微混合锐化结果，防止噪点过大
    img1_warped = cv2.addWeighted(img1_warped, 0.8, img1_warped_sharpened, 0.2, 0)

    # 清理 Mask (二值化)
    _, mask_binary = cv2.threshold(mask_warped, 128, 255, cv2.THRESH_BINARY)
    
    # 移除 Mask 边缘的噪点，防止 seamlessClone 产生黑边
    kernel_clean = np.ones((5,5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_clean)
    mask_binary = cv2.GaussianBlur(mask_binary, (5, 5), 0)
    _, mask_binary = cv2.threshold(mask_binary, 128, 255, cv2.THRESH_BINARY)

    face_mask_warped = None
    if face_mask is not None:
        try:
            fm = face_mask
            if fm.shape[:2] != img1_src_bgr.shape[:2]:
                fm = cv2.resize(fm, (img1_src_bgr.shape[1], img1_src_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            face_mask_warped = cv2.warpPerspective(fm, M, (w_bg, h_bg), flags=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"BiSeNet 面部 mask 变换失败: {e}")
            face_mask_warped = None

    # ===========================
    # 3. 颜色迁移 (Color Transfer)
    # ===========================
    if use_color:
        print("正在进行颜色迁移...")
        if progress: progress(0.65, desc="正在进行色彩协调 (Harmonization)...")
        # 对 warped 后的图进行颜色迁移，使其匹配背景图在相同区域的色调
        img1_warped_transferred = color_transfer(img1_warped, img2_target_bgr, mask_binary)
    else:
        img1_warped_transferred = img1_warped.copy()

    # ===========================
    # 4. 泊松融合 (消除边缘色差与阴影)
    # ===========================
    print("正在进行边缘无缝融合...")
    if progress: progress(0.8, desc="正在进行泊松融合 (无缝连接)...")
    
    # 找到 Mask 的中心
    y_indices, x_indices = np.where(mask_binary > 0)
    if len(x_indices) == 0:
        return None, "对齐后图像在视野外。"
        
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

    # 关键改进：优化 mask 以消除边缘阴影
    # 1. 稍微收缩 mask，确保不包含原图抠图边缘可能存在的残余背景色
    kernel_shrink = np.ones((3,3), np.uint8)
    mask_for_clone = cv2.erode(mask_binary, kernel_shrink, iterations=2)
    
    # 2. 确保 mask 不触碰图像边界 (seamlessClone 的硬性要求)
    mask_for_clone[0:10, :] = 0
    mask_for_clone[-10:, :] = 0
    mask_for_clone[:, 0:10] = 0
    mask_for_clone[:, -10:] = 0

    seamless = None
    if mode.startswith("poisson"):
        clone_flag = cv2.NORMAL_CLONE if mode == "poisson_normal" else cv2.MIXED_CLONE
        try:
            seamless = cv2.seamlessClone(img1_warped_transferred, img2_target_bgr, mask_for_clone, center, clone_flag)
        except Exception as e:
            print(f"泊松融合失败: {e}，回退到普通混合")
            seamless = img2_target_bgr.copy()
            mask_inv = cv2.bitwise_not(mask_binary)
            img2_bg = cv2.bitwise_and(seamless, seamless, mask=mask_inv)
            img1_fg = cv2.bitwise_and(img1_warped_transferred, img1_warped_transferred, mask=mask_binary)
            seamless = cv2.add(img2_bg, img1_fg)
    else:
        # 直接进入 Alpha 羽化混合的路径
        seamless = img2_target_bgr.copy()

    # ===========================
    # 5. 边缘平滑与全局合成 (Final Blending)
    # ===========================
    print("正在进行边缘平滑处理...")
    if progress: progress(0.9, desc="正在进行边缘羽化与最终合成...")
    
    # 1. 优化边缘羽化：基于图 1 的原始 Mask 结构进行精细柔化
    # 参考 PS 的边缘羽化逻辑：先收缩再超大模糊，确保边缘完全融入背景
    
    # 稍微收缩 mask，确保不包含任何抠图残留的硬边
    kernel_feather = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_main = cv2.erode(mask_binary, kernel_feather, iterations=3)
    
    # 计算动态羽化半径：图像越大，羽化越宽
    # 模仿 Harmonization 的柔和边缘感
    feather_amount = int(min(w_bg, h_bg) * 0.008) 
    if feather_amount < 11: feather_amount = 11
    if feather_amount % 2 == 0: feather_amount += 1
    
    # 使用高斯模糊产生平滑的 Alpha 渐变层
    mask_feathered = cv2.GaussianBlur(mask_main, (feather_amount, feather_amount), 0)
    
    # 2. 核心区域保护：锁定面部核心细节
    kernel_size_protect = int(min(w_bg, h_bg) * 0.02)
    if kernel_size_protect % 2 == 0: kernel_size_protect += 1
    kernel_protect = np.ones((kernel_size_protect, kernel_size_protect), np.uint8)
    mask_eroded = cv2.erode(mask_main, kernel_protect, iterations=1)
    if face_mask_warped is not None:
        _, face_bin = cv2.threshold(face_mask_warped, 128, 255, cv2.THRESH_BINARY)
        kernel_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        face_bin = cv2.dilate(face_bin, kernel_face, iterations=1)
        before_core = int(np.count_nonzero(mask_eroded))
        face_core = int(np.count_nonzero(face_bin))
        mask_eroded = cv2.bitwise_or(mask_eroded, face_bin)
        after_core = int(np.count_nonzero(mask_eroded))
        print(f"BiSeNet core protection: base={before_core}, face={face_core}, merged={after_core}")
    
    blur_radius_protect = int(min(w_bg, h_bg) * 0.08) 
    if blur_radius_protect % 2 == 0: blur_radius_protect += 1
    mask_blur_protect = cv2.GaussianBlur(mask_eroded, (blur_radius_protect, blur_radius_protect), 0)

    alpha_edge = mask_feathered.astype(float) / 255.0
    alpha_edge = np.stack([alpha_edge] * 3, axis=2)

    alpha_protect_single = mask_blur_protect.astype(float) / 255.0
    core_strong = (alpha_protect_single >= 0.6).astype(np.float32)
    alpha_protect_single = alpha_protect_single * (1.0 - core_strong) + core_strong
    alpha_protect = np.stack([alpha_protect_single] * 3, axis=2)
    
    # A. 局部协调混合：将 Harmonized 前景与泊松融合边缘结合
    blended_inner = (img1_warped_transferred.astype(float) * alpha_protect + seamless.astype(float) * (1 - alpha_protect))
    
    # B. 全局协调合成：将结果羽化到背景
    final_output = (blended_inner * alpha_edge + img2_target_bgr.astype(float) * (1 - alpha_edge)).astype(np.uint8)
    
    tag = {
        ("poisson_normal", True): "泊松-标准+协调",
        ("poisson_mixed", True): "泊松-混合+协调",
        ("poisson_normal", False): "泊松-标准",
        ("poisson_mixed", False): "泊松-混合",
        ("alpha", True): "羽化-Alpha+协调",
        ("alpha", False): "羽化-Alpha"
    }.get((mode, use_color), "融合结果")
    return final_output, f"处理成功：{tag}"

def main_process(img1_pil, img2_pil, enable_algo_b=False, enable_algo_c=False, progress=gr.Progress()):
    if img1_pil is None or img2_pil is None:
        return None, None, None, None, None, None, "请先上传两张图片。"

    progress(0, desc="正在初始化...")
    # 1. 转换为 OpenCV 格式
    img1 = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2BGR)

    face_mask = None
    try:
        face_mask = get_face_mask_bisenet(img1)
    except Exception as e:
        print(f"BiSeNet 面部分割失败: {e}")

    # 1.1 调整底图分辨率：最长边设置为 6048px (LANCZOS4 插值)
    h, w = img2.shape[:2]
    target_max = 6048
    if max(h, w) != target_max:
        scale = target_max / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"正在调整底图分辨率: {w}x{h} -> {new_w}x{new_h} (LANCZOS4)")
        progress(0.1, desc=f"正在调整底图分辨率至 {target_max}px...")
        img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = img2.shape[:2]

    print("Step 0: 正在对图2进行人脸内容填充...")
    progress(0.15, desc="正在对底图原有人脸做内容填充...")
    img2_filled, bg_face_mask = inpaint_face_region(img2)
    if bg_face_mask is not None:
        print("背景人脸已通过内容填充移除，将使用填充后的底图进行叠图")
        img2 = img2_filled

    # 2. 前景蒙版 (BiSeNet)
    print("Step 1: 正在生成前景蒙版 (BiSeNet)...")
    progress(0.2, desc="正在进行 BiSeNet 前景分割...")
    person_mask = get_person_mask_bisenet(img1)
    if person_mask is None:
        print("BiSeNet 前景分割失败，退化为整图前景")
        alpha = np.full(img1.shape[:2], 255, dtype=np.uint8)
    else:
        alpha = person_mask
    if face_mask is not None:
        h_fg, w_fg = img1.shape[:2]
        if face_mask.shape[:2] != (h_fg, w_fg):
            face_mask_resized = cv2.resize(face_mask, (w_fg, h_fg), interpolation=cv2.INTER_NEAREST)
        else:
            face_mask_resized = face_mask
        kernel_face_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        face_safe = cv2.dilate(face_mask_resized, kernel_face_safe, iterations=2)
        before_alpha_core = int(np.count_nonzero(alpha))
        alpha = cv2.bitwise_or(alpha, face_safe)
        after_alpha_core = int(np.count_nonzero(alpha))
        print(f"Strict face protection applied on alpha: base={before_alpha_core}, merged={after_alpha_core}")
    img1_nobg_cv = np.dstack([img1, alpha])

    # 3. 对齐与融合
    print(f"Step 2: 对齐与融合 (底图分辨率: {img2.shape[1]}x{img2.shape[0]})...")
    progress(0.4, desc="正在开始对齐与融合...")
    # 生成多种算法结果
    variants = [
        ("算法A：泊松-标准", dict(mode="poisson_normal", use_color=True)),
        ("算法B：泊松-混合", dict(mode="poisson_mixed", use_color=True)),
        ("算法C：羽化-Alpha", dict(mode="alpha", use_color=False)),
    ]

    before_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    compares = []
    files_list = []
    msgs = []

    def save_png_jpg(img_bgr, tag):
        temp_dir = tempfile.gettempdir()
        pid = os.getpid()
        png_path = os.path.join(temp_dir, f"fusion_{tag}_{pid}.png")
        cv2.imwrite(png_path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        jpg_path = os.path.join(temp_dir, f"fusion_{tag}_{pid}.jpg")
        quality = 99
        while quality > 50:
            cv2.imwrite(jpg_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if os.path.getsize(jpg_path) < 10 * 1024 * 1024:
                break
            quality -= 5
        wm_text = "Lanmei AI Portrait Studio"
        wm_img = img_bgr.copy()
        h, w = wm_img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = max(w, h) / 1600.0
        font_scale = max(base_scale, 0.8)
        thickness = int(max(w, h) / 900)
        if thickness < 1:
            thickness = 1
        # 使用放大画布 + 45 度旋转的方式生成斜向平铺水印
        scale_canvas = 1.6
        big_h = int(h * scale_canvas)
        big_w = int(w * scale_canvas)
        overlay_big = np.zeros((big_h, big_w, 3), dtype=wm_img.dtype)
        text_size, _ = cv2.getTextSize(wm_text, font, font_scale, thickness)
        tw, th = text_size
        # 缩小水平方向间距(更紧密)，增大垂直方向间距(更舒展)
        step_x = int(tw * 1.2)
        step_y = int(th * 2.2)
        if step_x < 80:
            step_x = 80
        if step_y < 60:
            step_y = 60
        for y in range(0, big_h + step_y, step_y):
            for x in range(-big_w // 2, big_w + step_x, step_x):
                cv2.putText(overlay_big, wm_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        center_big = (big_w // 2, big_h // 2)
        rot_mat = cv2.getRotationMatrix2D(center_big, 45, 1.0)
        overlay_big = cv2.warpAffine(overlay_big, rot_mat, (big_w, big_h))
        y0 = (big_h - h) // 2
        x0 = (big_w - w) // 2
        overlay = overlay_big[y0:y0 + h, x0:x0 + w]
        alpha = 0.15
        wm_img = cv2.addWeighted(overlay, alpha, wm_img, 1 - alpha, 0)
        wm_path = os.path.join(temp_dir, f"fusion_{tag}_{pid}_watermark.jpg")
        cv2.imwrite(wm_path, wm_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return png_path, jpg_path, wm_path

    for idx, (title, params) in enumerate(variants):
        # 检查是否启用算法 B 或 C
        if idx == 1 and not enable_algo_b:
            compares.append(None)
            files_list.append(None)
            continue
        if idx == 2 and not enable_algo_c:
            compares.append(None)
            files_list.append(None)
            continue

        progress(0.5 + idx * 0.15, desc=f"正在生成 {title} ...")
        res_img, msg = blend_images_smart(img1_nobg_cv, img2, progress=progress, face_mask=face_mask, **params)
        if res_img is None:
            compares.append(None)
            files_list.append(None)
            msgs.append(f"{title}: 失败 | {msg}")
            continue
        out_h, out_w = res_img.shape[:2]
        msgs.append(f"{title}: {msg} | 分辨率: {out_w}x{out_h}")
        res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        compares.append([before_rgb, res_rgb])
        png_path, jpg_path, wm_path = save_png_jpg(res_img, f"v{idx+1}")
        files_list.append([png_path, jpg_path, wm_path])

    final_msg = " | ".join([m for m in msgs if m])
    progress(1.0, desc="全部结果生成完成！")
    return (
        compares[0], files_list[0],
        compares[1], files_list[1],
        compares[2], files_list[2],
        final_msg
    )

def preview_segmentation(img1_pil, enable_bisenet):
    if img1_pil is None:
        return None
    img1 = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
    bisenet_preview = None
    if enable_bisenet:
        try:
            mask = get_face_mask_bisenet(img1)
        except Exception as e:
            print(f"BiSeNet 预览失败: {e}")
            mask = None
        if mask is not None:
            overlay = img1.copy()
            colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
            bisenet_preview = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return bisenet_preview

# ===========================
# Gradio 界面构建
# ===========================
with gr.Blocks(title="蓝梅 AI 智能叠图工具") as demo:
    gr.Markdown("# 📸 蓝梅 AI 智能人像叠图工具")
    gr.Markdown("自动抠图｜像素级对齐｜智能边缘融合")
    with gr.Row():
        gr.Markdown(f"当前硬件加速环境：**{DEVICE_LABEL}**")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            input_fg = gr.Image(type="pil", label="图1：人物特写/半身 (将被抠图)", height=240)
            input_bg = gr.Image(type="pil", label="图2：底图/全景 (背景参考)", height=240)
            with gr.Row():
                bisenet_toggle = gr.Checkbox(label="预览 BiSeNet 分割", value=False)
                enable_algo_b = gr.Checkbox(label="启用算法B (泊松-混合)", value=False)
                enable_algo_c = gr.Checkbox(label="启用算法C (羽化-Alpha)", value=False)
            bisenet_preview_img = gr.Image(label="BiSeNet 面部分割预览", interactive=False)
            btn_run = gr.Button("🚀 开始叠图", variant="primary")

        with gr.Column(scale=4, min_width=520):
            output_msg = gr.Textbox(label="状态信息")
            with gr.Tabs():
                with gr.Tab("算法A：泊松-标准"):
                    output_img_a = gr.Gallery(label="对比预览A (前/后)", columns=2)
                    output_file_a = gr.File(label="下载A (PNG/JPG)", height=100)
                with gr.Tab("算法B：泊松-混合"):
                    output_img_b = gr.Gallery(label="对比预览B (前/后)", columns=2)
                    output_file_b = gr.File(label="下载B (PNG/JPG)", height=100)
                with gr.Tab("算法C：羽化-Alpha"):
                    output_img_c = gr.Gallery(label="对比预览C (前/后)", columns=2)
                    output_file_c = gr.File(label="下载C (PNG/JPG)", height=100)
            
    btn_run.click(
        fn=main_process,
        inputs=[input_fg, input_bg, enable_algo_b, enable_algo_c],
        outputs=[
            output_img_a, output_file_a,
            output_img_b, output_file_b,
            output_img_c, output_file_c,
            output_msg,
        ],
    )

    def bind_seg_preview(component):
        component.change(
            fn=preview_segmentation,
            inputs=[input_fg, bisenet_toggle],
            outputs=[bisenet_preview_img],
        )

    bind_seg_preview(input_fg)
    bind_seg_preview(bisenet_toggle)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)
