import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# BiSeNet 模型（原方案 - 主方案）
BISENET_MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "bisenet")
BISENET_ONNX_PATH = os.path.join(BISENET_MODEL_DIR, "resnet18.onnx")

# MODNet 模型（新方案 - 备选）
MODNET_MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "modnet")
MODNET_ONNX_PATH = os.path.join(MODNET_MODEL_DIR, "modnet_photographic_portrait_matting.onnx")

# 图像处理参数
MAX_RESOLUTION = 6048
SIFT_FEATURES = 4000
SIFT_CONTRAST_THRESHOLD = 0.03
SIFT_MATCH_RATIO = 0.65
MIN_GOOD_MATCHES = 20
ALIGNMENT_MAX_PX = 2000
POISSON_MAX_PX = 2000

# BiSeNet 分割 ID
FACE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17]
PERSON_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17, 18]

# MODNet 配置
MODNET_REF_SIZE = 512

# ONNX 推理配置
ONNX_INTER_OP_THREADS = 2     # 跨算子并行线程
ONNX_INTRA_OP_THREADS = 4     # 算子内部并行线程

# 水印配置
WATERMARK_TEXT = "Lanmei AI Portrait Studio"
WATERMARK_ALPHA = 0.15
WATERMARK_ANGLE = 45

# 服务配置
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860
MAX_UPLOAD_SIZE_MB = 20
MAX_CONCURRENT_WORKERS = 4
TEMP_FILE_MAX_AGE_HOURS = 2

# URL 下载配置
URL_DOWNLOAD_TIMEOUT = 30      # 秒
URL_MAX_SIZE_MB = 20
