# AI Portrait Fusion Studio

AI 人像智能叠图工具 -- 自动抠图、像素级对齐、双算法融合。

将图1的人物面部无缝融合到图2的场景中，支持 Web UI 和 REST API。

## 功能特性

- **双算法融合**：泊松融合（主方案）和 Alpha 径向融合（备选），同时输出对比
- **双模型分割**：BiSeNet 面部语义分割 + MODNet 人像抠图
- **智能对齐**：SIFT 特征匹配 + FLANN + RANSAC 仿射变换
- **色彩协调**：LAB 色彩空间自动匹配，消除光照/色温差异
- **面部保护**：面部核心区域不被融合算法模糊
- **并发支持**：FastAPI + ThreadPoolExecutor，线程安全 ONNX 推理
- **REST API**：支持文件上传和 URL 输入，含 SSRF 防护
- **局域网可访问**：监听 `0.0.0.0:7860`

## 技术架构

```
fusion-tools/
├── app.py                    # FastAPI + Gradio 入口
├── config.py                 # 配置集中管理
├── requirements.txt
├── api/
│   └── routes.py             # REST API（文件上传 + URL 输入）
├── core/
│   ├── model_manager.py      # 双模型 ONNX 管理（线程安全单例）
│   ├── segmentation.py       # BiSeNet 语义分割 + MODNet 人像抠图
│   ├── alignment.py          # SIFT 特征对齐（缩图加速）
│   ├── color_transfer.py     # LAB 色彩迁移（ROI 优化）
│   ├── blending.py           # 泊松融合 + 拉普拉斯金字塔 + Alpha 混合
│   ├── pipeline.py           # 双算法流水线编排
│   └── watermark.py          # 水印生成
├── ui/
│   └── gradio_app.py         # Gradio Web 界面
models/
├── bisenet/resnet18.onnx     # BiSeNet 面部分割模型 (51MB)
└── modnet/modnet_*.onnx      # MODNet 人像抠图模型 (25MB)
```

## 算法概述

### 方案 A：BiSeNet + 泊松融合（主方案）

适合两张图光照接近、人物姿态相似的场景。泊松融合在梯度域求解边界值问题，能自然消除边缘色差。

**优点**：边缘过渡最自然，泊松方程保证梯度连续性
**缺点**：大图计算量大（已优化为缩图融合），光照差异极大时面部可能偏色

### 方案 B：MODNet + 径向 Alpha 融合（备选）

适合需要精确控制融合范围的场景。MODNet 提供发丝级 alpha matte，径向渐变限制融合区域在面部附近。

**优点**：融合边界可控，不会影响身体/物体，alpha 渐变无色差
**缺点**：没有泊松的梯度域匹配，依赖色调预匹配质量

## 处理流程

```
输入：图1（人物特写）+ 图2（场景底图）
        │
        ▼
┌─── SIFT 特征对齐（共用）───┐
│  图1/图2 缩到 2000px       │
│  SIFT 特征提取 (4000点)    │
│  FLANN 匹配 + RANSAC      │
│  计算仿射变换矩阵          │
└────────────┬───────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
 方案 A            方案 B
 BiSeNet            MODNet
     │               │
     ▼               ▼
 面部语义分割     人像 alpha matte
 (19类标签)      (连续 0~255)
     │               │
     ▼               ▼
 底图人脸 inpaint  BiSeNet 面部定位
     │               │
     ▼               ▼
 色彩迁移(LAB)    径向渐变 mask
     │            (smoothstep)
     ▼               │
 泊松融合          ▼
 (seamlessClone)  边缘色调匹配
     │               │
     ▼               ▼
 边缘羽化+         直接 Alpha
 面部保护          Blending
     │               │
     ▼               ▼
   输出 A          输出 B
```

## 性能优化

| 优化项 | 说明 |
|--------|------|
| BiSeNet 单次推理 | 同一张图只推理一次，派生 face_mask + person_mask |
| SIFT 缩图匹配 | 在 2000px 缩图上匹配，变换矩阵还原到原图尺度 |
| 泊松缩图融合 | 大图缩到 2000px 做 seamlessClone，上采样回原尺寸 |
| 色彩迁移 ROI | 只对 mask bounding box 区域做 LAB 转换 |
| numpy 广播 | alpha 混合用向量化广播替代 Python for 循环 |
| ONNX 优化 | graph optimization + 多线程推理 |
| 线程安全 | 每个 ONNX 模型独立推理锁，支持并发请求 |

## 安装与运行

```bash
# 克隆仓库
git clone <repo-url>
cd lanmei-ai-fusion-tools

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r fusion-tools/requirements.txt

# 启动服务
python3 fusion-tools/app.py
```

服务启动后访问 `http://localhost:7860`

## API 接口

### 人像融合

```bash
# 文件上传
curl -X POST http://localhost:7860/api/fusion \
  -F "fg=@person.jpg" -F "bg=@background.jpg"

# URL 输入
curl -X POST http://localhost:7860/api/fusion \
  -F "fg_url=https://example.com/person.jpg" \
  -F "bg_url=https://example.com/bg.jpg"
```

### 下载结果

```bash
curl -O http://localhost:7860/api/download?path=<返回的路径>
```

### 健康检查

```bash
curl http://localhost:7860/api/health
```

## 模型说明

| 模型 | 用途 | 大小 | 来源 |
|------|------|------|------|
| BiSeNet (ResNet18) | 面部 19 类语义分割 | 51MB | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) |
| MODNet | 人像抠图 (alpha matte) | 25MB | [MODNet](https://github.com/ZHKKKe/MODNet) |

## 依赖

- Python 3.10+
- FastAPI + Uvicorn
- Gradio
- OpenCV (contrib, headless)
- ONNX Runtime
- NumPy, Pillow, httpx

## 许可证

MIT License
