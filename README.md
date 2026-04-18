# AI Portrait Fusion Studio

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-Web%20API-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Gradio-WebUI-F97316?logo=gradio&logoColor=white" alt="Gradio" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime-Inference-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime" />
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="MIT License" />
</p>

<p align="center">
  AI 人像智能叠图工具：自动抠图、像素级对齐、双算法融合、局域网 Web UI + REST API。
</p>

---

## 项目简介

`AI Portrait Fusion Studio` 是一个面向 **人像换脸 / 场景叠图 / 脸部融合** 场景的本地 AI 工具。

它的目标不是做通用 Photoshop 替代，而是专注于一个具体问题：

> **把图 1 中的人物面部或人物主体，自然地融合到图 2 的目标场景中，同时尽量避免明显边缘、重影、色差、身体崩坏和物体变形。**

项目目前同时提供两套融合方案：

- **方案 A：BiSeNet + 泊松融合**（主方案）
- **方案 B：MODNet + 径向 Alpha 融合**（备选方案）

这样可以根据不同图片条件切换结果：
- 如果你更看重 **边缘自然、整体融合感**，优先看 **方案 A**
- 如果你更看重 **融合范围可控、避免身体/船体/物体被错误变形**，优先看 **方案 B**

---

## 核心特性

- **双算法输出**：同时生成两种融合结果，方便直接对比挑选
- **双模型协同**：BiSeNet 做面部语义解析，MODNet 做高质量人像 alpha matte
- **像素级自动对齐**：SIFT + FLANN + RANSAC 自动估计仿射变换
- **色调自动协调**：基于 LAB 色彩空间的局部色彩迁移 / 边缘色调匹配
- **面部保护机制**：面部核心区域避免被泊松融合过度抹平
- **局域网部署友好**：监听 `0.0.0.0:7860`，可直接给同网段设备访问
- **API 可接入**：支持文件上传与图片 URL 输入
- **线程安全 ONNX 推理**：双模型独立推理锁，支持多请求并发
- **模型随仓库同步**：BiSeNet 与 MODNet 模型均纳入仓库（通过 Git LFS）

---

## 适用场景

本项目适合以下类型的图片处理任务：

- 古风 / 婚纱 / 写真人像叠图
- 将人物面部替换到已有构图场景中
- 在不手工抠图的前提下快速得到可用融合图
- 做“多算法结果对比”并从中人工挑选最佳结果
- 私有部署 / 局域网内部图像处理服务

不适合的情况：

- 人物姿态差异极大、头部朝向完全不同
- 两张图透视关系差异极大
- 背景遮挡关系复杂（例如手挡脸、头发被复杂物体遮住）
- 需要影视级精修的商用海报合成

---

## 技术架构

```text
fusion-tools/
├── app.py                    # FastAPI + Gradio 入口
├── config.py                 # 配置集中管理
├── requirements.txt
├── api/
│   └── routes.py             # REST API（文件上传 + URL 输入 + 结果下载）
├── core/
│   ├── model_manager.py      # 双模型 ONNX 管理（线程安全单例）
│   ├── segmentation.py       # BiSeNet 语义分割 + MODNet 人像抠图
│   ├── alignment.py          # SIFT 特征对齐（缩图加速）
│   ├── color_transfer.py     # LAB 色彩迁移（ROI 优化）
│   ├── blending.py           # 泊松融合 / Alpha 融合 / 金字塔工具
│   ├── pipeline.py           # 双算法流水线编排
│   └── watermark.py          # 水印生成
├── ui/
│   └── gradio_app.py         # Gradio Web 界面
models/
├── bisenet/resnet18.onnx     # BiSeNet 面部分割模型 (~51MB)
└── modnet/modnet_*.onnx      # MODNet 人像抠图模型 (~25MB)
```

### 运行层次

```text
Browser / Client
   ├── Gradio UI
   └── REST API
          │
          ▼
      FastAPI App
          │
          ▼
     Fusion Pipeline
   ├── Alignment
   ├── Segmentation
   ├── Color Matching
   ├── Blending
   └── Result Export
          │
          ▼
   ONNX Runtime + OpenCV
```

---

## 算法概述

本项目采用“**公共对齐阶段 + 双融合策略**”的设计：

- **公共阶段**：SIFT + FLANN + RANSAC 完成人物位置对齐
- **方案 A（主方案）**：BiSeNet + 泊松融合，强调整体融合感与边缘自然过渡
- **方案 B（备选）**：MODNet + 径向 Alpha 融合，强调局部可控性与避免身体/物体变形

详细算法说明请查看：

- [方案 A：BiSeNet + 泊松融合详解](docs/image_algorithm1.md)
- [方案 B：MODNet + 径向 Alpha 融合详解](docs/image_algorithm2.md)

### 两种方案如何选择

| 场景 | 推荐方案 |
|------|----------|
| 追求整体自然融合感 | 方案 A（BiSeNet + 泊松融合） |
| 追求只改脸、不破坏身体和物体 | 方案 B（MODNet + 径向 Alpha） |
| 两张图姿态很接近 | 优先方案 A |
| 两张图身体差异大 / 背景物体复杂 | 优先方案 B |
| 容忍一定计算量，想要更自然边缘 | 优先方案 A |
| 更看重结果可控性 | 优先方案 B |

---

## 处理流程

```text
输入：图1（人物特写） + 图2（场景底图）
        │
        ▼
┌──────────────────────────────────┐
│      公共前处理：SIFT 特征对齐      │
│  1. 缩图到 2000px                │
│  2. SIFT 提取关键点              │
│  3. FLANN 匹配                  │
│  4. RANSAC 求仿射矩阵           │
│  5. warp 到图2坐标系            │
└──────────────────────────────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
┌────────────────┐   ┌────────────────┐
│ 方案 A（主方案） │   │ 方案 B（备选） │
├────────────────┤   ├────────────────┤
│ BiSeNet 分割     │   │ MODNet 抠图     │
│ face/person mask │   │ alpha matte    │
│        │         │   │       │        │
│ 底图人脸 inpaint │   │ BiSeNet 面部定位 │
│        │         │   │       │        │
│ LAB 色彩迁移     │   │ 径向 alpha 裁剪 │
│        │         │   │       │        │
│ 泊松融合         │   │ 边缘色调匹配    │
│        │         │   │       │        │
│ 边缘羽化+面部保护 │   │ 直接 alpha 混合 │
└────────────────┘   └────────────────┘
         │                 │
         └────────┬────────┘
                  ▼
            双结果输出对比
```

---

## 性能优化

| 优化项 | 说明 |
|--------|------|
| BiSeNet 单次推理 | 同一张图只推理一次，派生 face mask / person mask |
| SIFT 缩图匹配 | 在 2000px 缩图上匹配，矩阵恢复到原图尺度 |
| 泊松融合优化 | 大图缩到 `POISSON_MAX_PX` 处理，再只替换 mask 区域 |
| 色彩迁移 ROI | 只在 mask bounding box 内做 LAB 转换 |
| 线程安全 ONNX | 每个模型独立推理锁，避免并发 `session.run()` 冲突 |
| numpy 广播 | 避免 Python 通道循环，减少内存与 CPU 开销 |
| SSRF 防护 | URL 输入限制 scheme、IP 范围、下载体积 |
| Warp 边界修复 | `BORDER_REFLECT` + alpha 内缩，消除细黑边 |

---

## 安装与运行

```bash
# 克隆仓库
git clone https://github.com/crazyma99/lanmei-ai-fusion.git
cd lanmei-ai-fusion

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r fusion-tools/requirements.txt

# 启动服务
python3 fusion-tools/app.py
```

启动后访问：

- **Web UI**：`http://localhost:7860`
- **局域网访问**：`http://<你的局域网IP>:7860`
- **Health API**：`http://localhost:7860/api/health`

---

## API 接口

### 1）人像融合

```bash
# 文件上传
curl -X POST http://localhost:7860/api/fusion \
  -F "fg=@person.jpg" \
  -F "bg=@background.jpg"

# URL 输入
curl -X POST http://localhost:7860/api/fusion \
  -F "fg_url=https://example.com/person.jpg" \
  -F "bg_url=https://example.com/background.jpg"
```

返回示例：

```json
{
  "message": "方案A：泊松融合完成 | 方案B：Alpha融合完成",
  "algorithms": {
    "bisenet_poisson": {
      "png": "/api/download?path=/tmp/fusion_bisenet_xxx.png",
      "jpg": "/api/download?path=/tmp/fusion_bisenet_xxx.jpg",
      "watermark": "/api/download?path=/tmp/fusion_bisenet_xxx_wm.jpg"
    },
    "modnet_alpha": {
      "png": "/api/download?path=/tmp/fusion_modnet_xxx.png",
      "jpg": "/api/download?path=/tmp/fusion_modnet_xxx.jpg",
      "watermark": "/api/download?path=/tmp/fusion_modnet_xxx_wm.jpg"
    }
  }
}
```

### 2）下载结果

```bash
curl -L "http://localhost:7860/api/download?path=/tmp/fusion_bisenet_xxx.png" -o result.png
```

### 3）健康检查

```bash
curl http://localhost:7860/api/health
```

---

## 模型说明

| 模型 | 用途 | 大小 | 来源 |
|------|------|------|------|
| BiSeNet (ResNet18) | 面部 19 类语义分割 | ~51MB | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) |
| MODNet | 人像抠图（连续 alpha matte） | ~25MB | [MODNet](https://github.com/ZHKKKe/MODNet) |

### 模型职责分工

- **BiSeNet**：更偏“理解面部结构”，适合做面部保护、人物区域控制、底图人脸去除
- **MODNet**：更偏“高质量柔性边缘”，适合做精细 alpha blending

---

## 依赖

- Python 3.10+
- FastAPI + Uvicorn
- Gradio
- OpenCV (contrib, headless)
- ONNX Runtime
- NumPy
- httpx
- Git LFS（用于同步模型）

---

## 开发说明

### 为什么保留双方案

因为真实图片条件差异很大，单一算法很难覆盖所有场景：

- 有些图适合整块融合（方案 A）
- 有些图只能动脸，不能动身体或物体（方案 B）

保留双方案的意义不是“功能堆叠”，而是让用户在一次处理中直接看到两种策略的结果，减少反复调参数的时间。

### 当前已解决的典型问题

- 身体非重叠区域崩坏
- 船体 / 物体被错误融合
- 融合边缘细黑线
- URL 输入 SSRF 风险
- ONNX 并发推理安全问题
- 泊松融合大图性能瓶颈

### 仍然存在的客观限制

- 如果两张图姿态差异极大，对齐质量仍然会受限
- 如果遮挡关系复杂（手挡脸、头发遮挡物体），自动融合仍可能失败
- 如果光照方向完全相反，仅靠色调匹配也不能完全解决真实感问题

---

## License

MIT License
