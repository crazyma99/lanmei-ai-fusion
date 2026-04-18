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

## 算法概述（详细版）

本项目不是单一算法，而是“**公共对齐阶段 + 两套独立融合策略**”的组合设计。

### 公共阶段：SIFT 特征对齐

无论最终使用哪套融合方案，都会先做一次图像对齐：

1. 将两张图缩放到较小尺寸（默认最长边 2000px）
2. 用 **SIFT** 提取关键点与描述子
3. 用 **FLANN** 做最近邻特征匹配
4. 通过 **Lowe ratio test** 筛掉错误匹配
5. 使用 **RANSAC + estimateAffinePartial2D** 求稳定的仿射矩阵
6. 将仿射矩阵恢复到原图尺度并对图 1 做 warp

这样可以把“图 1 人物”尽量放到“图 2 对应位置”上，为后续融合创造基础条件。

### 方案 A：BiSeNet + 泊松融合（主方案）

这是当前的**主方案**，核心目标是：

> 利用语义分割得到较稳定的人脸/人物区域，再通过泊松融合解决边缘断层，让融合结果更“像一张图”。

#### A1. 为什么选 BiSeNet

BiSeNet 在这里并不是做通用抠图，而是做 **面部语义解析**：

- 能识别皮肤、眉毛、眼睛、耳朵、鼻子、嘴唇、头发、脖子、帽子等类别
- 能提供比普通二值分割更稳定的“面部核心区域”
- 对面部保护特别有帮助

项目里基于 parsing map 生成两类 mask：

- **face mask**：用于面部保护、底图人脸去除
- **person mask**：用于前景整体融合

#### A2. 为什么要先对底图做人脸 inpaint

如果图 2 里本来就有人脸，直接把图 1 的脸叠上去，会产生：

- 双眼/双嘴重影
- 鼻梁高光错位
- 脸部边缘重复

所以主方案会先对图 2 的原人脸区域做一次 `inpaint`，把底图原有脸部弱化掉，再把图 1 的人物融合进去。

#### A3. 色彩迁移的作用

即使位置对齐了，两张图通常仍有这些差异：

- 色温不同（偏黄 / 偏青）
- 曝光不同（亮 / 暗）
- 对比度不同
- 白平衡不同

因此主方案在泊松融合前，会先基于 **LAB 色彩空间** 做一次局部色彩迁移：

- 只在 ROI 内进行
- 对 L 通道和 AB 通道分别做统计匹配
- 尽量让图 1 前景靠近图 2 对应区域的整体色调

这一步不能完全消除差异，但能显著降低后续融合的难度。

#### A4. 泊松融合的核心价值

泊松融合（`seamlessClone`）本质上是在**梯度域**里处理边界过渡：

- 不直接硬拷贝像素
- 而是尽量保持前景内部结构，同时让边界梯度与背景连续

所以它特别适合解决：

- 脸边缘一圈明显切割线
- 头发与背景交界的硬边
- 衣服轮廓与背景之间的亮度断层

#### A5. 面部保护为什么还要做二次处理

泊松融合虽然自然，但也有副作用：

- 面部纹理会被“抹平”
- 五官锐度会下降
- 眼睛、鼻梁等关键区域会丢细节

所以主方案在最后还会做一次 **edge feather + face core protect**：

- 边缘区域用羽化结果自然过渡
- 面部核心区域保留更多来自前景图的高频细节

#### A6. 方案 A 的适用情况

更适合：

- 两张图人物姿态接近
- 光照方向差异不大
- 你更在意“整体融合感”
- 面部和头发边缘要尽量自然

不太适合：

- 图 1 身体部分和图 2 身体部分差异很大
- 非面部物体（船、花、手臂、衣摆）容易被一并拉进融合区域

---

### 方案 B：MODNet + 径向 Alpha 融合（备选方案）

这是当前的**备选方案**，核心目标是：

> 用高质量 alpha matte 精准限定融合区域，尽量只动面部附近，避免身体或场景物体被错误扭曲。

#### B1. 为什么选 MODNet

MODNet 是做人像 matting 的模型，不是语义分类模型：

- 输出的是 **连续 alpha matte**（0~255）
- 能保留发丝、半透明头发边缘等细节
- 比二值 mask 更适合做直接 alpha blending

它的优势不是“知道脸的哪个器官是什么”，而是：

> **知道每个像素有多少比例属于前景人物。**

这让它特别适合做“柔和叠加”。

#### B2. 为什么不用整个人像 alpha 直接叠

如果直接把 warp 后的整个人像 alpha 拿来融合，会出现一个典型问题：

- 图 1 里有的身体、衣服、手、物体
- 图 2 对应位置未必有这些结构
- 一旦整块融合，就会出现身体崩坏、船体变形、衣摆异常

所以这里不会直接用 MODNet alpha 全量融合，而是进一步做“**面部中心径向裁剪**”。

#### B3. 径向 Alpha Mask 的设计目的

方案 B 先用 BiSeNet 辅助找到面部中心，再基于面部中心生成一个**径向衰减 mask**：

- 面部中心附近：alpha 最大
- 离面部越远：alpha 越低
- 到一定半径之外：alpha → 0

再与 MODNet alpha 相乘，得到最终融合 alpha。

这样做的好处是：

- 面部能完整融合
- 脖子 / 肩膀可以渐变过渡
- 身体、船、花、手等远离脸的区域自动回退到图 2 原样

#### B4. 为什么还要做边缘色调匹配

即使 alpha 过渡足够平滑，如果图 1 和图 2 在边缘区域的颜色差异很大，仍然会看到色差带。

所以方案 B 在最终 alpha blending 前，会对 **融合边缘环带** 做一次局部色调匹配：

- 采样 alpha 处于 5%~40% 的环带
- 在 LAB 空间统计源图和目标图的亮度与色彩分布
- 让图 1 在“交汇边缘”更接近图 2

然后再执行：

```text
result = source * alpha + target * (1 - alpha)
```

#### B5. 为什么方案 B 不再用拉普拉斯金字塔

最早方案 B 曾尝试过分频融合，但出现过：

- 低频 mask 扩散
- 边缘晕染
- 色差带难以消除

后来调整为：

> **MODNet 连续 alpha + 径向 mask + 边缘色调匹配 + 直接 alpha blending**

原因很简单：

- MODNet 本身已经给出了高质量柔性边缘
- 再做复杂的频段分解，反而引入额外色调漂移
- 直接 alpha blending 更稳定、更可控

#### B6. 方案 B 的适用情况

更适合：

- 你只想融合脸，不想动身体或场景物体
- 图 1 和图 2 的身体结构差异大
- 背景里有船、花、手臂、衣摆等不希望被扭曲的区域
- 你更重视“局部可控性”而不是“整块融合感”

不太适合：

- 需要整个人物都融进场景
- 想让头发和衣服边缘都完全由梯度域处理

---

## 两种方案如何选择

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
