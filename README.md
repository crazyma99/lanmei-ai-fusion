# Photo Fusion AI

这是一个基于 AI 的人像叠图工具，利用深度学习模型实现像素级对齐和智能融合。

## 功能特点

- **自动人像分割**：使用 BiSeNet (ResNet18) 进行高精度的面部和前景分割。
- **智能对齐**：基于特征点匹配和相似变换，自动对齐两张图片中的人物。
- **多种融合算法**：
    - **算法A（泊松-标准）**：经典的泊松融合，适合大多数场景。
    - **算法B（泊松-混合）**：更激进的融合方式，保留更多细节。
    - **算法C（羽化-Alpha）**：简单的 Alpha 混合，边缘处理柔和。
- **面部保护**：专门针对面部区域进行保护，防止过度融合导致面部特征丢失。
- **背景修复**：自动识别并修复底图中的人脸区域，避免重影。
- **实时预览**：支持 BiSeNet 分割结果的实时预览。

## 安装说明

1.  **环境要求**：
    - Python 3.8+
    - 建议使用虚拟环境 (venv)

2.  **安装依赖**：
    ```bash
    pip install -r fusion-tools/requirements.txt
    ```

3.  **模型文件**：
    - 本项目包含必要的模型文件，位于 `models/bisenet/` 目录下。
    - 确保 `models/bisenet/resnet18.onnx` 和 `models/bisenet/bisenet_face.pth` 存在。

## 使用方法

### 本地运行

直接运行 Python 脚本启动 Gradio 服务：

```bash
python fusion-tools/app.py
```

服务启动后，在浏览器访问 `http://localhost:7860` 即可使用。

## 项目结构

```
photo_fusion/
├── fusion-tools/          # 核心代码
│   ├── app.py             # Gradio 应用入口
│   └── requirements.txt   # 依赖列表
├── models/                # 模型文件
│   └── bisenet/
│       ├── resnet18.onnx
│       └── bisenet_face.pth
├── start_public.sh        # 公网启动脚本
└── README.md              # 项目说明文档
```

## 注意事项

- 请确保两张图片的人物姿态和角度尽量一致，以获得最佳融合效果。
- 模型文件较大，请留意仓库大小限制（如果使用 Git LFS 请按需配置）。
