import gradio as gr
import cv2
import numpy as np

from core.pipeline import process_fusion
from core.segmentation import modnet_alpha_matte


def _preview_segmentation(img1_pil, enable_preview):
    if img1_pil is None or not enable_preview:
        return None
    img1 = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
    try:
        alpha = modnet_alpha_matte(img1)
    except Exception as e:
        print(f"MODNet 预览失败: {e}")
        return None
    h, w = img1.shape[:2]
    block = 20
    rows = np.arange(h) // block
    cols = np.arange(w) // block
    parity = (rows[:, None] + cols[None, :]) % 2
    checker = np.where(parity[:, :, None], np.uint8([255, 255, 255]), np.uint8([200, 200, 200]))
    alpha_f = alpha.astype(np.float32) / 255.0
    alpha_3 = np.stack([alpha_f] * 3, axis=2)
    img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32)
    result = (img_rgb * alpha_3 + checker.astype(np.float32) * (1 - alpha_3)).astype(np.uint8)
    return result


def _main_process(img1_pil, img2_pil, progress=gr.Progress()):
    if img1_pil is None or img2_pil is None:
        return None, None, None, None, "请先上传两张图片。"

    img1 = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2BGR)

    result = process_fusion(img1, img2, progress_callback=progress)

    if not result.success:
        return None, None, None, None, result.message

    before_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    def _extract(key):
        if key not in result.images:
            return None, None
        data = result.images[key]
        return [before_rgb, data["rgb"]], [data["png"], data["jpg"], data["watermark"]]

    comp_a, files_a = _extract("algo_a")
    comp_b, files_b = _extract("algo_b")

    return comp_a, files_a, comp_b, files_b, result.message


def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="蓝梅 AI 智能叠图工具") as demo:
        gr.Markdown("# 蓝梅 AI 智能人像叠图工具")
        gr.Markdown("自动抠图 | 像素级对齐 | 双算法融合")

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                input_fg = gr.Image(type="pil", label="图1：人物特写/半身 (将被抠图)", height=240)
                input_bg = gr.Image(type="pil", label="图2：底图/全景 (背景参考)", height=240)
                preview_toggle = gr.Checkbox(label="预览 MODNet 抠图", value=False)
                preview_img = gr.Image(label="MODNet 人像分割预览", interactive=False)
                btn_run = gr.Button("开始叠图", variant="primary")

            with gr.Column(scale=4, min_width=520):
                output_msg = gr.Textbox(label="状态信息")
                with gr.Tabs():
                    with gr.Tab("方案A：泊松融合（主）"):
                        output_img_a = gr.Gallery(label="对比预览 (底图/融合)", columns=2)
                        output_file_a = gr.File(label="下载 (PNG/JPG/水印)", height=100)
                    with gr.Tab("方案B：分频融合（备选）"):
                        output_img_b = gr.Gallery(label="对比预览 (底图/融合)", columns=2)
                        output_file_b = gr.File(label="下载 (PNG/JPG/水印)", height=100)

        btn_run.click(
            fn=_main_process,
            inputs=[input_fg, input_bg],
            outputs=[output_img_a, output_file_a, output_img_b, output_file_b, output_msg],
        )

        for component in [input_fg, preview_toggle]:
            component.change(
                fn=_preview_segmentation,
                inputs=[input_fg, preview_toggle],
                outputs=[preview_img],
            )

    return demo
