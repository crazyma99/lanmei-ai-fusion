import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from contextlib import asynccontextmanager
from fastapi import FastAPI
from config import SERVER_HOST, SERVER_PORT
from core.model_manager import ModelManager
from core.pipeline import start_cleanup_timer
from api.routes import router as api_router
from ui.gradio_app import create_gradio_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在加载模型...")
    try:
        manager = ModelManager()
        manager.bisenet.get_session()
        manager.modnet.get_session()
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载警告: {e}")
    start_cleanup_timer()
    print(f"服务就绪 | http://{SERVER_HOST}:{SERVER_PORT}/")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="蓝梅 AI 智能叠图工具", version="2.0.0", lifespan=lifespan)
    app.include_router(api_router)
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
