import asyncio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from config import MAX_UPLOAD_SIZE_MB, MAX_CONCURRENT_WORKERS
from core.pipeline import download_image_from_url

router = APIRouter(prefix="/api")
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)


async def _read_image(file: UploadFile | None, url: str | None, name: str) -> np.ndarray:
    if file is not None:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(400, f"{name} 文件大小超过 {MAX_UPLOAD_SIZE_MB}MB 限制")
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"{name} 不是有效的图片文件")
        return img
    if url:
        try:
            return download_image_from_url(url)
        except Exception as e:
            raise HTTPException(400, f"{name} URL 下载失败: {e}")
    raise HTTPException(400, f"请提供 {name} 的文件或 URL")


@router.post("/fusion")
async def fusion_endpoint(
    fg: UploadFile | None = File(None),
    bg: UploadFile | None = File(None),
    fg_url: str | None = Form(None),
    bg_url: str | None = Form(None),
):
    img1 = await _read_image(fg, fg_url, "前景图(fg)")
    img2 = await _read_image(bg, bg_url, "底图(bg)")

    from core.pipeline import process_fusion
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, process_fusion, img1, img2)

    if not result.success:
        return JSONResponse(status_code=422, content={"error": result.message})

    response = {"message": result.message, "algorithms": {}}
    for key, data in result.images.items():
        response["algorithms"][key] = {
            "png": f"/api/download?path={data['png']}",
            "jpg": f"/api/download?path={data['jpg']}",
            "watermark": f"/api/download?path={data['watermark']}",
        }
    return response


@router.get("/download")
async def download_file(path: str):
    """下载融合结果文件"""
    import os
    import tempfile
    # 安全检查：只允许下载 temp 目录下的 fusion_ 文件
    temp_dir = tempfile.gettempdir()
    real_path = os.path.realpath(path)
    if not real_path.startswith(os.path.realpath(temp_dir) + os.sep):
        raise HTTPException(403, "禁止访问该路径")
    basename = os.path.basename(real_path)
    if not basename.startswith("fusion_"):
        raise HTTPException(403, "禁止访问非融合结果文件")
    if not os.path.exists(real_path):
        raise HTTPException(404, "文件不存在或已过期")
    return FileResponse(real_path, filename=basename)


@router.get("/health")
async def health():
    from core.model_manager import ModelManager
    try:
        manager = ModelManager()
        bisenet = manager.bisenet.get_session()
        modnet = manager.modnet.get_session()
        return {
            "status": "ok",
            "bisenet_loaded": bisenet is not None,
            "modnet_loaded": modnet is not None,
            "providers": bisenet.get_providers() if bisenet else [],
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})
