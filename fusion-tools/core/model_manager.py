import os
import threading
import numpy as np
import onnxruntime as ort
from config import (
    BISENET_ONNX_PATH, MODNET_ONNX_PATH,
    ONNX_INTER_OP_THREADS, ONNX_INTRA_OP_THREADS,
)


class _ONNXModel:
    """单个 ONNX 模型的线程安全封装"""

    def __init__(self, model_path: str, name: str):
        self._path = model_path
        self._name = name
        self._session = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._input_name = None
        self._output_name = None

    def _create_session(self) -> ort.InferenceSession:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"{self._name} 模型文件不存在: {self._path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.inter_op_num_threads = ONNX_INTER_OP_THREADS
        opts.intra_op_num_threads = ONNX_INTRA_OP_THREADS
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        session = ort.InferenceSession(self._path, sess_options=opts, providers=providers)
        print(f"{self._name} 已加载: {self._path} | {session.get_providers()}")
        return session

    def get_session(self) -> ort.InferenceSession:
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is None:
                self._session = self._create_session()
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
        return self._session

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        session = self.get_session()
        with self._infer_lock:
            return session.run([self._output_name], {self._input_name: tensor})[0]

    def warmup(self, dummy_shape: tuple):
        self.get_session()
        dummy = np.random.randn(*dummy_shape).astype(np.float32)
        self.infer(dummy)
        print(f"{self._name} 预热完成")


class ModelManager:
    """双模型管理器（单例）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.bisenet = _ONNXModel(BISENET_ONNX_PATH, "BiSeNet")
        self.modnet = _ONNXModel(MODNET_ONNX_PATH, "MODNet")
        self._initialized = True

    def warmup(self):
        self.bisenet.warmup((1, 3, 512, 512))
        self.modnet.warmup((1, 3, 512, 512))

    def get_bisenet_session(self) -> ort.InferenceSession:
        return self.bisenet.get_session()

    def get_modnet_session(self) -> ort.InferenceSession:
        return self.modnet.get_session()
