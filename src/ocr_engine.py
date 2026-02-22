import logging
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

from hw_detect import get_optimal_providers, get_physical_cpu_cores

logger = logging.getLogger(__name__)


def create_session(model_path: str | Path, providers: list[str]) -> ort.InferenceSession:
    """
    Initializes an ONNX Runtime InferenceSession with strictly controlled
    memory and thread parameters to ensure predictable performance and bounded VRAM.
    """
    model_path = str(model_path)
    
    # Session Options Configuration
    sess_opts = ort.SessionOptions()
    
    # 1. Enable full graph optimizations
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 2. Control memory allocation strategy (kSameAsRequested essentially prevents excessive Arena Growth)
    # We use session config entries to bind dynamic block base.
    sess_opts.add_session_config_entry("session.dynamic_block_base", "4")
    
    # 3. CPU execution threading - avoid logical core over-subscription
    physical_cores = get_physical_cpu_cores()
    sess_opts.intra_op_num_threads = physical_cores
    
    # If the user specifically configures sequential execution instead of parallel
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    logger.info(f"Loading ONNX model from: {model_path}")
    logger.debug(f"ORT Session intra_op_threads: {physical_cores}")
    
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    
    # Validate actual applied providers
    logger.debug(f"Model executed by providers: {session.get_providers()}")
    return session


def preprocess_image_to_tensor(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Standardize the OpenCV input to the specific (N, C, H, W) float32 Contiguous Array required by ORT.
    """
    # Resize assuming `shape` is (height, width)
    resized = cv2.resize(img, (shape[1], shape[0]))
    
    # Normalize (typically ImageNet mean/std or specific for PP-OCR)
    # PaddleOCR defaults: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    resized = resized.astype(np.float32) / 255.0
    resized -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    resized /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # HWC to CHW
    chw = np.transpose(resized, (2, 0, 1))
    
    # Add batch dimension NCHW
    nchw = np.expand_dims(chw, axis=0)
    
    # CRITICAL: Force memory contiguity before feeding to C++ ORT Core
    tensor = np.ascontiguousarray(nchw, dtype=np.float32)
    return tensor


class PaddleOcrEngine:
    """
    Dual-stage high-speed inference wrapper for DBNet Detection and SVTR_LCNet Recognition models.
    Supports dynamic language switching for the Recognition model.
    """
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.providers = get_optimal_providers()
        
        # Load Detection (DBNet) - Language Independent
        det_path = self.models_dir / "det.onnx"
        if not det_path.exists():
            logger.warning(f"Detection model not found at {det_path}")
            
        self.det_session = create_session(det_path, self.providers) if det_path.exists() else None
        self.det_input_name = self.det_session.get_inputs()[0].name if self.det_session else None
        
        # Recognition state
        self.rec_session = None
        self.rec_input_name = None
        self.char_list = []
        self.current_lang = None
        
    def load_recognizer(self, lang: str):
        """
        Dynamically loads the recognition model and dictionary for the specified language.
        Destroys the previous session to free VRAM.
        """
        if lang == self.current_lang and self.rec_session is not None:
            return # Already loaded
            
        logger.info(f"Switching OCR Recognizer to language: {lang}")
        
        # 1. Destroy existing session to free memory
        if self.rec_session is not None:
            del self.rec_session
            self.rec_session = None
            
        import gc
        gc.collect()
        
        # 2. Map language to model/dict paths
        # Examples: "ch" -> ch_PP-OCRv4_rec.onnx, "chinese_cht" -> chinese_cht_PP-OCRv3_rec.onnx
        if lang == "ch":
            model_name = "ch_PP-OCRv4_rec.onnx"
            dict_name = "ppocr_keys_v1.txt"
        elif lang == "chinese_cht":
            model_name = "chinese_cht_PP-OCRv3_rec.onnx"
            dict_name = "chinese_cht_dict.txt"
        elif lang == "japan":
            model_name = "japan_PP-OCRv3_rec.onnx"
            dict_name = "japan_dict.txt"
        elif lang == "korean":
            model_name = "korean_PP-OCRv3_rec.onnx"
            dict_name = "korean_dict.txt"
        elif lang == "latin":
            model_name = "latin_PP-OCRv3_rec.onnx"
            dict_name = "latin_dict.txt"
        elif lang == "en":
            model_name = "en_PP-OCRv3_rec.onnx"
            dict_name = "en_dict.txt"
        else:
            # Fallback
            logger.warning(f"Unknown language '{lang}', falling back to 'ch'")
            model_name = "ch_PP-OCRv4_rec.onnx"
            dict_name = "ppocr_keys_v1.txt"
            lang = "ch"
            
        rec_model_path = self.models_dir / "rec" / model_name
        dict_path = self.models_dir / "dict" / dict_name
        
        if not rec_model_path.exists():
            logger.error(f"Recognition model missing: {rec_model_path}")
            return
            
        # 3. Load new session
        self.rec_session = create_session(rec_model_path, self.providers)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        
        # 4. Load Dictionary
        if dict_path.exists():
            self._load_dict(str(dict_path))
        else:
            logger.warning(f"Dictionary missing: {dict_path}")
            
        self.current_lang = lang
        
    def _load_dict(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        self.char_list = [line.strip("\n") for line in lines]
        self.char_list.insert(0, "blank")
        # Add space
        self.char_list.append(" ")
        
    def detect_text_polygons(self, image_rgb: np.ndarray):
        """
        Stage 2a: Text Detection (DBNet).
        Outputs four-point polygon coordinates: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        if self.det_session is None:
            return []
            
        # Dynamic resizing to nearest multiples of 32 for DBNet
        h, w = image_rgb.shape[:2]
        new_h = int(max(round(h / 32) * 32, 32))
        new_w = int(max(round(w / 32) * 32, 32))
        
        det_input = preprocess_image_to_tensor(image_rgb, (new_h, new_w))
        
        # ORT Inference
        pred = self.det_session.run(None, {self.det_input_name: det_input})[0]
        
        prob_map = pred[0, 0, :, :]
        # TODO: Binarization, contour generation, Box clipping via pyclipper
        
        polys = [] # Placeholder
        return polys
        
    def recognize_text(self, cropped_image: np.ndarray) -> tuple[str, float]:
        """
        Stage 2b: Text Recognition (SVTR_LCNet) applied to a perspectively wrapped crop.
        """
        if self.rec_session is None:
            return "", 0.0
            
        # Standardize Recognition shape (H=48, W=dynamic/320)
        rec_input = preprocess_image_to_tensor(cropped_image, (48, 320))
        
        # ORT Inference
        pred = self.rec_session.run(None, {self.rec_input_name: rec_input})[0]
        
        # CTC Decoding
        preds_idx = pred.argmax(axis=2)[0]
        preds_prob = pred.max(axis=2)[0]
        
        text = ""
        conf = 0.0
        # TODO: CTC Merge duplicate consecutive chars
        
        return text, conf
