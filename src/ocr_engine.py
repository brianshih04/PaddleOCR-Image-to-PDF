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
        self.active_provider = self.det_session.get_providers()[0] if self.det_session else "None"
        
    def get_active_provider(self) -> str:
        return self.active_provider
        
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
            logger.error(f"Recognition model missing: {rec_model_path}. Attempting to fallback to 'ch'.")
            if lang != "ch":
                self.current_lang = None
                return self.load_recognizer("ch")
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
        
        # Binarization
        mask = (prob_map > 0.3).astype(np.uint8) * 255
        
        # Contour generation
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        import pyclipper
        from shapely.geometry import Polygon
        
        polys = []
        for contour in contours:
            if contour.shape[0] < 4:
                continue
                
            box = cv2.minAreaRect(contour)
            points = cv2.boxPoints(box)
            
            poly = Polygon(points)
            if not poly.is_valid or poly.area < 5 or poly.length == 0:
                continue
                
            # Unclip (dilation) to get full text region
            distance = poly.area * 1.5 / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            
            if len(expanded) == 0:
                continue
                
            expanded_points = np.array(expanded[0])
            box = cv2.minAreaRect(expanded_points)
            final_points = cv2.boxPoints(box)
            
            # Rescale points to original image
            ratio_h = h / new_h
            ratio_w = w / new_w
            
            scaled_points = [[float(pt[0] * ratio_w), float(pt[1] * ratio_h)] for pt in final_points]
            polys.append(scaled_points)
            
        return polys
        
    def recognize_text_batch(self, cropped_images: list[np.ndarray], batch_size: int = 6) -> list[tuple[str, float]]:
        """
        Stage 2b: Text Recognition (SVTR_LCNet) batched inference.
        """
        if self.rec_session is None or not cropped_images:
            return [("", 0.0)] * len(cropped_images)
            
        import math
        results = []
        
        for i in range(0, len(cropped_images), batch_size):
            batch_crops = cropped_images[i:i+batch_size]
            max_w = 1
            processed_crops = []
            
            for crop in batch_crops:
                h, w = crop.shape[:2]
                if h == 0 or w == 0:
                    processed_crops.append(None)
                    continue
                    
                ratio = w / float(h)
                new_w = max(int(math.ceil(48 * ratio)), 1)
                max_w = max(max_w, new_w)
                
                tensor = preprocess_image_to_tensor(crop, (48, new_w))
                processed_crops.append(tensor)
                
            batch_tensor = np.zeros((len(batch_crops), 3, 48, max_w), dtype=np.float32)
            valid_indices = []
            
            for j, tensor in enumerate(processed_crops):
                if tensor is not None:
                    c_w = tensor.shape[3]
                    batch_tensor[j, :, :, :c_w] = tensor[0] # copy and implicitly pad right with 0s
                    valid_indices.append(j)
                    
            if not valid_indices:
                results.extend([("", 0.0)] * len(batch_crops))
                continue
                
            pred = self.rec_session.run(None, {self.rec_input_name: batch_tensor})[0]
            
            for j in range(len(batch_crops)):
                if j not in valid_indices:
                    results.append(("", 0.0))
                    continue
                    
                preds_idx = pred[j].argmax(axis=1)
                preds_prob = pred[j].max(axis=1)
                
                text = ""
                confs = []
                for k in range(len(preds_idx)):
                    idx = preds_idx[k]
                    if idx == 0 or idx >= len(self.char_list):
                        continue
                    if k > 0 and idx == preds_idx[k - 1]:
                        continue
                        
                    text += self.char_list[idx]
                    confs.append(float(preds_prob[k]))
                    
                conf = sum(confs) / len(confs) if confs else 0.0
                results.append((text, conf))
                
        return results
