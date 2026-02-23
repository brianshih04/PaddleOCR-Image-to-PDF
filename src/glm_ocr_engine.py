import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image

# For falling back to Paddle's Detection module
from ocr_engine import PaddleOcrEngine

logger = logging.getLogger(__name__)

class GlmOcrEngine(PaddleOcrEngine):
    """
    Wrapper for ZhipuAI's GLM-OCR Model (Local).
    Inherits from PaddleOcrEngine to reuse DBNet for Text Detection.
    Overrides recognize_text_batch to use GLM-OCR via Transformers.
    """
    def __init__(self, models_dir: str, device: str = "Auto"):
        # Initialize parent to load the Paddle DBNet detector
        super().__init__(models_dir, device)
        
        self.glm_model = None
        self.glm_processor = None
        self.active_provider = f"GLM-OCR (Local) on {self.device}"
        logger.info(f"Initializing {self.active_provider}")
        
    def get_active_provider(self) -> str:
        return self.active_provider
        
    def load_recognizer(self, lang: str):
        """
        Dynamically loads the GLM-OCR local model (0.9B parameters).
        Language parameter is ignored as GLM is natively multilingual.
        """
        if self.glm_model is not None:
            return
            
        self.current_lang = lang
        logger.info("[GLM-OCR] Loading HuggingFace model 'zai-org/GLM-OCR'...")
        
        try:
            # We must import transformers inside the thread to avoid premature loading
            from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
            
            # Map GUI logical device down to PyTorch device
            pt_dev = "cuda" if "GPU" in self.device else "cpu"
            
            self.glm_processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
            self.glm_model = AutoModelForCausalLM.from_pretrained(
                "zai-org/GLM-OCR",
                trust_remote_code=True,
                torch_dtype=torch.float32 if pt_dev == "cpu" else torch.bfloat16,
                device_map="auto" if pt_dev == "cuda" else None
            ).eval()
            
            if pt_dev == "cpu":
                self.glm_model = self.glm_model.to("cpu")
                
            logger.info("[GLM-OCR] 0.9B Local Model loaded successfully.")
        except Exception as e:
            logger.error(f"[GLM-OCR] Failed to load local transformer model: {e}")
            self.glm_model = None
        
    def recognize_text_batch(self, cropped_images: list[np.ndarray], batch_size: int = 6) -> list[tuple[str, float]]:
        """
        Stage 2: Text Recognition. Overridden to transcribe crops using GLM-OCR locally.
        Runs the standard processor -> generate pipeline for each cropped paragraph/word.
        """
        if self.glm_model is None or not cropped_images:
            return [("", 0.0)] * len(cropped_images)
            
        results = []
        
        for crop in cropped_images:
            if min(crop.shape[:2]) == 0:
                results.append(("", 0.0))
                continue
                
            try:
                # Convert OpenCV BGR/RGB crop to PIL Image
                pil_img = Image.fromarray(crop)
                
                # Zhipu standard prompt for simple text reading
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Recognize the text:"}]}]
                
                inputs = self.glm_processor(
                    images=[pil_img],
                    text=self.glm_processor.apply_chat_template(msgs, add_generation_prompt=True),
                    return_tensors="pt"
                ).to(self.glm_model.device)
                
                # Generate
                with torch.no_grad():
                    output_ids = self.glm_model.generate(**inputs, max_new_tokens=512)
                    
                # Extract only the generated tokens
                generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
                text = self.glm_processor.decode(generated_ids, skip_special_tokens=True).strip()
                
                results.append((text, 1.0)) # GLM doesn't easily return conf scores, fallback to 1.0
                
            except Exception as e:
                logger.error(f"[GLM-OCR] Crop recognition failed: {e}")
                results.append(("", 0.0))
                
        return results
