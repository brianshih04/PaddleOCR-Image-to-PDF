import json
import logging
import re
from pathlib import Path
import numpy as np

# We import onnxruntime conditionally or locally to ensure gc can wipe it
# tokenizers is used for blazing fast BPE/WordPiece
from tokenizers import Tokenizer
import onnxruntime as ort

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """
    Dual-Track NLP Classification Engine.
    Track 1: High-Priority Regex Rule Engine.
    Track 2: Lightweight ONNX Transformer (e.g. MiniLM-L6) with Softmax Confidence threshold.
    """
    def __init__(self, rules_path: str, nlp_model_dir: str):
        self.rules_path = Path(rules_path)
        self.nlp_model_dir = Path(nlp_model_dir)
        self.rules = self._load_rules()
        self.threshold = 0.6
        
    def _load_rules(self) -> list[dict]:
        if not self.rules_path.exists():
            return []
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            return []

    def classify_text(self, text: str) -> str:
        """
        Main entry point. Automatically performs truncation and dual-track resolution,
        returning the predicted Label string.
        """
        if not text or text.strip() == "":
            return "Uncategorized"
            
        # 1. Regex Fast-Track
        fast_result = self._apply_rules(text)
        if fast_result:
            logger.info(f"Fast-Track Matched: {fast_result}")
            return fast_result
            
        # 2. NLP Slow-Track (ONNX Fallback)
        try:
            nlp_result = self._apply_nlp(text)
            return nlp_result
        except Exception as e:
            logger.exception("NLP Tracking failed, returning Uncategorized.")
            return "Uncategorized"

    def _apply_rules(self, text: str) -> str | None:
        """
        Track 1: Regex rules matching. E.g. {"label": "Invoice", "keywords": ["INVOICE", "發票"]}
        """
        for rule in self.rules:
            label = rule.get("label")
            keywords = rule.get("keywords", [])
            for kw in keywords:
                if re.search(kw, text, re.IGNORECASE):
                    return label
        return None

    def _apply_nlp(self, text: str) -> str:
        """
        Track 2: Tokenizer + ONNX Transformer Inference.
        """
        tokenizer_path = self.nlp_model_dir / "tokenizer.json"
        model_path = self.nlp_model_dir / "model.onnx"
        labels_path = self.nlp_model_dir / "labels.json"
        
        if not tokenizer_path.exists() or not model_path.exists():
            logger.warning("NLP Engine not configured. Skipping fallback.")
            return "Uncategorized"
            
        # Create Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Enforce exact max length truncation to strictly bound inference overhead
        tokenizer.enable_truncation(max_length=512)
        tokenizer.enable_padding(length=512)
        
        encoded = tokenizer.encode(text)
        
        # Load constraints
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1 # Keep NLP single threaded to yield to UI
        
        session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=['CPUExecutionProvider'])
        
        # Prepare inputs based on typical HuggingFace standard
        inputs = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64)
        }
        
        # Inference
        logits = session.run(None, inputs)[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        max_idx = np.argmax(probs)
        confidence = probs[max_idx]
        
        # Guaranteed teardown to reclaim RAM instantly 
        del session
        del inputs
        import gc
        gc.collect()
        
        if confidence < self.threshold:
            logger.info(f"NLP Confidence {confidence:.2f} below threshold, mark as Uncategorized")
            return "Uncategorized"
            
        # Parse labels map if provided, otherwise return string index
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_map = json.load(f)
                return labels_map.get(str(max_idx), f"Class_{max_idx}")
        except:
            return f"Class_{max_idx}"
