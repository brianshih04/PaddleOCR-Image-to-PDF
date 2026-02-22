import logging
import time
import shutil
import gc
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from rasterizer import load_data
from ocr_engine import PaddleOcrEngine
from coord_mapper import CoordMapper
from pdf_writer import PDFReconstructor
from classifier import DocumentClassifier

logger = logging.getLogger(__name__)

class OCRWorker(QThread):
    """
    Background Thread handling heavy OCR / NLP computations to prevent UI Blocking.
    """
    # Signals to communicate to the GUI layer safely
    log_emitted = Signal(str)
    file_started = Signal(str)
    progress_updated = Signal(int, int) # Current page, Total pages
    task_finished = Signal(bool)
    
    def __init__(self, queue: list[str], output_dir: str, lang: str, 
                 enable_classification: bool, hot_folder: str = None, 
                 det_model: Path = None, rec_models_dir: Path = None, 
                 dict_models_dir: Path = None, nlp_model_dir: Path = None, rules_path: Path = None, 
                 hw_device: str = "Auto", parent=None):
        super().__init__(parent)
        self.queue = queue
        self.output_dir = Path(output_dir) if output_dir else None
        self.lang = lang
        self.enable_classification = enable_classification
        self.hot_folder = Path(hot_folder) if hot_folder else None
        
        self.det_model = det_model
        self.rec_models_dir = rec_models_dir
        self.dict_models_dir = dict_models_dir
        
        self.nlp_model_dir = nlp_model_dir
        self.rules_path = rules_path
        self.hw_device = hw_device
        
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.log_emitted.emit("[SYSTEM] Abort signal received. Cleaning up after current page...")

    def run(self):
        self.log_emitted.emit(">>> Initialization Started <<<")
        
        try:
            # Reconstruct the models_dir structure conceptually
            models_dir = self.det_model.parent if self.det_model else None
            engine = PaddleOcrEngine(str(models_dir), device=self.hw_device)
            engine.load_recognizer(self.lang)
            
            classifier = None
            if self.enable_classification:
                classifier = DocumentClassifier(str(self.rules_path), str(self.nlp_model_dir))
                
            active_hw = engine.get_active_provider()
            self.log_emitted.emit(f"Engine Ready. Hardware Accelerator: [{active_hw}]")
            self.log_emitted.emit(f"Found {len(self.queue)} files to process.")
        except Exception as e:
            self.log_emitted.emit(f"[ERROR] Engine Init Failed: {str(e)}")
            self.task_finished.emit(False)
            return

        for filepath_str in self.queue:
            if not self._is_running:
                break
                
            input_path = Path(filepath_str)
            self.file_started.emit(input_path.name)
            self.log_emitted.emit(f"\n--- Processing: {input_path.name} ---")
            
            # Decide the absolute target directory
            target_base_dir = self.output_dir if self.output_dir else input_path.parent
            
            # Enforce .pdf output extension regardless of input
            # If input is already pdf, we suffix it to avoid clash before cleanup
            temp_output_path = target_base_dir / f"{input_path.stem}_ocr_{int(time.time())}.pdf"
            
            try:
                pages = load_data(input_path, dpi=300)
                total_pages = len(pages)
                self.progress_updated.emit(0, total_pages)
                
                pdf_gen = PDFReconstructor()
                full_text_buffer = []
                
                for i, page in enumerate(pages):
                    if not self._is_running:
                        break
                        
                    self.log_emitted.emit(f"Parsing page {i+1}/{total_pages}...")
                    
                    page_start_time = time.perf_counter()
                    
                    polygons = engine.detect_text_polygons(page.image_rgb)
                    
                    page_text = []
                    extracted_blocks = []
                    crop_list = []
                    bbox_list = []
                    
                    for poly in polygons:
                        x_min, y_min, x_max, y_max = CoordMapper.polygon_to_orthogonal_bbox(poly)
                        h, w = page.image_rgb.shape[:2]
                        y0, y1 = max(0, int(y_min)), min(h, int(y_max))
                        x0, x1 = max(0, int(x_min)), min(w, int(x_max))
                        
                        cropped = page.image_rgb[y0:y1, x0:x1]
                        if min(cropped.shape[:2]) > 0:
                            crop_list.append(cropped)
                            bbox_list.append((x_min, y_min, x_max, y_max))
                            
                    if crop_list:
                        batch_results = engine.recognize_text_batch(crop_list, batch_size=12)
                        for (text, conf), bbox in zip(batch_results, bbox_list):
                            if text.strip() != "":
                                extracted_blocks.append((text, bbox))
                                page_text.append(text)
                                
                    pdf_gen.add_page(page.image_rgb, extracted_blocks)
                    
                    page_end_time = time.perf_counter()
                    elapsed = page_end_time - page_start_time
                    self.log_emitted.emit(f"Page {i+1} completed in {elapsed:.2f}s")
                    
                    # Accumulate text for classification (limited to sensible amounts)
                    if len(full_text_buffer) < 2000:
                        full_text_buffer.append(" ".join(page_text))
                        
                    del page
                    gc.collect()
                    self.progress_updated.emit(i+1, total_pages)
                
                if not self._is_running:
                     continue
                
                # NLP Classification Pipeline
                label = "Uncategorized"
                if self.enable_classification and classifier:
                    self.log_emitted.emit("Running Auto-Classification Track...")
                    combined_text = " ".join(full_text_buffer)
                    label = classifier.classify_text(combined_text)
                    self.log_emitted.emit(f"Assigned Classification Label: [{label}]")
                
                # Reconstruct Output PDF path inside classification folder
                final_folder = target_base_dir / label
                final_folder.mkdir(parents=True, exist_ok=True)
                
                final_output_path = final_folder / f"{input_path.stem}_searchable.pdf"
                
                # Build Metadata & Save
                pdf_gen.set_metadata(label, self.lang)
                pdf_gen.save(temp_output_path)
                
                # Atomic Move
                shutil.move(str(temp_output_path), str(final_output_path))
                
                # Cleanup Original Input Image from the monitoring hot folder
                if self.hot_folder and self.hot_folder in input_path.parents:
                    backup_folder = self.hot_folder / "_Backup" / label
                    backup_folder.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(input_path), str(backup_folder / input_path.name))
                    self.log_emitted.emit(f"Moved original file {input_path.name} to Backup")
                
                self.log_emitted.emit(f"OK! Saved as: {final_output_path.name}")
                    
            except Exception as e:
                self.log_emitted.emit(f"[ERROR] Failed to process {input_path.name}: {str(e)}")
                # Deadlock Avoidance: Clear failed files from hot folder
                if self.hot_folder and self.hot_folder in input_path.parents:
                    error_folder = self.hot_folder / "_Error"
                    error_folder.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(str(input_path), str(error_folder / input_path.name))
                        self.log_emitted.emit(f"Deadlock avoided: Moved faulting file to {error_folder}")
                    except Exception as move_e:
                        self.log_emitted.emit(f"[CRITICAL] Could not move faulting file: {move_e}")
                            
        # Gracefully release C++ memory handlers inside the worker thread
        if 'engine' in locals():
            del engine
        if 'classifier' in locals():
            del classifier
        gc.collect()
        
        self.task_finished.emit(True)
        self.log_emitted.emit(">>> Queue Finished <<<")
