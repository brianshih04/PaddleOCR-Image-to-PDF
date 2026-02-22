import cv2
import sys
from pathlib import Path
from src.ocr_engine import PaddleOcrEngine

def main():
    base_dir = Path(__file__).parent
    engine = PaddleOcrEngine(str(base_dir / "models"))
    engine.load_recognizer("ch")
    
    from src.rasterizer import load_data
    test_pdf = base_dir.parent / "glm-ocr-engine" / "1.pdf"
    pages = load_data(str(test_pdf), dpi=300)
    img = pages[0].image_rgb
    
    polys = engine.detect_text_polygons(img)
    print(f"Found {len(polys)} polygons.")
    
    if not polys:
        return
        
    crop_list = []
    
    from src.coord_mapper import CoordMapper
    for i, poly in enumerate(polys[:5]):
        x_min, y_min, x_max, y_max = CoordMapper.polygon_to_orthogonal_bbox(poly)
        h, w = img.shape[:2]
        y0, y1 = max(0, int(y_min)), min(h, int(y_max))
        x0, x1 = max(0, int(x_min)), min(w, int(x_max))
        cropped = img[y0:y1, x0:x1]
        
        crop_list.append(cropped)
        cv2.imwrite(f"crop_{i}.jpg", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        
    results = engine.recognize_text_batch(crop_list)
    for i, (text, conf) in enumerate(results):
        print(f"Crop {i} -> Text: '{text}' [Conf: {conf:.2f}]")

if __name__ == "__main__":
    main()
