import logging

logger = logging.getLogger(__name__)

class CoordMapper:
    """
    Stage 3: Coordinate Transform and Rectification.
    Converts raw Text Detection Polygons to standard Bounding Boxes
    and maintains the 1:1 Pixel-to-Point absolute mapping logic.
    """
    @staticmethod
    def polygon_to_orthogonal_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
        """
        Extracts orthogonal bounding box [x_min, y_min, x_max, y_max] from four point polygon.
        """
        x_coords = [pt[0] for pt in polygon]
        y_coords = [pt[1] for pt in polygon]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return float(x_min), float(y_min), float(x_max), float(y_max)
    
    @staticmethod
    def map_pixels_to_points(bbox_pixels: tuple[float, float, float, float], 
                             scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
        """
        Maps pixel bounding box to PDF point coordinates.
        1:1 projection when image width matches PDF width.
        """
        x_min, y_min, x_max, y_max = bbox_pixels
        return (x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y)
