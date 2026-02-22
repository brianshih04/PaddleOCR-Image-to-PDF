import os
import sys
from pathlib import Path

def get_resource_path(relative_path: str) -> Path:
    """
    Get the absolute path to a resource.
    Works for standard development mode and PyInstaller's static --windowed execution.
    PyInstaller unpacks data to a temporary folder stored in sys._MEIPASS.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Normal python execution
        # We assume the working directory is the repository root or src/
        base_path = Path(os.path.abspath("."))
        
    target = base_path / relative_path
    
    # Simple fallback check if we are in src/
    if not target.exists() and base_path.name == "src":
        target = base_path.parent / relative_path
        
    return target
