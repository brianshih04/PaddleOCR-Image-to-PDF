import logging
import time
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

# Extensions whitelist
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

class HotFolderHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def is_valid_file(self, path_str: str) -> bool:
        path = Path(path_str)
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return False
        # Filter temp files
        if path.name.startswith("~$") or path.suffix.lower() == ".tmp":
            return False
        return True

    def on_created(self, event):
        if not event.is_directory and self.is_valid_file(event.src_path):
            self.callback(event.src_path)

    def on_moved(self, event):
        # Applies when a file is mapped/copied fully into the folder and triggers moved instead of created
        if not event.is_directory and self.is_valid_file(event.dest_path):
            self.callback(event.dest_path)


class DirectoryObserverThread(QThread):
    """
    Background Daemon utilizing watchdog to monitor the Hot Folder.
    Safely probes file locks to prevent race conditions during scanner streaming.
    """
    file_detected = Signal(str, str) # filepath, source='[Auto]'
    
    def __init__(self, hot_folder: str):
        super().__init__()
        self.hot_folder = hot_folder
        self.observer = None
        self._is_running = True

    def run(self):
        if not self.hot_folder or not Path(self.hot_folder).exists():
            logger.error("Hot folder invalid. Monitor stopping.")
            return

        logger.info(f"Starting Hot Folder Daemon on: {self.hot_folder}")
        handler = HotFolderHandler(self._handle_new_file)
        self.observer = Observer()
        self.observer.schedule(handler, self.hot_folder, recursive=False)
        self.observer.start()

        try:
            while self._is_running:
                time.sleep(1)
        finally:
            self.observer.stop()
            self.observer.join()
            logger.info("Hot Folder Daemon terminated.")

    def stop(self):
        self._is_running = False

    def _wait_for_file_unlock(self, filepath: str, timeout: int = 30) -> bool:
        """
        Probes the file using exclusive append mode to detect OS file locks.
        This evades partial-read crases from slow network scanners.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._is_running:
                return False
            try:
                # Attempt to open exclusively
                with open(filepath, 'a'):
                    pass
                return True # Success, file is fully released by OS
            except PermissionError:
                # Locked by another process (likely writing)
                time.sleep(1)
            except FileNotFoundError:
                # File deleted mid-flight
                return False
        return False

    def _handle_new_file(self, filepath: str):
        """
        Callback fired by Watchdog Handler -> Probe lock -> Emit Signal to UI
        """
        logger.info(f"Daemon Detected: {filepath}. Probing lock...")
        if self._wait_for_file_unlock(filepath):
            logger.info(f"File unlocked. Dispatching to auto-queue: {filepath}")
            self.file_detected.emit(filepath, "[Auto]")
        else:
            logger.warning(f"File lock probe timed out or failed for: {filepath}")
