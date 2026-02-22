import sys
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QProgressBar, QTextEdit,
    QFileDialog, QComboBox, QCheckBox, QGroupBox, QTableWidget,
    QTableWidgetItem, QDialog, QMessageBox, QHeaderView
)
from PySide6.QtCore import Qt, QSettings, Signal

# Import custom architecture components
from gui_utils import get_resource_path
from gui_worker import OCRWorker
from monitor import DirectoryObserverThread


class RuleManagerDialog(QDialog):
    """
    Dialog to manage rules.json for the regex fast-track of the Auto-Classification module.
    """
    def __init__(self, rules_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分類規則管理器 (Rule Manager)")
        self.setMinimumSize(500, 300)
        self.rules_path = rules_path
        self.rules = []
        
        self.layout = QVBoxLayout(self)
        
        # Table
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Label (類別名稱)", "Keywords (觸發關鍵字, 逗號分隔)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("新增規則")
        self.add_btn.clicked.connect(self.add_row)
        self.save_btn = QPushButton("儲存變更")
        self.save_btn.clicked.connect(self.save_rules)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.save_btn)
        self.layout.addLayout(btn_layout)
        
        self.load_rules()
        
    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("New_Category"))
        self.table.setItem(row, 1, QTableWidgetItem("keyword1, keyword2"))
        
    def load_rules(self):
        if self.rules_path.exists():
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    self.rules = json.load(f)
                    
                self.table.setRowCount(len(self.rules))
                for i, rule in enumerate(self.rules):
                    self.table.setItem(i, 0, QTableWidgetItem(rule.get("label", "")))
                    kw_str = ", ".join(rule.get("keywords", []))
                    self.table.setItem(i, 1, QTableWidgetItem(kw_str))
            except Exception as e:
                QMessageBox.warning(self, "讀取失敗", f"無法載入規則: {e}")

    def save_rules(self):
        new_rules = []
        for i in range(self.table.rowCount()):
            label_item = self.table.item(i, 0)
            kw_item = self.table.item(i, 1)
            
            if not label_item or not kw_item or not label_item.text().strip():
                continue
                
            label = label_item.text().strip()
            kw_raw = kw_item.text()
            keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
            
            new_rules.append({
                "label": label,
                "keywords": keywords
            })
            
        try:
            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump(new_rules, f, ensure_ascii=False, indent=2)
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "儲存失敗", f"備份失敗: {e}")


class ConfigPanel(QGroupBox):
    def __init__(self):
        super().__init__("系統參數與輸出配置")
        layout = QVBoxLayout(self)
        
        # Output Dir
        row1 = QHBoxLayout()
        self.out_lbl = QLabel("輸出目錄: (留空則為原目錄)")
        self.out_input = QLabel("")
        self.out_input.setStyleSheet("border: 1px solid gray; padding: 2px;")
        self.out_btn = QPushButton("選擇目錄")
        row1.addWidget(self.out_lbl)
        row1.addWidget(self.out_input, 1)
        row1.addWidget(self.out_btn)
        layout.addLayout(row1)
        
        # Language Switcher
        row2 = QHBoxLayout()
        self.lang_lbl = QLabel("OCR 語系模型:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            "ch (簡體中文+英)", "chinese_cht (繁體中文+英)", 
            "japan (日+英)", "korean (韓+英)", "latin (拉丁語系)", "en (純正英文)"
        ])
        row2.addWidget(self.lang_lbl)
        row2.addWidget(self.lang_combo, 1)
        layout.addLayout(row2)
        
        # NLP and UI Features
        row3 = QHBoxLayout()
        self.nlp_chk = QCheckBox("啟用文件自動歸檔分類 (Auto-Classification)")
        self.rule_btn = QPushButton("管理規則引擎")
        row3.addWidget(self.nlp_chk)
        row3.addWidget(self.rule_btn)
        layout.addLayout(row3)
        
        # Hot Folder Watchdog
        row4 = QHBoxLayout()
        self.monitor_chk = QCheckBox("啟用背景熱區監控")
        self.hf_input = QLabel("")
        self.hf_input.setStyleSheet("border: 1px solid gray; padding: 2px;")
        self.hf_btn = QPushButton("設定熱區")
        row4.addWidget(self.monitor_chk)
        row4.addWidget(self.hf_input, 1)
        row4.addWidget(self.hf_btn)
        layout.addLayout(row4)
        

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PaddleOCR-Engine: Offline Searchable PDF Factory")
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)
        
        # Persisted Settings
        self.settings = QSettings("Antigravity", "PaddleOCREngine")
        
        # Threads
        self.worker = None
        self.monitor = None
        
        # UI Setup
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        self.config_panel = ConfigPanel()
        main_layout.addWidget(self.config_panel)
        
        # Connect config signals
        self.config_panel.out_btn.clicked.connect(self.select_output_dir)
        self.config_panel.hf_btn.clicked.connect(self.select_hot_folder)
        self.config_panel.rule_btn.clicked.connect(self.open_rule_manager)
        self.config_panel.monitor_chk.toggled.connect(self.toggle_monitor)
        
        # Queue List
        self.queue_label = QLabel("待處理檔案佇列 (拖曳檔案至此):")
        main_layout.addWidget(self.queue_label)
        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(QListWidget.ExtendedSelection)
        main_layout.addWidget(self.queue_list)
        
        btn_layout = QHBoxLayout()
        self.add_file_btn = QPushButton("新增檔案")
        self.add_file_btn.clicked.connect(self.add_files)
        self.remove_btn = QPushButton("移除選擇")
        self.remove_btn.clicked.connect(self.remove_selected)
        self.start_btn = QPushButton("開始轉換 (Start)")
        self.start_btn.setStyleSheet("background-color: darkgreen; color: white;")
        self.start_btn.clicked.connect(self.start_processing)
        self.abort_btn = QPushButton("強制終止 (Abort)")
        self.abort_btn.setStyleSheet("background-color: darkred; color: white;")
        self.abort_btn.clicked.connect(self.abort_processing)
        self.abort_btn.setEnabled(False)
        
        btn_layout.addWidget(self.add_file_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.abort_btn)
        main_layout.addLayout(btn_layout)
        
        # Progress & Logging
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: black; color: lightgreen; font-family: Consolas;")
        main_layout.addWidget(self.log_console)
        
        self.load_settings()

    def add_log(self, msg: str):
        self.log_console.append(msg)
        
    # --- Drag and Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        from monitor import ALLOWED_EXTENSIONS
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in ALLOWED_EXTENSIONS:
                self.queue_list.addItem(str(path))
                
    # --- Configs & Persistence ---
    def load_settings(self):
        self.config_panel.out_input.setText(self.settings.value("output_dir", ""))
        self.config_panel.hf_input.setText(self.settings.value("hot_folder", ""))
        self.config_panel.nlp_chk.setChecked(self.settings.value("enable_nlp", "false") == "true")
        
        lang_idx = int(self.settings.value("lang_index", 0))
        if lang_idx < self.config_panel.lang_combo.count():
            self.config_panel.lang_combo.setCurrentIndex(lang_idx)
            
        monitor_on = self.settings.value("monitor_enabled", "false") == "true"
        self.config_panel.monitor_chk.setChecked(monitor_on)
        # Checkbox toggle will automatically trigger thread start if True
        if monitor_on:
            self.toggle_monitor(True)
            
    def closeEvent(self, event):
        self.settings.setValue("output_dir", self.config_panel.out_input.text())
        self.settings.setValue("hot_folder", self.config_panel.hf_input.text())
        self.settings.setValue("enable_nlp", self.config_panel.nlp_chk.isChecked())
        self.settings.setValue("lang_index", self.config_panel.lang_combo.currentIndex())
        self.settings.setValue("monitor_enabled", self.config_panel.monitor_chk.isChecked())
        event.accept()

    def select_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.config_panel.out_input.setText(d)
            
    def select_hot_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Hot Folder")
        if d:
            self.config_panel.hf_input.setText(d)
            # Restart monitor if it relies on this path
            if self.config_panel.monitor_chk.isChecked():
                self.toggle_monitor(False)
                self.toggle_monitor(True)

    def add_files(self):
        from monitor import ALLOWED_EXTENSIONS
        filter_str = "Images/PDFs (" + " ".join([f"*{ext}" for ext in ALLOWED_EXTENSIONS]) + ")"
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", filter_str)
        for f in files:
            self.queue_list.addItem(f)
            
    def remove_selected(self):
        for item in self.queue_list.selectedItems():
            self.queue_list.takeItem(self.queue_list.row(item))
            
    def open_rule_manager(self):
        # We store rules inside the models folder
        rules_path = get_resource_path("models/rules.json")
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        dlg = RuleManagerDialog(rules_path, self)
        dlg.exec()

    # --- Hot Folder Monitor Thread ---
    def toggle_monitor(self, checked):
        hf_path = self.config_panel.hf_input.text()
        if checked and hf_path:
            if not self.monitor:
                self.add_log(f"Starting Background Daemon on: {hf_path}")
                self.monitor = DirectoryObserverThread(hf_path)
                self.monitor.file_detected.connect(self.on_hot_folder_detected)
                self.monitor.start()
        else:
            if self.monitor:
                self.add_log("Stopping Background Daemon...")
                self.monitor.stop()
                self.monitor = None

    def on_hot_folder_detected(self, filepath: str, source: str):
        # If UI is idle, auto-start. If already running, append to queue.
        # Ensure we only add files that aren't already in the list
        items = [self.queue_list.item(i).text() for i in range(self.queue_list.count())]
        target_str = f"{source} {filepath}"
        
        # Don't add duplicate auto items
        for existing in items:
            if existing.endswith(filepath):
                return
                
        self.queue_list.addItem(target_str)
        
        # Auto-trigger if we aren't already running
        if self.worker is None or not self.worker.isRunning():
            self.add_log(f"Auto-dispatching {filepath} from Daemon...")
            self.start_processing()

    # --- OCR Worker Thread ---
    def start_processing(self):
        if self.queue_list.count() == 0:
            return
            
        items = []
        for i in range(self.queue_list.count()):
            text = self.queue_list.item(i).text()
            if text.startswith("[Auto] "):
                text = text.replace("[Auto] ", "")
            items.append(text)
            
        lang_str = self.config_panel.lang_combo.currentText().split(" ")[0]
        
        # Resolution of absolute static paths for PyInstaller MEIPASS
        det_path = get_resource_path("models/det.onnx")
        rec_dir = get_resource_path("models/rec")
        dict_dir = get_resource_path("models/dict")
        nlp_dir = get_resource_path("models/nlp")
        rules_path = get_resource_path("models/rules.json")
        
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        
        # Clear UI Queue list (worker takes custody)
        self.queue_list.clear()

        self.worker = OCRWorker(
            queue=items,
            output_dir=self.config_panel.out_input.text(),
            lang=lang_str,
            enable_classification=self.config_panel.nlp_chk.isChecked(),
            hot_folder=self.config_panel.hf_input.text(),
            det_model=det_path,
            rec_models_dir=rec_dir,
            dict_models_dir=dict_dir,
            nlp_model_dir=nlp_dir,
            rules_path=rules_path,
            parent=self
        )
        
        self.worker.log_emitted.connect(self.add_log)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.task_finished.connect(self.on_task_finished)
        self.worker.start()

    def abort_processing(self):
        if self.worker:
            self.worker.stop()
            self.abort_btn.setEnabled(False)
            self.add_log("Sending abort signal to native solver...")

    def update_progress(self, current, total):
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)

    def on_task_finished(self, success):
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        
        # Safe disposal
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
