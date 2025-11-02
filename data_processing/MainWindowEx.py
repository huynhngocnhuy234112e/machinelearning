import requests
from urllib.parse import quote
from PyQt6.QtWidgets import QMainWindow
from data_processing.MainWindow import Ui_MainWindow

LANG_MAP = {
    "VIETNAMESE": "vi",
    "ENGLISH": "en",
}

class MainWindowEx(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.trans)

    def trans(self):
        text = self.lineEditText.text().strip()
        src_display = self.comboBoxSource.currentText()
        tgt_display = self.comboBoxTarget.currentText()

        if not text:
            self.labelResult.setText("Please enter text.")
            return

        source_lang = LANG_MAP.get(src_display, "auto")
        target_lang = LANG_MAP.get(tgt_display, "en")

        try:
            encoded = quote(text, safe="")  # URL-encode
            url = f"https://lingva.ml/api/v1/{source_lang}/{target_lang}/{encoded}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                self.labelResult.setText(f"HTTP {resp.status_code}")
                return
            translated = resp.json().get("translation", "")
            self.labelResult.setText(translated or "(no result)")
        except Exception as e:
            self.labelResult.setText(f"Error: {e}")
