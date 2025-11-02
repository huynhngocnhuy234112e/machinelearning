import os
from pathlib import Path
import sys
from PyQt6 import QtWidgets
from PyQt6.uic import loadUi

from house_price_prediction.core.model_utils import load_model_zip


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_ZIP = str(BASE_DIR / "models" / "house_price_model.zip")


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        loadUi(os.path.abspath("house_price_prediction/ui/pyqt6/MainWindow.ui"), self)
        self.setWindowTitle("House Price Predictor — PyQt6")
        self.model = None
        self.feature_names = []
        self.btnPredict.clicked.connect(self.predict)
        self.btnReload.clicked.connect(self.load_model)
        self.load_model()

    def load_model(self):
        try:
            self.model, self.feature_names = load_model_zip(MODEL_ZIP)
            self.lblResult.setText(f"Loaded model: {os.path.basename(MODEL_ZIP)}")
        except Exception:
            self.model, self.feature_names = None, []
            self.lblResult.setText("Model not found. Train first.")

    def predict(self):
        if not self.model:
            QtWidgets.QMessageBox.critical(self, "Error", "Model not loaded.")
            return
        try:
            x = [[
                float(self.editIncome.text()),
                float(self.editAge.text()),
                float(self.editRooms.text()),
                float(self.editBeds.text()),
                float(self.editPop.text()),
            ]]
            pred = float(self.model.predict(x)[0])
            self.lblResult.setText(f"Prediction: {pred:,.2f}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid input: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(420, 300)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()