import os
from pathlib import Path
import sys
from PyQt6 import QtWidgets
from house_price_prediction.core.model_utils import load_model_zip


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_ZIP = str(BASE_DIR / "models" / "house_price_model.zip")


class HousePriceWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("House Price Predictor — PyQt6")
        self.model = None
        self.feature_names = []
        self._build_ui()
        self._load_model()

    def _build_ui(self):
        layout = QtWidgets.QFormLayout(self)
        self.inputs = {}
        fields = [
            'Avg Area Income',
            'Avg Area House Age',
            'Avg Area Number of Rooms',
            'Avg Area Number of Bedrooms',
            'Area Population',
        ]
        for f in fields:
            le = QtWidgets.QLineEdit(self)
            self.inputs[f] = le
            layout.addRow(f, le)

        btn_predict = QtWidgets.QPushButton("Predict", self)
        btn_predict.clicked.connect(self.predict)
        btn_reload = QtWidgets.QPushButton("Reload Model", self)
        btn_reload.clicked.connect(self._load_model)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(btn_predict)
        hl.addWidget(btn_reload)
        layout.addRow(hl)

        self.result_label = QtWidgets.QLabel("", self)
        layout.addRow(self.result_label)

    def _load_model(self):
        try:
            self.model, self.feature_names = load_model_zip(MODEL_ZIP)
            self.result_label.setText(f"Loaded model: {os.path.basename(MODEL_ZIP)}")
        except Exception:
            self.model, self.feature_names = None, []
            self.result_label.setText("Model not found. Train first.")

    def predict(self):
        if not self.model:
            QtWidgets.QMessageBox.critical(self, "Error", "Model not loaded.")
            return
        try:
            x = [[
                float(self.inputs['Avg Area Income'].text()),
                float(self.inputs['Avg Area House Age'].text()),
                float(self.inputs['Avg Area Number of Rooms'].text()),
                float(self.inputs['Avg Area Number of Bedrooms'].text()),
                float(self.inputs['Area Population'].text()),
            ]]
            pred = float(self.model.predict(x)[0])
            self.result_label.setText(f"Predicted Price: {pred:,.2f}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid input: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = HousePriceWindow()
    w.resize(480, 320)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()