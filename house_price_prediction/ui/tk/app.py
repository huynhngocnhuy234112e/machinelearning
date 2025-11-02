import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from house_price_prediction.core.model_utils import load_model_zip


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_ZIP = str(BASE_DIR / "models" / "house_price_model.zip")


class HousePriceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("House Price Predictor — Tkinter")
        self.geometry("640x360")
        self.model = None
        self.feature_names = []
        self._build_ui()
        self._load_model()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True)

        self.inputs = {}
        fields = [
            'Avg Area Income',
            'Avg Area House Age',
            'Avg Area Number of Rooms',
            'Avg Area Number of Bedrooms',
            'Area Population',
        ]
        for i, f in enumerate(fields):
            ttk.Label(frm, text=f).grid(row=i, column=0, sticky=tk.W, **pad)
            var = tk.StringVar()
            ent = ttk.Entry(frm, textvariable=var, width=30)
            ent.grid(row=i, column=1, **pad)
            self.inputs[f] = var

        ttk.Button(frm, text="Predict", command=self.predict).grid(row=len(fields), column=0, **pad)
        ttk.Button(frm, text="Reload Model", command=self._load_model).grid(row=len(fields), column=1, **pad)

        self.result_var = tk.StringVar()
        ttk.Label(frm, textvariable=self.result_var, font=("Arial", 12, "bold")).grid(row=len(fields)+1, column=0, columnspan=2, **pad)

    def _load_model(self):
        try:
            self.model, self.feature_names = load_model_zip(MODEL_ZIP)
            self.result_var.set(f"Loaded model: {os.path.basename(MODEL_ZIP)}")
        except Exception as e:
            self.model, self.feature_names = None, []
            self.result_var.set("Model not found. Train first.")

    def predict(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return
        try:
            x = [[
                float(self.inputs['Avg Area Income'].get()),
                float(self.inputs['Avg Area House Age'].get()),
                float(self.inputs['Avg Area Number of Rooms'].get()),
                float(self.inputs['Avg Area Number of Bedrooms'].get()),
                float(self.inputs['Area Population'].get()),
            ]]
            pred = float(self.model.predict(x)[0])
            self.result_var.set(f"Predicted Price: {pred:,.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")


if __name__ == "__main__":
    app = HousePriceApp()
    app.mainloop()