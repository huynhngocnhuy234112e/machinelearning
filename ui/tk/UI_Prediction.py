import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from house_price_prediction.ui.tk.DatasetViewer import DatasetViewer
from house_price_prediction.ui.tk.Predictor import save_model, load_model_any
from house_price_prediction.core.data_loader import ensure_columns


FEATURES = [
    'Avg Area Income',
    'Avg Area House Age',
    'Avg Area Number of Rooms',
    'Avg Area Number of Bedrooms',
    'Area Population',
]

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ZIP = str(MODEL_DIR / 'house_price_model.zip')


class UIPrediction(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('House Price Prediction — Tkinter')
        # Tăng kích thước để khớp layout như ảnh minh họa
        self.geometry('1200x700')
        self.df = None
        self.model = None
        self._build_ui()

    def _build_ui(self):
        # Header (màu nền vàng như ảnh)
        header = tk.Label(self, text='House Pricing Prediction', bg='#ffe45c', fg='#000', font=('Arial', 18, 'bold'))
        header.pack(fill=tk.X, padx=0, pady=0)

        # Top bar: Select Dataset ở trái, 1–2 ở phải
        top = tk.Frame(self, bg='#fff3a1')
        top.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(top, text='Select Dataset:', bg='#fff3a1').pack(side=tk.LEFT, padx=(0,4))
        self.dataset_path_var = tk.StringVar(value='(none)')
        tk.Label(top, textvariable=self.dataset_path_var, bg='#fff3a1').pack(side=tk.LEFT, padx=(0,10))

        ttk.Button(top, text='2. View Dataset', command=self.view_dataset).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text='1. Pick Dataset', command=self.pick_dataset).pack(side=tk.RIGHT, padx=4)

        # Main area: left dataset viewer, right evaluation panel
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Đặt panel đánh giá bên phải, cố định chiều rộng để luôn hiển thị
        right = ttk.LabelFrame(main, text='Evaluation is finished')
        right.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(8,0))
        right.configure(width=380)
        right.pack_propagate(False)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Thanh giữa: Training Rate + nút 3–4 đặt ở trên bảng dữ liệu
        mid = ttk.Frame(left)
        mid.pack(fill=tk.X, padx=0, pady=(0,6))
        ttk.Label(mid, text='Training Rate:').pack(side=tk.LEFT)
        self.train_rate_var = tk.StringVar(value='80')  # 80%
        ttk.Entry(mid, textvariable=self.train_rate_var, width=6).pack(side=tk.LEFT, padx=(2,10))
        ttk.Button(mid, text='3. Train Model', command=self.train_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(mid, text='4. Evaluate Model', command=self.evaluate_model).pack(side=tk.LEFT, padx=4)

        self.viewer = DatasetViewer(left)
        self.viewer.pack(fill=tk.BOTH, expand=True)

        # Coefficient viewer + metrics trong panel phải
        ttk.Label(right, text='Coefficient').pack(anchor='w', padx=6, pady=(6,0))
        self.coeff_text = tk.Text(right, height=12, width=40)
        self.coeff_text.configure(font=('Consolas', 10))
        self.coeff_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.coeff_text.configure(state='disabled')

        self.lbl_intercept = ttk.Label(right, text='Intercept: -')
        self.lbl_intercept.pack(anchor='w', padx=6, pady=(4,0))
        self.lbl_mae = ttk.Label(right, text='Mean Absolute Error (MAE): -')
        self.lbl_mae.pack(anchor='w', padx=6)
        self.lbl_mse = ttk.Label(right, text='Mean Square Error (MSE): -')
        self.lbl_mse.pack(anchor='w', padx=6)
        self.lbl_rmse = ttk.Label(right, text='Root Mean Square Error (RMSE): -')
        self.lbl_rmse.pack(anchor='w', padx=6)
        ttk.Button(right, text='5. Save Model', command=self.save_model_ui).pack(padx=6, pady=6)

        # Bottom area: Load Model + prediction inputs
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(bottom, text='6. Load Model').grid(row=0, column=0, sticky='w', padx=4, pady=4)
        self.model_choice = tk.StringVar(value=os.path.basename(MODEL_ZIP))
        self.model_choices = []
        self.model_menu = ttk.OptionMenu(bottom, self.model_choice, None)
        self.model_menu.grid(row=0, column=1, sticky='w', padx=4, pady=4)
        ttk.Button(bottom, text='Load Model', command=self.load_model_ui).grid(row=0, column=2, sticky='w', padx=4, pady=4)
        ttk.Button(bottom, text='Browse...', command=self.browse_model_file).grid(row=0, column=3, sticky='w', padx=4, pady=4)

        # Prediction inputs
        fields = FEATURES
        self.pred_inputs = {}
        for i, f in enumerate(fields, start=1):
            ttk.Label(bottom, text=f).grid(row=i, column=0, sticky='w', padx=4, pady=4)
            var = tk.StringVar()
            ttk.Entry(bottom, textvariable=var, width=20).grid(row=i, column=1, sticky='w', padx=4, pady=4)
            self.pred_inputs[f] = var
        ttk.Button(bottom, text='7. Prediction House Pricing', command=self.predict_inline).grid(row=len(fields)+1, column=0, sticky='w', padx=4, pady=6)
        self.pred_out_var = tk.StringVar(value='Prediction Price: -')
        ttk.Label(bottom, textvariable=self.pred_out_var).grid(row=len(fields)+1, column=1, sticky='w', padx=4, pady=6)

        # Status
        self.status = tk.StringVar(value='Ready')
        ttk.Label(self, textvariable=self.status).pack(fill=tk.X, padx=8, pady=6)

        # Initialize dropdown
        self._refresh_model_dropdown()

    # Actions
    def pick_dataset(self):
        path = filedialog.askopenfilename(title='Select dataset CSV', filetypes=[('CSV files', '*.csv')])
        if path:
            try:
                self.df = pd.read_csv(path)
                # Chuẩn hóa tên cột theo chuẩn không dấu chấm (Avg. -> Avg)
                ensure_columns(self.df)
                self.dataset_path_var.set(path)
                self.status.set(f'Loaded dataset: {os.path.basename(path)}')
            except Exception as e:
                messagebox.showerror('Error', str(e))

    def view_dataset(self):
        if self.df is None:
            messagebox.showinfo('Info', 'Pick dataset first.')
            return
        self.viewer.load_dataframe(self.df)

    def _get_X_y(self):
        X = self.df[FEATURES]
        y = self.df['Price']
        return X, y

    def train_model(self):
        if self.df is None:
            messagebox.showinfo('Info', 'Pick dataset first.')
            return
        X, y = self._get_X_y()
        try:
            rate_percent = int(self.train_rate_var.get())
            rate_percent = max(50, min(rate_percent, 95))  # clamp 50..95
        except Exception:
            rate_percent = 80
        rate = rate_percent / 100.0
        test_size = 1.0 - rate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        lin = LinearRegression()
        lin.fit(X_train, y_train)
        save_model(lin, FEATURES, MODEL_ZIP)
        self.model = lin
        messagebox.showinfo('Train', f'Trained LinearRegression (Training Rate={rate_percent}%). Saved: {MODEL_ZIP}')
        self._refresh_model_dropdown()

    def evaluate_model(self):
        if self.df is None:
            messagebox.showinfo('Info', 'Pick dataset first.')
            return
        # Ưu tiên mô hình đang có trong bộ nhớ; nếu chưa có thì nạp theo lựa chọn dropdown, cuối cùng mới rơi về mặc định
        if not self.model:
            chosen = (self.model_choice.get() or '').strip()
            candidate = MODEL_DIR / chosen if chosen else None
            load_path = None
            if candidate and candidate.exists():
                load_path = str(candidate)
            elif os.path.exists(MODEL_ZIP):
                load_path = MODEL_ZIP
            else:
                load_path = None
            if not load_path:
                self.status.set('Model not found. Train or load a model first.')
                return
            try:
                self.model, _ = load_model_any(load_path, default_feature_names=FEATURES)
                self.status.set(f'Loaded model for evaluation: {os.path.basename(load_path)}')
            except Exception as e:
                messagebox.showerror('Error', f'Không thể nạp mô hình: {e}')
                return
        X, y = self._get_X_y()
        # sử dụng cùng tỉ lệ test theo Training Rate
        try:
            rate_percent = int(self.train_rate_var.get())
            rate_percent = max(50, min(rate_percent, 95))
        except Exception:
            rate_percent = 80
        test_size = 1.0 - (rate_percent/100.0)
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        # Cập nhật panel đánh giá và hiển thị coefficients (dạng văn bản giống ảnh)
        lines = ["Coefficient:"]
        for fname, coef in zip(FEATURES, self.model.coef_):
            lines.append(f"- {fname:<30} {coef:>12.6f}")
        self.coeff_text.configure(state='normal')
        self.coeff_text.delete('1.0', tk.END)
        self.coeff_text.insert('1.0', "\n".join(lines))
        self.coeff_text.configure(state='disabled')
        self.lbl_intercept.config(text=f'Intercept: {self.model.intercept_:.6f}')
        self.lbl_mae.config(text=f'Mean Absolute Error (MAE): {mae:.6f}')
        self.lbl_mse.config(text=f'Mean Square Error (MSE): {mse:.6f}')
        self.lbl_rmse.config(text=f'Root Mean Square Error (RMSE): {rmse:.6f}')

        # Hiển thị bảng dự đoán bên trái theo đúng layout
        df2 = self.df.copy()
        df2['Prediction'] = self.model.predict(df2[FEATURES])
        df_show = df2[[*FEATURES, 'Price', 'Prediction']].rename(columns={'Price': 'Original Price', 'Prediction': 'Prediction Price'})
        self.viewer.load_dataframe(df_show)

    def predict_inline(self):
        try:
            # ưu tiên model đang nạp; nếu chưa có thì nạp mặc định
            if not self.model:
                self.model, _ = load_model_any(MODEL_ZIP, default_feature_names=FEATURES)
            x = [[float(self.pred_inputs[f].get()) for f in FEATURES]]
            pred = float(self.model.predict(x)[0])
            self.pred_out_var.set(f'Prediction Price: {pred:,.2f}')
        except Exception as e:
            messagebox.showerror('Error', f'Invalid input: {e}')

    def predict_by_dataset(self):
        if self.df is None:
            messagebox.showinfo('Info', 'Pick dataset first.')
            return
        try:
            self.model, _ = load_model_any(MODEL_ZIP, default_feature_names=FEATURES)
        except Exception:
            self.status.set('Model not found. Train first.')
            return
        df2 = self.df.copy()
        df2['Prediction'] = self.model.predict(df2[FEATURES])
        self.viewer.load_dataframe(df2[[*FEATURES, 'Price', 'Prediction']])

    # Save/Load model UI helpers
    def save_model_ui(self):
        if not self.model:
            # Nếu chưa train, thử nạp mặc định
            try:
                self.model, _ = load_model_any(MODEL_ZIP, default_feature_names=FEATURES)
            except Exception:
                messagebox.showinfo('Info', 'Train model before saving.')
                return
        if not messagebox.askyesno('Save Model', 'Lưu mô hình hiện tại với tên kèm thời gian?'):
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_zip = MODEL_DIR / f'house_price_model_{ts}.zip'
        save_model(self.model, FEATURES, str(out_zip))
        messagebox.showinfo('Saved', f'Lưu mô hình vào: {out_zip}')
        self._refresh_model_dropdown()

    def _refresh_model_dropdown(self):
        # Quét thư mục models để nạp danh sách zip
        zips = sorted([p.name for p in MODEL_DIR.glob('*.zip')])
        if not zips:
            zips = [os.path.basename(MODEL_ZIP)] if os.path.exists(MODEL_ZIP) else []
        self.model_choices = zips
        menu = self.model_menu['menu']
        menu.delete(0, 'end')
        for name in self.model_choices:
            menu.add_command(label=name, command=lambda n=name: self.model_choice.set(n))
        if zips:
            # đặt giá trị mặc định
            self.model_choice.set(zips[-1])

    def load_model_ui(self):
        name = (self.model_choice.get() or '').strip()
        candidate = MODEL_DIR / name if name else None
        if not name or not candidate.exists():
            # Fallback: open file dialog to let user pick any model file
            path = filedialog.askopenfilename(title='Select model file',
                                              filetypes=[('Model files', '*.zip *.pkl *.pickle *.bin *.model'), ('All files', '*.*')])
            if not path:
                messagebox.showinfo('Info', 'Không tìm thấy file mô hình.')
                return
            load_path = path
            loaded_name = os.path.basename(path)
        else:
            load_path = str(candidate)
            loaded_name = name
        try:
            self.model, _ = load_model_any(load_path, default_feature_names=FEATURES)
            self.status.set(f'Loaded model: {loaded_name}')
            # If the file is inside models/, reflect in dropdown
            if os.path.dirname(load_path) == str(MODEL_DIR):
                self.model_choice.set(loaded_name)
        except Exception as e:
            messagebox.showerror('Error', f'Không thể nạp mô hình: {e}')

    def browse_model_file(self):
        path = filedialog.askopenfilename(title='Select model file',
                                          filetypes=[('Model files', '*.zip *.pkl *.pickle *.bin *.model'), ('All files', '*.*')])
        if not path:
            return
        try:
            self.model, _ = load_model_any(path, default_feature_names=FEATURES)
            self.status.set(f'Loaded model: {os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('Error', f'Không thể nạp mô hình: {e}')


if __name__ == '__main__':
    app = UIPrediction()
    app.mainloop()