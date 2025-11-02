from flask import Flask, render_template_string, request, send_file
import os
from pathlib import Path
from house_price_prediction.core.model_utils import load_model_zip

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>House Price Predictor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{ --primary:#2563eb; --bg:#f7f8fb; --border:#e5e7eb; --text:#0f172a; }
    *{ box-sizing:border-box }
    body{ margin:0; font-family:Arial, sans-serif; background:var(--bg); color:var(--text) }
    .header{ background:linear-gradient(90deg,#4F46E5,#3B82F6); color:#fff; padding:16px 24px; box-shadow:0 6px 16px rgba(0,0,0,.08) }
    .container{ max-width:960px; margin:0 auto; padding:24px }
    .card{ background:#fff; border:1px solid var(--border); border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.06); padding:16px }
    .grid{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:12px }
    label{ font-size:13px; color:#334155 }
    input{ width:100%; padding:10px; border:1px solid var(--border); border-radius:8px }
    .actions{ display:flex; gap:12px; margin-top:12px }
    .btn{ background:var(--primary); color:#fff; border:none; padding:10px 14px; border-radius:8px; font-weight:600; cursor:pointer }
    .btn.secondary{ background:#0ea5e9 }
    .result{ margin-top:16px; font-size:16px; font-weight:700 }
  </style>
  </head>
<body>
  <div class="header"><h1>House Price Predictor</h1></div>
  <div class="container">
    <div class="card">
      <form method="post">
        <div class="grid">
          <div><label>Avg Area Income</label><input name="Avg Area Income" value="{{values['Avg Area Income']}}" type="number" step="0.01"></div>
          <div><label>Avg Area House Age</label><input name="Avg Area House Age" value="{{values['Avg Area House Age']}}" type="number" step="0.01"></div>
          <div><label>Avg Area Number of Rooms</label><input name="Avg Area Number of Rooms" value="{{values['Avg Area Number of Rooms']}}" type="number" step="0.01"></div>
          <div><label>Avg Area Number of Bedrooms</label><input name="Avg Area Number of Bedrooms" value="{{values['Avg Area Number of Bedrooms']}}" type="number" step="0.01"></div>
          <div><label>Area Population</label><input name="Area Population" value="{{values['Area Population']}}" type="number" step="0.01"></div>
        </div>
        <div class="actions">
          <button class="btn" type="submit">Predict</button>
          <a class="btn secondary" href="/download-model">Download Trained Model</a>
        </div>
      </form>
      {% if prediction is not none %}
      <div class="result">Predicted Price: {{ prediction | round(2) }}</div>
      {% endif %}
    </div>
  </div>
  </body>
</html>
"""


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ZIP = str(BASE_DIR / "models" / "house_price_model.zip")


def create_app(model_zip: str | None = None):
    app = Flask(__name__)
    model, feature_names = None, []

    def _load():
        nonlocal model, feature_names
        mz = model_zip or DEFAULT_MODEL_ZIP
        if os.path.exists(mz):
            model, feature_names = load_model_zip(mz)
        else:
            model, feature_names = None, []

    _load()

    @app.route('/', methods=['GET', 'POST'])
    def index():
        values = {f: request.form.get(f, "") for f in ['Avg Area Income','Avg Area House Age','Avg Area Number of Rooms','Avg Area Number of Bedrooms','Area Population']}
        pred = None
        if request.method == 'POST' and model is not None:
            try:
                x = [[float(values['Avg Area Income']),
                      float(values['Avg Area House Age']),
                      float(values['Avg Area Number of Rooms']),
                      float(values['Avg Area Number of Bedrooms']),
                      float(values['Area Population'])]]
                pred = float(model.predict(x)[0])
            except Exception:
                pred = None
        return render_template_string(TEMPLATE, values=values, prediction=pred)

    @app.route('/download-model')
    def download_model():
        mz = model_zip or DEFAULT_MODEL_ZIP
        if os.path.exists(mz):
            return send_file(mz, as_attachment=True)
        return "Model file not found. Train the model first.", 404

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', port=5001, debug=True)