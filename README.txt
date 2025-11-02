House Price Prediction — Linear Regression

Structure:
- core/: shared logic (data loading, training, model IO)
- ui/: user interfaces
  - web/: Flask web app
  - tk/: Tkinter desktop app
  - qt/: PyQt6 app with QtDesigner .ui
- models/: saved trained model (zip)
- train_console.py: CLI to train and evaluate

Features expected (USA Housing style):
- Avg Area Income
- Avg Area House Age
- Avg Area Number of Rooms
- Avg Area Number of Bedrooms
- Area Population
- Price (target)

Run training:
  python house_price_prediction/train_console.py --data "path/to/USA_Housing.csv" --out models/house_price_model.zip

Run Flask web:
  python house_price_prediction/ui/web/app.py

Run Tkinter app:
  python house_price_prediction/ui/tk/app.py

Run PyQt6 app:
  python house_price_prediction/ui/qt/main.py

Required packages:
  pip install pandas scikit-learn flask pyqt6