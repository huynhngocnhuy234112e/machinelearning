# Credit Risk Management System

A professional PyQt6 application for credit risk assessment, featuring user billing management and admin risk scoring/analysis.

## Features

### User Portal
- **Login/Register**: Secure authentication
- **Billing Dashboard**: View invoices, amounts due, overdue amounts
- **Payment Processing**: Mark invoices as paid, process payments
- **Status Filtering**: Filter invoices by status (All, Due, Overdue, Paid)

### Admin Portal
- **Data Input & Scoring**
  - Upload CSV datasets
  - Automatic schema validation
  - Risk probability scoring with demo model
  - Live threshold adjustment with slider
  - Real-time KPI updates (Confusion Matrix, Expected Cost)
  - Find optimal threshold minimizing cost
  - Export predictions to CSV

- **Dashboard & Analysis**
  - KPI summary (Average PD, Precision, Recall, Expected Cost)
  - Probability distribution visualization
  - Gains curve analysis
  - Calibration plots
  - Confusion matrix heatmap
  - Segment analysis by customer groups
  - Top-K high-risk customers with suggestions
  - Rule-based risk mitigation recommendations
  - Export flagged customers with suggestions

- **Settings & Model Management**
  - Model version tracking
  - Threshold configuration (High/Medium risk)
  - Cost matrix setup (FN/FP costs)
  - Model upload (.pkl, .json)
  - Configuration management

## Architecture

\`\`\`
credit-risk-dashboard/
├── main.py                          # Entry point
├── ui/
│   ├── theme.py                     # Styling & color scheme
│   ├── main_window.py               # Main application window
│   ├── pages/
│   │   ├── login_page.py            # Authentication
│   │   ├── user_portal_page.py      # User billing interface
│   │   ├── admin_input_page.py      # CSV upload & scoring
│   │   ├── admin_dashboard_page.py  # Analysis & visualization
│   │   └── admin_settings_page.py   # Configuration
│   └── components/
│       ├── kpi_card.py              # KPI metric cards
│       └── chart_placeholder.py     # Chart placeholders
└── utils/
    ├── models.py                    # Data models (UserState, AppState)
    └── metrics.py                   # Metrics & calculations
\`\`\`

## Running the Application

\`\`\`bash
pip install PyQt6 pandas numpy matplotlib
python main.py
\`\`\`

## Demo Credentials

- **User**: demo@mlba.vn / 123
- **Admin**: admin / admin

## Key Metrics

- **Confusion Matrix**: TP, FP, FN, TN at given threshold
- **Precision**: TP / (TP + FP) - Hit rate of declines
- **Recall**: TP / (TP + FN) - Capture rate of defaults
- **F1 Score**: Harmonic mean of precision and recall
- **Expected Cost**: FN × C_FN + FP × C_FP
- **Gains Curve**: Cumulative capture of positives vs population

## Rule Engine

Automatic suggestions based on customer attributes:
- Payment history (PAY_0, PAY_2, etc.)
- Credit utilization ratio
- Age and credit limit
- Predicted default probability

## Future Enhancements

- Matplotlib chart integration for visualizations
- SHAP feature importance explanations
- Database backend (MySQL/PostgreSQL)
- REST API integration
- Real ML model loading (.pkl files)
- Advanced filtering and search
- Batch processing capabilities
