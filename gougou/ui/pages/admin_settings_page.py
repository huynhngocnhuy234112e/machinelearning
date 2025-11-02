from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog,
    QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.theme import AppTheme

class AdminSettingsPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Settings & Model Management")
        title.setFont(AppTheme.get_font(16, bold=True))
        title.setObjectName("lblTitle")
        layout.addWidget(title)
        
        # Model version section
        model_group = QGroupBox("Current Model")
        model_layout = QVBoxLayout(model_group)
        
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Model Version:"))
        self.model_version = QLabel("v2.1.0")
        self.model_version.setObjectName("lblKPI")
        info_layout.addWidget(self.model_version)
        info_layout.addStretch()
        layout.addWidget(model_group)
        
        model_layout.addLayout(info_layout)
        
        info_layout2 = QHBoxLayout()
        info_layout2.addWidget(QLabel("Last Updated:"))
        self.model_date = QLabel("2025-10-15")
        self.model_date.setObjectName("lblSecondary")
        info_layout2.addWidget(self.model_date)
        info_layout2.addStretch()
        model_layout.addLayout(info_layout2)
        
        layout.addWidget(model_group)
        
        # Threshold settings
        threshold_group = QGroupBox("Threshold Configuration")
        threshold_layout = QVBoxLayout(threshold_group)
        
        # High risk threshold
        high_layout = QHBoxLayout()
        high_layout.addWidget(QLabel("High Risk Threshold:"))
        self.high_threshold = QDoubleSpinBox()
        self.high_threshold.setMinimum(0.0)
        self.high_threshold.setMaximum(1.0)
        self.high_threshold.setSingleStep(0.05)
        self.high_threshold.setValue(0.70)
        high_layout.addWidget(self.high_threshold)
        high_layout.addStretch()
        threshold_layout.addLayout(high_layout)
        
        # Medium risk threshold
        medium_layout = QHBoxLayout()
        medium_layout.addWidget(QLabel("Medium Risk Threshold:"))
        self.medium_threshold = QDoubleSpinBox()
        self.medium_threshold.setMinimum(0.0)
        self.medium_threshold.setMaximum(1.0)
        self.medium_threshold.setSingleStep(0.05)
        self.medium_threshold.setValue(0.40)
        medium_layout.addWidget(self.medium_threshold)
        medium_layout.addStretch()
        threshold_layout.addLayout(medium_layout)
        
        layout.addWidget(threshold_group)
        
        # Model upload section
        upload_group = QGroupBox("Upload New Model")
        upload_layout = QVBoxLayout(upload_group)
        
        # Model file
        model_file_layout = QHBoxLayout()
        model_file_layout.addWidget(QLabel("Model File (.pkl):"))
        self.model_file_label = QLabel("No file selected")
        self.model_file_label.setObjectName("lblSecondary")
        model_file_layout.addWidget(self.model_file_label)
        model_file_layout.addStretch()
        
        model_file_btn = QPushButton("Choose File")
        model_file_btn.setMaximumWidth(120)
        model_file_btn.clicked.connect(self.choose_model_file)
        model_file_layout.addWidget(model_file_btn)
        
        upload_layout.addLayout(model_file_layout)
        
        # Config file
        config_file_layout = QHBoxLayout()
        config_file_layout.addWidget(QLabel("Config File (.json):"))
        self.config_file_label = QLabel("No file selected")
        self.config_file_label.setObjectName("lblSecondary")
        config_file_layout.addWidget(self.config_file_label)
        config_file_layout.addStretch()
        
        config_file_btn = QPushButton("Choose File")
        config_file_btn.setMaximumWidth(120)
        config_file_btn.clicked.connect(self.choose_config_file)
        config_file_layout.addWidget(config_file_btn)
        
        upload_layout.addLayout(config_file_layout)
        
        layout.addWidget(upload_group)
        
        # Cost matrix section
        cost_group = QGroupBox("Cost Matrix (FN/FP)")
        cost_layout = QVBoxLayout(cost_group)
        
        fn_layout = QHBoxLayout()
        fn_layout.addWidget(QLabel("Cost of False Negative (missed default):"))
        self.fn_cost = QDoubleSpinBox()
        self.fn_cost.setMinimum(0.0)
        self.fn_cost.setMaximum(10000.0)
        self.fn_cost.setValue(1000.0)
        fn_layout.addWidget(self.fn_cost)
        fn_layout.addStretch()
        cost_layout.addLayout(fn_layout)
        
        fp_layout = QHBoxLayout()
        fp_layout.addWidget(QLabel("Cost of False Positive (rejected good customer):"))
        self.fp_cost = QDoubleSpinBox()
        self.fp_cost.setMinimum(0.0)
        self.fp_cost.setMaximum(10000.0)
        self.fp_cost.setValue(100.0)
        fp_layout.addWidget(self.fp_cost)
        fp_layout.addStretch()
        cost_layout.addLayout(fp_layout)
        
        layout.addWidget(cost_group)
        
        layout.addStretch()
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        action_layout.addWidget(save_btn)
        
        upload_btn = QPushButton("Upload Model")
        upload_btn.setObjectName("btnSuccess")
        upload_btn.clicked.connect(self.upload_model)
        action_layout.addWidget(upload_btn)
        
        back_btn = QPushButton("← Back to Dashboard")
        back_btn.setObjectName("btnSecondary")
        back_btn.clicked.connect(lambda: self.main_window.show_page("admin_dashboard"))
        action_layout.addWidget(back_btn)
        
        layout.addLayout(action_layout)
    
    def choose_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PKL Files (*.pkl)")
        if file_path:
            self.model_file_label.setText(f"✓ {file_path.split('/')[-1]}")
    
    def choose_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Config File", "", "JSON Files (*.json)")
        if file_path:
            self.config_file_label.setText(f"✓ {file_path.split('/')[-1]}")
    
    def save_settings(self):
        QMessageBox.information(self, "Success", "Settings saved successfully!")
    
    def upload_model(self):
        QMessageBox.information(self, "Success", "Model uploaded successfully!")
