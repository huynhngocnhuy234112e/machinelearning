from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.theme import AppTheme

class KPICard(QGroupBox):
    def __init__(self, title, value, color=AppTheme.PRIMARY):
        super().__init__(title)
        self.color = color
        self.value_label = None
        self.setup_ui(value)
        
    def setup_ui(self, value):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        self.value_label = QLabel(value)
        self.value_label.setFont(AppTheme.get_font(18, bold=True))
        self.value_label.setStyleSheet(f"color: {self.color};")
        layout.addWidget(self.value_label)
        
        # Subtitle
        subtitle = QLabel("KPI Metric")
        subtitle.setFont(AppTheme.get_font(10))
        subtitle.setObjectName("lblSecondary")
        layout.addWidget(subtitle)
        
        # Set border color
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {self.color};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {AppTheme.SURFACE};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: {AppTheme.TEXT_SECONDARY};
            }}
        """)
    
    def update_value(self, new_value: str):
        """Update the KPI value"""
        if self.value_label:
            self.value_label.setText(new_value)
