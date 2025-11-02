from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.theme import AppTheme

class ChartPlaceholder(QWidget):
    def __init__(self, title):
        super().__init__()
        self.setup_ui(title)
        
    def setup_ui(self, title):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Placeholder
        placeholder = QLabel(f"📊 {title}\n\n(Chart will be rendered here)")
        placeholder.setFont(AppTheme.get_font(12))
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"""
            background-color: {AppTheme.SURFACE_VARIANT};
            border: 2px dashed {AppTheme.BORDER};
            border-radius: 6px;
            padding: 40px;
            color: {AppTheme.TEXT_SECONDARY};
        """)
        placeholder.setMinimumHeight(250)
        
        layout.addWidget(placeholder)
