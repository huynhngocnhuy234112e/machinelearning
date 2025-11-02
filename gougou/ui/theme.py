from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

class AppTheme:
    # Color Palette
    PRIMARY = "#2563EB"          # Blue
    PRIMARY_DARK = "#1E40AF"     # Dark Blue
    SECONDARY = "#10B981"        # Green (success)
    DANGER = "#EF4444"           # Red (danger)
    WARNING = "#F59E0B"          # Amber (warning)
    
    BACKGROUND = "#F9FAFB"       # Light gray
    SURFACE = "#FFFFFF"          # White
    SURFACE_VARIANT = "#F3F4F6"  # Light gray variant
    
    TEXT_PRIMARY = "#111827"     # Dark gray
    TEXT_SECONDARY = "#6B7280"   # Medium gray
    TEXT_TERTIARY = "#9CA3AF"    # Light gray
    
    BORDER = "#E5E7EB"           # Border gray
    
    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_SMALL = 10
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_LARGE = 12
    FONT_SIZE_TITLE = 14
    FONT_SIZE_HEADING = 16
    
    @staticmethod
    def get_stylesheet():
        return f"""
        QMainWindow {{
            background-color: {AppTheme.BACKGROUND};
        }}
        
        QWidget {{
            background-color: {AppTheme.BACKGROUND};
            color: {AppTheme.TEXT_PRIMARY};
            font-family: {AppTheme.FONT_FAMILY};
            font-size: {AppTheme.FONT_SIZE_NORMAL}pt;
        }}
        
        QGroupBox {{
            background-color: {AppTheme.SURFACE};
            border: 1px solid {AppTheme.BORDER};
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
            color: {AppTheme.TEXT_PRIMARY};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }}
        
        QPushButton {{
            background-color: {AppTheme.PRIMARY};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
            font-size: {AppTheme.FONT_SIZE_NORMAL}pt;
        }}
        
        QPushButton:hover {{
            background-color: {AppTheme.PRIMARY_DARK};
        }}
        
        QPushButton:pressed {{
            background-color: #1E3A8A;
        }}
        
        QPushButton#btnSecondary {{
            background-color: {AppTheme.SURFACE_VARIANT};
            color: {AppTheme.TEXT_PRIMARY};
            border: 1px solid {AppTheme.BORDER};
        }}
        
        QPushButton#btnSecondary:hover {{
            background-color: {AppTheme.BORDER};
        }}
        
        QPushButton#btnDanger {{
            background-color: {AppTheme.DANGER};
        }}
        
        QPushButton#btnDanger:hover {{
            background-color: #DC2626;
        }}
        
        QPushButton#btnSuccess {{
            background-color: {AppTheme.SECONDARY};
        }}
        
        QPushButton#btnSuccess:hover {{
            background-color: #059669;
        }}
        
        QLineEdit, QTextEdit {{
            background-color: {AppTheme.SURFACE};
            border: 1px solid {AppTheme.BORDER};
            border-radius: 4px;
            padding: 6px;
            color: {AppTheme.TEXT_PRIMARY};
            selection-background-color: {AppTheme.PRIMARY};
        }}
        
        QLineEdit:focus, QTextEdit:focus {{
            border: 2px solid {AppTheme.PRIMARY};
        }}
        
        QComboBox {{
            background-color: {AppTheme.SURFACE};
            border: 1px solid {AppTheme.BORDER};
            border-radius: 4px;
            padding: 6px;
            color: {AppTheme.TEXT_PRIMARY};
        }}
        
        QComboBox:focus {{
            border: 2px solid {AppTheme.PRIMARY};
        }}
        
        QComboBox::drop-down {{
            border: none;
        }}
        
        QSlider::groove:horizontal {{
            border: 1px solid {AppTheme.BORDER};
            height: 6px;
            background: {AppTheme.SURFACE_VARIANT};
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background: {AppTheme.PRIMARY};
            border: 1px solid {AppTheme.PRIMARY_DARK};
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: {AppTheme.PRIMARY_DARK};
        }}
        
        QTableView {{
            background-color: {AppTheme.SURFACE};
            alternate-background-color: {AppTheme.SURFACE_VARIANT};
            gridline-color: {AppTheme.BORDER};
            border: 1px solid {AppTheme.BORDER};
            border-radius: 4px;
        }}
        
        QTableView::item {{
            padding: 4px;
        }}
        
        QTableView::item:selected {{
            background-color: {AppTheme.PRIMARY};
            color: white;
        }}
        
        QHeaderView::section {{
            background-color: {AppTheme.SURFACE_VARIANT};
            color: {AppTheme.TEXT_PRIMARY};
            padding: 6px;
            border: none;
            border-right: 1px solid {AppTheme.BORDER};
            border-bottom: 1px solid {AppTheme.BORDER};
            font-weight: bold;
        }}
        
        QLabel {{
            color: {AppTheme.TEXT_PRIMARY};
        }}
        
        QLabel#lblSecondary {{
            color: {AppTheme.TEXT_SECONDARY};
        }}
        
        QLabel#lblTertiary {{
            color: {AppTheme.TEXT_TERTIARY};
        }}
        
        QLabel#lblTitle {{
            font-size: {AppTheme.FONT_SIZE_HEADING}pt;
            font-weight: bold;
            color: {AppTheme.TEXT_PRIMARY};
        }}
        
        QLabel#lblSubtitle {{
            font-size: {AppTheme.FONT_SIZE_LARGE}pt;
            font-weight: bold;
            color: {AppTheme.TEXT_SECONDARY};
        }}
        
        QLabel#lblKPI {{
            font-size: {AppTheme.FONT_SIZE_TITLE}pt;
            font-weight: bold;
            color: {AppTheme.PRIMARY};
        }}
        
        QLabel#lblSuccess {{
            color: {AppTheme.SECONDARY};
        }}
        
        QLabel#lblDanger {{
            color: {AppTheme.DANGER};
        }}
        
        QLabel#lblWarning {{
            color: {AppTheme.WARNING};
        }}
        
        QScrollArea {{
            background-color: {AppTheme.BACKGROUND};
            border: none;
        }}
        
        QScrollBar:vertical {{
            background-color: {AppTheme.BACKGROUND};
            width: 12px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {AppTheme.BORDER};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {AppTheme.TEXT_TERTIARY};
        }}
        
        QTabBar::tab {{
            background-color: {AppTheme.SURFACE_VARIANT};
            color: {AppTheme.TEXT_SECONDARY};
            padding: 8px 16px;
            border: none;
            border-bottom: 2px solid transparent;
        }}
        
        QTabBar::tab:selected {{
            background-color: {AppTheme.SURFACE};
            color: {AppTheme.PRIMARY};
            border-bottom: 2px solid {AppTheme.PRIMARY};
        }}
        
        QTabBar::tab:hover {{
            background-color: {AppTheme.SURFACE};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {AppTheme.BORDER};
        }}
        """
    
    @staticmethod
    def get_font(size=None, bold=False):
        font = QFont(AppTheme.FONT_FAMILY)
        if size:
            font.setPointSize(size)
        if bold:
            font.setBold(True)
        return font
