from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QStackedWidget, QPushButton, QLabel, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.theme import AppTheme
from ui.pages.login_page import LoginPage
from ui.pages.user_portal_page import UserPortalPage
from ui.pages.admin_input_page import AdminInputPage
from ui.pages.admin_dashboard_page import AdminDashboardPage
from ui.pages.admin_settings_page import AdminSettingsPage
from utils.models import AppState

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Credit Risk Management System")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(AppTheme.get_stylesheet())
        
        # Shared state
        self.app_state = AppState()
        self.current_user = None
        self.is_admin = False
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        
        self.login_page = LoginPage(self)
        self.user_portal_page = UserPortalPage(self)
        
        self.admin_tab_widget = QTabWidget()
        self.admin_input_page = AdminInputPage(self)
        self.admin_dashboard_page = AdminDashboardPage(self)
        self.admin_settings_page = AdminSettingsPage(self)
        
        self.admin_tab_widget.addTab(self.admin_input_page, "📊 Input & Scoring")
        self.admin_tab_widget.addTab(self.admin_dashboard_page, "📈 Dashboard & Analysis")
        self.admin_tab_widget.addTab(self.admin_settings_page, "⚙️ Settings")
        
        self.stacked_widget.addWidget(self.login_page)
        self.stacked_widget.addWidget(self.user_portal_page)
        self.stacked_widget.addWidget(self.admin_tab_widget)
        
        self.admin_input_page.results_ready.connect(self.on_scoring_complete)
        
        layout.addWidget(self.stacked_widget)
        
        # Show login page initially
        self.show_page("login")
        
    def create_header(self):
        header = QWidget()
        header.setStyleSheet(f"background-color: {AppTheme.PRIMARY};")
        header.setFixedHeight(60)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)
        
        # Logo/Title
        title = QLabel("Credit Risk Management System")
        title_font = AppTheme.get_font(14, bold=True)
        title.setFont(title_font)
        title.setStyleSheet("color: white;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # User info
        self.user_label = QLabel("Not logged in")
        self.user_label.setStyleSheet("color: white;")
        layout.addWidget(self.user_label)
        
        # Logout button
        self.logout_btn = QPushButton("Logout")
        self.logout_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid white;
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 0.3);
            }}
        """)
        self.logout_btn.clicked.connect(self.logout)
        self.logout_btn.hide()
        layout.addWidget(self.logout_btn)
        
        return header
    
    def show_page(self, page_name):
        pages = {
            "login": 0,
            "user_portal": 1,
            "admin": 2,
        }
        if page_name in pages:
            self.stacked_widget.setCurrentIndex(pages[page_name])
    
    def login(self, username, is_admin=False):
        self.current_user = username
        self.is_admin = is_admin
        self.user_label.setText(f"👤 {username} {'(Admin)' if is_admin else '(User)'}")
        self.logout_btn.show()
        
        if is_admin:
            self.show_page("admin")
        else:
            self.show_page("user_portal")
    
    def logout(self):
        self.current_user = None
        self.is_admin = False
        self.user_label.setText("Not logged in")
        self.logout_btn.hide()
        self.show_page("login")
    
    def on_scoring_complete(self):
        """Handle scoring completion - update dashboard state and render charts"""
        self.admin_dashboard_page.state = self.admin_input_page.state
        self.admin_dashboard_page.refresh_all(self.admin_input_page.state)
        self.admin_dashboard_page.render_charts()
        
        # Switch to dashboard tab
        self.admin_tab_widget.setCurrentIndex(1)
