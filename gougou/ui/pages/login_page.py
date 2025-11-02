from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.uic import loadUi
import os

from ui.theme import AppTheme

class LoginPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        ui_path = os.path.join(os.path.dirname(__file__), '..', 'forms', 'login_page.ui')
        if os.path.exists(ui_path):
            loadUi(ui_path, self)
        else:
            # Fallback to programmatic UI if .ui file not found
            self.setup_ui_programmatic()
        
        # Connect signals
        self.loginButton.clicked.connect(self.handle_login)
        self.adminLoginButton.clicked.connect(self.handle_admin_login)
        
    def setup_ui_programmatic(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 60, 40, 60)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Credit Risk Management System")
        title.setFont(AppTheme.get_font(24, bold=True))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Login to your account")
        subtitle.setFont(AppTheme.get_font(12))
        subtitle.setObjectName("lblSecondary")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(40)
        
        # Login form
        form_layout = QVBoxLayout()
        form_layout.setSpacing(16)
        
        # Username
        form_layout.addWidget(QLabel("Username"))
        self.usernameInput = QLineEdit()
        self.usernameInput.setPlaceholderText("Enter your username")
        self.usernameInput.setMinimumHeight(40)
        form_layout.addWidget(self.usernameInput)
        
        # Password
        form_layout.addWidget(QLabel("Password"))
        self.passwordInput = QLineEdit()
        self.passwordInput.setPlaceholderText("Enter your password")
        self.passwordInput.setEchoMode(QLineEdit.EchoMode.Password)
        self.passwordInput.setMinimumHeight(40)
        form_layout.addWidget(self.passwordInput)
        
        form_layout.addSpacing(10)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        self.loginButton = QPushButton("Login")
        self.loginButton.setMinimumHeight(40)
        button_layout.addWidget(self.loginButton)
        
        self.adminLoginButton = QPushButton("Login as Admin")
        self.adminLoginButton.setObjectName("btnSecondary")
        self.adminLoginButton.setMinimumHeight(40)
        button_layout.addWidget(self.adminLoginButton)
        
        form_layout.addLayout(button_layout)
        
        # Message
        self.messageLabel = QLabel("")
        self.messageLabel.setObjectName("lblSecondary")
        self.messageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form_layout.addWidget(self.messageLabel)
        
        # Center the form
        form_group = QGroupBox()
        form_group.setLayout(form_layout)
        form_group.setMinimumWidth(400)
        form_group.setMaximumWidth(500)
        
        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addWidget(form_group)
        center_layout.addStretch()
        
        layout.addLayout(center_layout)
        layout.addStretch()
        
    def handle_login(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text()
        
        if not username or not password:
            self.messageLabel.setText("❌ Please enter username and password")
            self.messageLabel.setObjectName("lblDanger")
            return
        
        # Demo: accept any non-empty credentials
        self.main_window.login(username, is_admin=False)
        self.clear_form()
    
    def handle_admin_login(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text()
        
        if not username or not password:
            self.messageLabel.setText("❌ Please enter username and password")
            self.messageLabel.setObjectName("lblDanger")
            return
        
        # Demo: accept any non-empty credentials for admin
        self.main_window.login(username, is_admin=True)
        self.clear_form()
    
    def clear_form(self):
        self.usernameInput.clear()
        self.passwordInput.clear()
        self.messageLabel.setText("")
