"""Welcome Overlay Widget - First-time user onboarding guide.

This widget displays a welcome overlay with step-by-step guidance
for new users when no images are loaded.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal

from .design_tokens import Colors, Typography, Spacing, BorderRadius, Shadows


class WelcomeOverlayWidget(QtWidgets.QWidget):
    """Welcome overlay showing onboarding steps for first-time users."""
    
    # Signals for triggering main dialog actions
    load_folder_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._apply_styles()
    
    def _setup_ui(self):
        """Setup the welcome overlay UI components."""
        # Main layout - center the welcome card
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Welcome card container
        self.welcome_card = QtWidgets.QFrame()
        self.welcome_card.setObjectName("WelcomeCard")
        self.welcome_card.setFixedSize(480, 360)
        main_layout.addWidget(self.welcome_card)
        
        # Card layout
        card_layout = QtWidgets.QVBoxLayout(self.welcome_card)
        card_layout.setContentsMargins(32, 28, 32, 28)
        card_layout.setSpacing(24)
        
        # Title
        title_label = QtWidgets.QLabel("歡迎使用 Fusi³")
        title_label.setObjectName("WelcomeTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QtWidgets.QLabel("SEM 影像融合與缺陷分析工具")
        subtitle_label.setObjectName("WelcomeSubtitle")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(subtitle_label)
        
        # Steps container
        steps_frame = QtWidgets.QFrame()
        steps_layout = QtWidgets.QVBoxLayout(steps_frame)
        steps_layout.setSpacing(16)
        
        # Step 1: Load images
        step1_widget = self._create_step_widget(
            "1", "載入影像資料夾", "選擇包含 SEM 影像的資料夾",
            "📁", self._on_load_folder_clicked
        )
        steps_layout.addWidget(step1_widget)
        
        # Step 2: Select base
        step2_widget = self._create_step_widget(
            "2", "選擇 Base 影像", "選定作為基準的影像",
            "🎯", None
        )
        steps_layout.addWidget(step2_widget)
        
        # Step 3: Select compare
        step3_widget = self._create_step_widget(
            "3", "選擇 Compare 影像", "選擇要比較的影像",
            "⚖️", None
        )
        steps_layout.addWidget(step3_widget)
        
        # Step 4: Execute
        step4_widget = self._create_step_widget(
            "4", "執行運算", "開始影像對齊與分析",
            "⚡", None
        )
        steps_layout.addWidget(step4_widget)
        
        card_layout.addWidget(steps_frame)
        
        # Get started button
        get_started_btn = QtWidgets.QPushButton("開始使用")
        get_started_btn.setObjectName("GetStartedButton")
        get_started_btn.clicked.connect(self._on_get_started_clicked)
        card_layout.addWidget(get_started_btn)
    
    def _create_step_widget(
        self, 
        step_num: str, 
        title: str, 
        description: str, 
        icon: str,
        click_handler: Optional[Callable] = None
    ) -> QtWidgets.QWidget:
        """Create a single step widget."""
        step_widget = QtWidgets.QFrame()
        step_widget.setObjectName("StepWidget")
        
        if click_handler:
            step_widget.setCursor(Qt.CursorShape.PointingHandCursor)
            step_widget.mousePressEvent = lambda event: click_handler()
        
        step_layout = QtWidgets.QHBoxLayout(step_widget)
        step_layout.setContentsMargins(16, 12, 16, 12)
        step_layout.setSpacing(12)
        
        # Step number circle
        step_circle = QtWidgets.QLabel(step_num)
        step_circle.setObjectName("StepCircle")
        step_circle.setFixedSize(28, 28)
        step_circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_layout.addWidget(step_circle)
        
        # Icon
        icon_label = QtWidgets.QLabel(icon)
        icon_label.setObjectName("StepIcon")
        icon_label.setFixedSize(24, 24)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_layout.addWidget(icon_label)
        
        # Text content
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(4)
        
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("StepTitle")
        text_layout.addWidget(title_label)
        
        desc_label = QtWidgets.QLabel(description)
        desc_label.setObjectName("StepDescription")
        text_layout.addWidget(desc_label)
        
        step_layout.addLayout(text_layout)
        step_layout.addStretch()
        
        return step_widget
    
    def _apply_styles(self):
        """Apply styles to the welcome overlay components."""
        self.setStyleSheet(f"""
            WelcomeOverlayWidget {{
                background-color: {Colors.OVERLAY_LIGHT};
            }}
            
            QFrame#WelcomeCard {{
                background-color: {Colors.BG_PANEL};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {BorderRadius.XL};
            }}
            
            QLabel#WelcomeTitle {{
                color: {Colors.TEXT_PRIMARY};
                font-size: {Typography.FONT_SIZE_H1};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
                font-family: {Typography.FONT_FAMILY};
                margin-bottom: 4px;
            }}
            
            QLabel#WelcomeSubtitle {{
                color: {Colors.TEXT_SECONDARY};
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_MEDIUM};
                font-family: {Typography.FONT_FAMILY};
                margin-bottom: 8px;
            }}
            
            QFrame#StepWidget {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {BorderRadius.MD};
            }}
            
            QFrame#StepWidget:hover {{
                background-color: {Colors.BG_SUBTLE};
                border-color: {Colors.BORDER_HOVER};
            }}
            
            QLabel#StepCircle {{
                background-color: {Colors.BRAND_PRIMARY};
                color: {Colors.TEXT_ON_PRIMARY};
                border-radius: 14px;
                font-size: {Typography.FONT_SIZE_SMALL};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
                font-family: {Typography.FONT_FAMILY};
            }}
            
            QLabel#StepIcon {{
                font-size: 18px;
            }}
            
            QLabel#StepTitle {{
                color: {Colors.TEXT_PRIMARY};
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
                font-family: {Typography.FONT_FAMILY};
            }}
            
            QLabel#StepDescription {{
                color: {Colors.TEXT_SECONDARY};
                font-size: {Typography.FONT_SIZE_SMALL};
                font-family: {Typography.FONT_FAMILY};
            }}
            
            QPushButton#GetStartedButton {{
                background-color: {Colors.BRAND_PRIMARY};
                color: {Colors.TEXT_ON_PRIMARY};
                border: 1px solid {Colors.BRAND_PRIMARY};
                border-radius: {BorderRadius.MD};
                padding: 12px 32px;
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
                font-family: {Typography.FONT_FAMILY};
                min-height: 44px;
            }}
            
            QPushButton#GetStartedButton:hover {{
                background-color: {Colors.BRAND_PRIMARY_HOVER};
                border-color: {Colors.BRAND_PRIMARY_HOVER};
            }}
            
            QPushButton#GetStartedButton:pressed {{
                background-color: {Colors.BRAND_PRIMARY_PRESSED};
                border-color: {Colors.BRAND_PRIMARY_PRESSED};
            }}
        """)
    
    def _on_load_folder_clicked(self):
        """Handle load folder step click."""
        self.load_folder_requested.emit()
    
    def _on_get_started_clicked(self):
        """Handle get started button click."""
        self.load_folder_requested.emit()
