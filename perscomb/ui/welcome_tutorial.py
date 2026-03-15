"""Welcome Tutorial Overlay - First-time user onboarding guide.

This module provides a step-by-step tutorial overlay to guide new users
through the basic workflow of the application.
"""
from __future__ import annotations

from typing import Optional, Callable, List, Dict, Any
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtWidgets import QGraphicsOpacityEffect

from .design_tokens import Colors, Typography, Spacing, BorderRadius, Shadows


class TutorialStep:
    """Single tutorial step configuration."""
    
    def __init__(
        self,
        title: str,
        description: str,
        target_widget: Optional[QtWidgets.QWidget] = None,
        target_name: str = "",
        arrow_direction: str = "bottom",  # top, bottom, left, right
        highlight_target: bool = True
    ):
        self.title = title
        self.description = description
        self.target_widget = target_widget
        self.target_name = target_name  # fallback for finding widget by object name
        self.arrow_direction = arrow_direction
        self.highlight_target = highlight_target


class WelcomeTutorialOverlay(QtWidgets.QWidget):
    """Semi-transparent overlay widget for displaying tutorial steps."""
    
    tutorial_finished = Signal()
    tutorial_skipped = Signal()
    
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        
        self.parent_widget = parent
        self.current_step = 0
        self.steps: List[TutorialStep] = []
        self.highlighted_widgets: List[QtWidgets.QWidget] = []
        self.highlight_effects: List[QGraphicsOpacityEffect] = []
        
        # Animation for smooth transitions
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self._setup_ui()
        self._setup_default_steps()
        self._connect_signals()
        
        # Make overlay cover entire parent
        self.setGeometry(parent.rect())
        
        # Enable mouse events
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        
        # Start hidden
        self.hide()
    
    def _setup_ui(self):
        """Setup overlay UI components."""
        self.setStyleSheet(f"""
            WelcomeTutorialOverlay {{
                background-color: {Colors.OVERLAY_DARK};
            }}
        """)
        
        # Main tutorial card
        self.tutorial_card = QtWidgets.QFrame(self)
        self.tutorial_card.setObjectName("TutorialCard")
        self.tutorial_card.setStyleSheet(f"""
            QFrame#TutorialCard {{
                background-color: {Colors.BG_PANEL};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {BorderRadius.LG};
                padding: {Spacing.XL};
            }}
        """)
        
        # Layout for tutorial card
        card_layout = QtWidgets.QVBoxLayout(self.tutorial_card)
        card_layout.setContentsMargins(24, 24, 24, 20)
        card_layout.setSpacing(16)
        
        # Header
        header_layout = QtWidgets.QHBoxLayout()
        
        self.step_indicator = QtWidgets.QLabel()
        self.step_indicator.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_MUTED};
                font-size: {Typography.FONT_SIZE_CAPTION};
                font-weight: {Typography.FONT_WEIGHT_MEDIUM};
                letter-spacing: 0.5px;
            }}
        """)
        
        header_layout.addWidget(self.step_indicator)
        header_layout.addStretch()
        
        # Skip button
        self.skip_button = QtWidgets.QPushButton("跳過導覽")
        self.skip_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_MUTED};
                border: none;
                font-size: {Typography.FONT_SIZE_SMALL};
                font-weight: {Typography.FONT_WEIGHT_MEDIUM};
                padding: 4px 8px;
                border-radius: {BorderRadius.SM};
            }}
            QPushButton:hover {{
                background-color: {Colors.BG_SUBTLE};
                color: {Colors.TEXT_SECONDARY};
            }}
        """)
        
        header_layout.addWidget(self.skip_button)
        card_layout.addLayout(header_layout)
        
        # Title
        self.title_label = QtWidgets.QLabel()
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_PRIMARY};
                font-size: {Typography.FONT_SIZE_H2};
                font-weight: {Typography.FONT_WEIGHT_BOLD};
                margin: 0px;
            }}
        """)
        card_layout.addWidget(self.title_label)
        
        # Description
        self.description_label = QtWidgets.QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
                font-size: {Typography.FONT_SIZE_BODY};
                line-height: 1.5;
                margin: 0px;
            }}
        """)
        card_layout.addWidget(self.description_label)
        
        card_layout.addSpacing(8)
        
        # Navigation buttons
        nav_layout = QtWidgets.QHBoxLayout()
        
        self.prev_button = QtWidgets.QPushButton("上一步")
        self.prev_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_SUBTLE};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {BorderRadius.MD};
                padding: 8px 16px;
                min-height: 36px;
                min-width: 80px;
                font-weight: {Typography.FONT_WEIGHT_MEDIUM};
                font-size: {Typography.FONT_SIZE_BODY};
            }}
            QPushButton:hover {{
                background-color: {Colors.BG_PANEL};
                border-color: {Colors.BORDER_HOVER};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_SUBTLE};
                color: {Colors.TEXT_MUTED};
                border-color: {Colors.BORDER_DEFAULT};
            }}
        """)
        
        self.next_button = QtWidgets.QPushButton("下一步")
        self.next_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BRAND_PRIMARY};
                color: {Colors.TEXT_ON_PRIMARY};
                border: 1px solid {Colors.BRAND_PRIMARY};
                border-radius: {BorderRadius.MD};
                padding: 8px 20px;
                min-height: 36px;
                min-width: 80px;
                font-weight: {Typography.FONT_WEIGHT_BOLD};
                font-size: {Typography.FONT_SIZE_BODY};
            }}
            QPushButton:hover {{
                background-color: {Colors.BRAND_PRIMARY_HOVER};
                border-color: {Colors.BRAND_PRIMARY_HOVER};
            }}
        """)
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_button)
        
        card_layout.addLayout(nav_layout)
        
        # Position tutorial card (will be adjusted in show_step)
        self.tutorial_card.resize(380, 240)
    
    def _setup_default_steps(self):
        """Setup default tutorial steps."""
        self.steps = [
            TutorialStep(
                title="歡迎使用 Fusi³",
                description="讓我們快速了解如何使用這個工具進行 SEM 影像融合分析。我們將引導您完成基本的操作流程。",
                target_widget=None,
                highlight_target=False
            ),
            TutorialStep(
                title="第一步：載入影像資料夾",
                description="點擊此按鈕選擇包含 SEM 影像的資料夾。系統會自動掃描並載入所有支援格式的影像檔案。",
                target_name="load_folder_btn",
                arrow_direction="bottom"
            ),
            TutorialStep(
                title="第二步：選擇 Base 影像",
                description="從載入的影像列表中選擇一張作為基準影像（Base）。這將是其他影像對齊和比較的參考標準。",
                target_name="base_combo",
                arrow_direction="right"
            ),
            TutorialStep(
                title="第三步：選擇 Compare 影像",
                description="勾選一張或多張影像作為比較對象。系統會將這些影像與 Base 影像進行對齊和運算分析。",
                target_name="compare_list",
                arrow_direction="right"
            ),
            TutorialStep(
                title="第四步：執行運算",
                description="設定完成後，點擊此按鈕開始執行影像對齊和融合運算。運算完成後您可以查看結果和分析報告。",
                target_name="compute_btn",
                arrow_direction="top"
            ),
            TutorialStep(
                title="導覽完成！",
                description="您已經了解了基本操作流程。現在可以開始使用 Fusi³ 進行您的 SEM 影像分析工作了。如需更多幫助，請參考說明文件。",
                target_widget=None,
                highlight_target=False
            )
        ]
    
    def _connect_signals(self):
        """Connect signal handlers."""
        self.skip_button.clicked.connect(self._on_skip)
        self.prev_button.clicked.connect(self._on_prev)
        self.next_button.clicked.connect(self._on_next)
    
    def show_tutorial(self):
        """Show tutorial overlay and start from first step."""
        self.current_step = 0
        self.show()
        self.raise_()
        self._show_current_step()
    
    def _show_current_step(self):
        """Display current tutorial step."""
        if not (0 <= self.current_step < len(self.steps)):
            return
        
        step = self.steps[self.current_step]
        
        # Update step indicator
        self.step_indicator.setText(f"步驟 {self.current_step + 1} / {len(self.steps)}")
        
        # Update content
        self.title_label.setText(step.title)
        self.description_label.setText(step.description)
        
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_step > 0)
        
        if self.current_step == len(self.steps) - 1:
            self.next_button.setText("完成")
        else:
            self.next_button.setText("下一步")
        
        # Clear previous highlights
        self._clear_highlights()
        
        # Find and highlight target widget
        target_widget = self._find_target_widget(step)
        if target_widget and step.highlight_target:
            self._highlight_widget(target_widget)
        
        # Position tutorial card
        self._position_card(target_widget, step.arrow_direction)
    
    def _find_target_widget(self, step: TutorialStep) -> Optional[QtWidgets.QWidget]:
        """Find target widget for current step."""
        if step.target_widget:
            return step.target_widget
        
        if step.target_name:
            # Search in parent widget hierarchy
            return self.parent_widget.findChild(QtWidgets.QWidget, step.target_name)
        
        return None
    
    def _highlight_widget(self, widget: QtWidgets.QWidget):
        """Add highlight effect to target widget."""
        if not widget:
            return
        
        # Store for cleanup
        self.highlighted_widgets.append(widget)
        
        # Create glow effect
        effect = QGraphicsOpacityEffect()
        effect.setOpacity(1.0)
        widget.setGraphicsEffect(effect)
        self.highlight_effects.append(effect)
        
        # Add temporary styling
        original_style = widget.styleSheet()
        widget.setProperty("originalStyle", original_style)
        
        highlight_style = f"""
            QWidget {{
                border: 2px solid {Colors.BRAND_PRIMARY};
                border-radius: {BorderRadius.MD};
                background-color: {Colors.BRAND_PRIMARY_SOFT};
            }}
        """
        
        widget.setStyleSheet(original_style + highlight_style)
    
    def _clear_highlights(self):
        """Remove highlight effects from all widgets."""
        for widget in self.highlighted_widgets:
            if widget:
                # Restore original style
                original_style = widget.property("originalStyle")
                if original_style is not None:
                    widget.setStyleSheet(original_style)
                
                # Remove graphics effect
                widget.setGraphicsEffect(None)
        
        self.highlighted_widgets.clear()
        self.highlight_effects.clear()
    
    def _position_card(self, target_widget: Optional[QtWidgets.QWidget], arrow_direction: str):
        """Position tutorial card relative to target widget."""
        parent_rect = self.parent_widget.rect()
        card_size = self.tutorial_card.size()
        
        # Default center position
        x = (parent_rect.width() - card_size.width()) // 2
        y = (parent_rect.height() - card_size.height()) // 2
        
        if target_widget:
            # Get target widget position in parent coordinates
            target_pos = target_widget.mapTo(self.parent_widget, QtCore.QPoint(0, 0))
            target_rect = QtCore.QRect(target_pos, target_widget.size())
            
            margin = 30
            
            if arrow_direction == "bottom":
                # Card above target
                x = max(20, min(target_rect.center().x() - card_size.width() // 2, 
                               parent_rect.width() - card_size.width() - 20))
                y = max(20, target_rect.top() - card_size.height() - margin)
            
            elif arrow_direction == "top":
                # Card below target
                x = max(20, min(target_rect.center().x() - card_size.width() // 2,
                               parent_rect.width() - card_size.width() - 20))
                y = min(target_rect.bottom() + margin, 
                       parent_rect.height() - card_size.height() - 20)
            
            elif arrow_direction == "right":
                # Card to the left of target
                x = max(20, target_rect.left() - card_size.width() - margin)
                y = max(20, min(target_rect.center().y() - card_size.height() // 2,
                               parent_rect.height() - card_size.height() - 20))
            
            elif arrow_direction == "left":
                # Card to the right of target
                x = min(target_rect.right() + margin, 
                       parent_rect.width() - card_size.width() - 20)
                y = max(20, min(target_rect.center().y() - card_size.height() // 2,
                               parent_rect.height() - card_size.height() - 20))
        
        self.tutorial_card.move(x, y)
    
    def _on_prev(self):
        """Handle previous step button click."""
        if self.current_step > 0:
            self.current_step -= 1
            self._show_current_step()
    
    def _on_next(self):
        """Handle next step button click."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._show_current_step()
        else:
            # Last step - finish tutorial
            self._finish_tutorial()
    
    def _on_skip(self):
        """Handle skip button click."""
        self._clear_highlights()
        self.hide()
        self.tutorial_skipped.emit()
    
    def _finish_tutorial(self):
        """Finish tutorial and hide overlay."""
        self._clear_highlights()
        self.hide()
        self.tutorial_finished.emit()
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Escape:
            self._on_skip()
        elif event.key() == Qt.Key.Key_Left:
            self._on_prev()
        elif event.key() == Qt.Key.Key_Right:
            self._on_next()
        else:
            super().keyPressEvent(event)
    
    def resizeEvent(self, event: QtGui.QResizeEvent):
        """Handle parent widget resize."""
        super().resizeEvent(event)
        # Update overlay size to match parent
        self.setGeometry(self.parent_widget.rect())
        # Reposition tutorial card
        if self.isVisible() and 0 <= self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            target_widget = self._find_target_widget(step)
            self._position_card(target_widget, step.arrow_direction)


def should_show_tutorial() -> bool:
    """Check if tutorial should be shown for first-time users."""
    settings = QtCore.QSettings()
    return not settings.value("tutorial/completed", False, type=bool)


def mark_tutorial_completed():
    """Mark tutorial as completed in settings."""
    settings = QtCore.QSettings()
    settings.setValue("tutorial/completed", True)
    settings.sync()


def reset_tutorial_flag():
    """Reset tutorial flag (for testing/debugging)."""
    settings = QtCore.QSettings()
    settings.remove("tutorial/completed")
    settings.sync()
