"""Alignment Progress Widget - Real-time progress visualization for image alignment.

Provides a comprehensive progress display for batch image alignment operations,
including current pair info, progress bar, time estimation, and cancellation support.

Author: Fusi³ Team
Date: 2026-01-23
Version: 1.0
"""
from __future__ import annotations

import time
from typing import Optional

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal

from .design_tokens import Colors, Typography, Spacing, BorderRadius, Shadows


class AlignmentProgressWidget(QtWidgets.QWidget):
    """Real-time progress widget for image alignment operations.
    
    Features:
    - Progress bar with percentage display
    - Current alignment pair information
    - Estimated completion time
    - Cancel operation support
    - Status indicators (processing, completed, cancelled)
    """
    
    # Signals
    cancel_requested = Signal()  # 用戶點擊取消按鈕
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Progress tracking state
        self._total_pairs = 0
        self._current_pair = 0
        self._start_time: Optional[float] = None
        self._is_cancelled = False
        self._is_completed = False
        
        # UI components
        self._progress_bar: Optional[QtWidgets.QProgressBar] = None
        self._status_label: Optional[QtWidgets.QLabel] = None
        self._current_pair_label: Optional[QtWidgets.QLabel] = None
        self._time_estimate_label: Optional[QtWidgets.QLabel] = None
        self._cancel_button: Optional[QtWidgets.QPushButton] = None
        
        self._setup_ui()
        self._apply_styles()
        
    def _setup_ui(self):
        """Initialize the UI components."""
        self.setFixedHeight(140)
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        
        # Header section
        header_layout = QtWidgets.QHBoxLayout()
        
        # Status label
        self._status_label = QtWidgets.QLabel("準備開始對齊...")
        self._status_label.setProperty("statusLabel", True)
        header_layout.addWidget(self._status_label)
        
        header_layout.addStretch()
        
        # Cancel button
        self._cancel_button = QtWidgets.QPushButton("取消")
        self._cancel_button.setProperty("variant", "ghost")
        self._cancel_button.setFixedWidth(64)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        header_layout.addWidget(self._cancel_button)
        
        layout.addLayout(header_layout)
        
        # Progress bar
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setTextVisible(False)
        layout.addWidget(self._progress_bar)
        
        # Info section
        info_layout = QtWidgets.QHBoxLayout()
        info_layout.setSpacing(16)
        
        # Current pair info
        self._current_pair_label = QtWidgets.QLabel("等待開始...")
        self._current_pair_label.setProperty("pairInfo", True)
        info_layout.addWidget(self._current_pair_label)
        
        info_layout.addStretch()
        
        # Time estimate
        self._time_estimate_label = QtWidgets.QLabel("")
        self._time_estimate_label.setProperty("timeEstimate", True)
        info_layout.addWidget(self._time_estimate_label)
        
        layout.addLayout(info_layout)
        
    def _apply_styles(self):
        """Apply consistent styling using design tokens."""
        self.setStyleSheet(f"""
            AlignmentProgressWidget {{
                background-color: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {BorderRadius.MD};
            }}
            
            QLabel[statusLabel="true"] {{
                color: {Colors.TEXT_PRIMARY};
                font-size: {Typography.FONT_SIZE_BODY};
                font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
                font-family: {Typography.FONT_FAMILY};
            }}
            
            QLabel[pairInfo="true"] {{
                color: {Colors.TEXT_SECONDARY};
                font-size: {Typography.FONT_SIZE_SMALL};
                font-weight: {Typography.FONT_WEIGHT_MEDIUM};
                font-family: {Typography.FONT_FAMILY};
            }}
            
            QLabel[timeEstimate="true"] {{
                color: {Colors.TEXT_MUTED};
                font-size: {Typography.FONT_SIZE_SMALL};
                font-family: {Typography.FONT_FAMILY_MONO};
            }}
            
            QPushButton[variant="ghost"] {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {BorderRadius.SM};
                padding: {Spacing.XS} {Spacing.MD};
                font-size: {Typography.FONT_SIZE_SMALL};
                font-weight: {Typography.FONT_WEIGHT_MEDIUM};
                font-family: {Typography.FONT_FAMILY};
            }}
            
            QPushButton[variant="ghost"]:hover {{
                background-color: {Colors.BG_SUBTLE};
                border-color: {Colors.BORDER_HOVER};
                color: {Colors.TEXT_PRIMARY};
            }}
            
            QPushButton[variant="ghost"]:pressed {{
                background-color: {Colors.BRAND_PRIMARY_SOFT};
            }}
            
            QPushButton[variant="ghost"]:disabled {{
                background-color: transparent;
                color: {Colors.TEXT_MUTED};
                border-color: {Colors.BORDER_DEFAULT};
            }}
            
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {Colors.BG_SUBTLE};
            }}
            
            QProgressBar::chunk {{
                background-color: {Colors.BRAND_PRIMARY};
                border-radius: 4px;
            }}
        """)
        
    def start_progress(self, total_pairs: int):
        """Start a new progress tracking session.
        
        Args:
            total_pairs: Total number of image pairs to process
        """
        self._total_pairs = max(1, total_pairs)  # Prevent division by zero
        self._current_pair = 0
        self._start_time = time.time()
        self._is_cancelled = False
        self._is_completed = False
        
        # Update UI state
        self._progress_bar.setValue(0)
        self._status_label.setText(f"對齊進行中... (0/{self._total_pairs})")
        self._current_pair_label.setText("準備處理第一個影像對...")
        self._time_estimate_label.setText("")
        self._cancel_button.setEnabled(True)
        self._cancel_button.setText("取消")
        
    def update_progress(self, current_pair: int, pair_label: str = ""):
        """Update the current progress.
        
        Args:
            current_pair: Current pair index (0-based)
            pair_label: Human-readable description of current pair
        """
        if self._is_cancelled or self._is_completed:
            return
            
        self._current_pair = current_pair
        
        # Calculate progress percentage
        progress_percent = int((current_pair / self._total_pairs) * 100)
        self._progress_bar.setValue(progress_percent)
        
        # Update status
        self._status_label.setText(f"對齊進行中... ({current_pair}/{self._total_pairs})")
        
        # Update current pair info
        if pair_label:
            self._current_pair_label.setText(f"正在對齊: {pair_label}")
        else:
            self._current_pair_label.setText(f"正在處理第 {current_pair + 1} 個影像對...")
        
        # Calculate time estimate
        if self._start_time and current_pair > 0:
            elapsed = time.time() - self._start_time
            avg_time_per_pair = elapsed / current_pair
            remaining_pairs = self._total_pairs - current_pair
            estimated_remaining = avg_time_per_pair * remaining_pairs
            
            self._time_estimate_label.setText(self._format_time_estimate(estimated_remaining))
        
    def complete_progress(self):
        """Mark the progress as completed."""
        if self._is_cancelled:
            return
            
        self._is_completed = True
        self._progress_bar.setValue(100)
        self._status_label.setText(f"對齊完成！({self._total_pairs}/{self._total_pairs})")
        self._current_pair_label.setText("所有影像對已成功對齊")
        
        if self._start_time:
            total_time = time.time() - self._start_time
            self._time_estimate_label.setText(f"總耗時: {self._format_duration(total_time)}")
        
        self._cancel_button.setText("完成")
        self._cancel_button.setEnabled(False)
        
    def cancel_progress(self):
        """Mark the progress as cancelled."""
        if self._is_completed:
            return
            
        self._is_cancelled = True
        self._status_label.setText(f"已取消 ({self._current_pair}/{self._total_pairs})")
        self._current_pair_label.setText("對齊操作已被用戶取消")
        
        if self._start_time:
            elapsed = time.time() - self._start_time
            self._time_estimate_label.setText(f"已耗時: {self._format_duration(elapsed)}")
        
        self._cancel_button.setText("已取消")
        self._cancel_button.setEnabled(False)
        
    def reset(self):
        """Reset the widget to initial state."""
        self._total_pairs = 0
        self._current_pair = 0
        self._start_time = None
        self._is_cancelled = False
        self._is_completed = False
        
        self._progress_bar.setValue(0)
        self._status_label.setText("準備開始對齊...")
        self._current_pair_label.setText("等待開始...")
        self._time_estimate_label.setText("")
        self._cancel_button.setEnabled(True)
        self._cancel_button.setText("取消")
        
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        if not self._is_cancelled and not self._is_completed:
            self.cancel_requested.emit()
            
    def _format_time_estimate(self, seconds: float) -> str:
        """Format estimated remaining time.
        
        Args:
            seconds: Remaining time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"預估剩餘: {int(seconds)}秒"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"預估剩餘: {minutes}分鐘"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"預估剩餘: {hours}小時{minutes}分鐘"
            
    def _format_duration(self, seconds: float) -> str:
        """Format elapsed time duration.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}分{remaining_seconds}秒"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}小時{minutes}分鐘"
            
    @property
    def is_cancelled(self) -> bool:
        """Check if the progress was cancelled."""
        return self._is_cancelled
        
    @property
    def is_completed(self) -> bool:
        """Check if the progress was completed."""
        return self._is_completed
        
    @property
    def current_progress(self) -> tuple[int, int]:
        """Get current progress as (current, total) pair."""
        return (self._current_pair, self._total_pairs)
