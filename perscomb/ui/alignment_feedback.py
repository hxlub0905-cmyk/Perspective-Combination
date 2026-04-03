"""
對齊失敗視覺回饋模組
提供影像對齊結果的視覺化警告和建議操作

作者：Claude AI
日期：2026-01-23
版本：1.0
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPalette

from .design_tokens import Colors, Typography, Spacing, BorderRadius, Shadows
from ..core.ebeam_snr import AlignResult


class AlignmentFeedbackWidget(QWidget):
    """
    對齊結果回饋 Widget
    
    根據 AlignResult 的狀態顯示不同的視覺回饋：
    - 成功：綠色標記 + 簡短確認訊息
    - 警告：橙色標記 + 建議操作
    - 失敗：紅色標記 + 詳細修正建議
    """
    
    # 信號：使用者點擊重新對齊按鈕
    retry_alignment_requested = Signal()
    
    # 信號：使用者點擊查看詳細資訊
    show_details_requested = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._current_result: Optional[AlignResult] = None
        
    def _setup_ui(self):
        """初始化 UI 元件"""
        self.setObjectName("AlignmentFeedbackWidget")
        
        # 主佈局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(int(Spacing.SM.replace('px', '')))
        
        # 回饋卡片容器
        self._feedback_card = QFrame()
        self._feedback_card.setObjectName("FeedbackCard")
        layout.addWidget(self._feedback_card)
        
        # 卡片內部佈局
        card_layout = QVBoxLayout(self._feedback_card)
        card_layout.setContentsMargins(
            int(Spacing.LG.replace('px', '')),
            int(Spacing.MD.replace('px', '')),
            int(Spacing.LG.replace('px', '')),
            int(Spacing.MD.replace('px', ''))
        )
        card_layout.setSpacing(int(Spacing.MD.replace('px', '')))
        
        # 標題行（圖示 + 狀態文字）
        title_layout = QHBoxLayout()
        title_layout.setSpacing(int(Spacing.SM.replace('px', '')))
        
        # 狀態圖示
        self._status_icon = QLabel()
        self._status_icon.setObjectName("StatusIcon")
        self._status_icon.setFixedSize(20, 20)
        self._status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(self._status_icon)
        
        # 狀態標題
        self._status_title = QLabel()
        self._status_title.setObjectName("StatusTitle")
        title_layout.addWidget(self._status_title)
        
        title_layout.addStretch()
        card_layout.addLayout(title_layout)
        
        # 詳細訊息
        self._message_label = QLabel()
        self._message_label.setObjectName("MessageLabel")
        self._message_label.setWordWrap(True)
        card_layout.addWidget(self._message_label)
        
        # 建議操作區域
        self._suggestions_widget = QWidget()
        suggestions_layout = QVBoxLayout(self._suggestions_widget)
        suggestions_layout.setContentsMargins(0, 0, 0, 0)
        suggestions_layout.setSpacing(int(Spacing.SM.replace('px', '')))
        
        # 建議標題
        self._suggestions_title = QLabel("建議操作：")
        self._suggestions_title.setObjectName("SuggestionsTitle")
        suggestions_layout.addWidget(self._suggestions_title)
        
        # 建議列表
        self._suggestions_list = QLabel()
        self._suggestions_list.setObjectName("SuggestionsList")
        self._suggestions_list.setWordWrap(True)
        suggestions_layout.addWidget(self._suggestions_list)
        
        card_layout.addWidget(self._suggestions_widget)
        
        # 操作按鈕區域
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(int(Spacing.SM.replace('px', '')))
        
        # 重新對齊按鈕
        self._retry_button = QPushButton("重新對齊")
        self._retry_button.setObjectName("RetryButton")
        self._retry_button.clicked.connect(self.retry_alignment_requested.emit)
        buttons_layout.addWidget(self._retry_button)
        
        # 查看詳細資訊按鈕
        self._details_button = QPushButton("查看詳細資訊")
        self._details_button.setObjectName("DetailsButton")
        self._details_button.clicked.connect(self.show_details_requested.emit)
        buttons_layout.addWidget(self._details_button)
        
        buttons_layout.addStretch()
        card_layout.addLayout(buttons_layout)
        
        # 預設隱藏
        self.hide()
        
        # 套用樣式
        self._apply_styles()
        
    def _apply_styles(self):
        """套用 CSS 樣式"""
        self.setStyleSheet(f"""
            #AlignmentFeedbackWidget {{
                background: transparent;
            }}
            
            #FeedbackCard {{
                background: {Colors.BG_PANEL};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {BorderRadius.MD};
                padding: 0px;
            }}
            
            #FeedbackCard.success {{
                border-color: {Colors.SUCCESS};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {Colors.BG_PANEL}, 
                    stop:1 rgba(22, 163, 74, 0.05));
            }}
            
            #FeedbackCard.warning {{
                border-color: {Colors.BRAND_PRIMARY};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {Colors.BG_PANEL}, 
                    stop:1 {Colors.BRAND_PRIMARY_SOFT});
            }}
            
            #FeedbackCard.error {{
                border-color: {Colors.WARNING};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {Colors.BG_PANEL}, 
                    stop:1 rgba(220, 38, 38, 0.05));
            }}
            
            #StatusIcon {{
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
                background: {Colors.BG_SUBTLE};
            }}
            
            #StatusIcon.success {{
                background: {Colors.SUCCESS};
                color: white;
            }}
            
            #StatusIcon.warning {{
                background: {Colors.BRAND_PRIMARY};
                color: white;
            }}
            
            #StatusIcon.error {{
                background: {Colors.WARNING};
                color: white;
            }}
            
            #StatusTitle {{
                font-size: {Typography.FONT_SIZE_H3};
                font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
                color: {Colors.TEXT_PRIMARY};
            }}
            
            #MessageLabel {{
                font-size: {Typography.FONT_SIZE_BODY};
                color: {Colors.TEXT_SECONDARY};
                line-height: 1.4;
            }}
            
            #SuggestionsTitle {{
                font-size: {Typography.FONT_SIZE_SMALL};
                font-weight: {Typography.FONT_WEIGHT_SEMIBOLD};
                color: {Colors.TEXT_PRIMARY};
                margin-top: {Spacing.SM};
            }}
            
            #SuggestionsList {{
                font-size: {Typography.FONT_SIZE_SMALL};
                color: {Colors.TEXT_SECONDARY};
                line-height: 1.5;
                padding-left: {Spacing.LG};
            }}
            
            #RetryButton, #DetailsButton {{
                background: {Colors.BG_SUBTLE};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {BorderRadius.SM};
                padding: {Spacing.SM} {Spacing.MD};
                font-size: {Typography.FONT_SIZE_SMALL};
                color: {Colors.TEXT_PRIMARY};
                min-width: 80px;
            }}
            
            #RetryButton:hover, #DetailsButton:hover {{
                background: {Colors.BRAND_PRIMARY_SOFT};
                border-color: {Colors.BRAND_PRIMARY};
            }}
            
            #RetryButton:pressed, #DetailsButton:pressed {{
                background: {Colors.BRAND_PRIMARY};
                color: white;
            }}
        """)
        
    def update_feedback(self, result: AlignResult):
        """
        更新回饋內容
        
        Args:
            result: 對齊結果
        """
        self._current_result = result
        
        if result is None:
            self.hide()
            return
            
        # 根據對齊狀態設定回饋內容
        feedback_config = self._get_feedback_config(result)
        
        # 更新 UI 元件
        self._status_icon.setText(feedback_config['icon'])
        self._status_icon.setProperty('class', feedback_config['status'])
        self._status_title.setText(feedback_config['title'])
        self._message_label.setText(feedback_config['message'])
        
        # 設定卡片樣式
        self._feedback_card.setProperty('class', feedback_config['status'])
        
        # 顯示/隱藏建議區域
        if feedback_config['suggestions']:
            self._suggestions_list.setText(feedback_config['suggestions'])
            self._suggestions_widget.show()
        else:
            self._suggestions_widget.hide()
            
        # 顯示/隱藏按鈕
        self._retry_button.setVisible(feedback_config['show_retry'])
        self._details_button.setVisible(feedback_config['show_details'])
        
        # 重新套用樣式（確保 property 變更生效）
        self.style().unpolish(self._feedback_card)
        self.style().polish(self._feedback_card)
        self.style().unpolish(self._status_icon)
        self.style().polish(self._status_icon)
        
        # 顯示 widget
        self.show()
        
    def _get_feedback_config(self, result: AlignResult) -> dict:
        """
        根據對齊結果取得回饋配置
        
        Args:
            result: 對齊結果
            
        Returns:
            回饋配置字典
        """
        # 判斷對齊品質
        score = getattr(result, 'correlation_score', 0.0)
        is_success = getattr(result, 'success', False)
        
        if is_success and score >= 0.8:
            # 高品質對齊
            return {
                'status': 'success',
                'icon': '✓',
                'title': '對齊成功',
                'message': f'影像對齊品質良好（相關係數: {score:.3f}），可以進行下一步操作。',
                'suggestions': '',
                'show_retry': False,
                'show_details': True
            }
            
        elif is_success and score >= 0.6:
            # 中等品質對齊
            return {
                'status': 'warning',
                'icon': '⚠',
                'title': '對齊品質中等',
                'message': f'影像已成功對齊，但品質可能不夠理想（相關係數: {score:.3f}）。',
                'suggestions': (
                    '• 檢查影像是否存在明顯的幾何差異\n'
                    '• 嘗試調整對齊參數或範圍\n'
                    '• 確認影像品質和解析度是否合適\n'
                    '• 可繼續處理，但建議密切關注結果品質'
                ),
                'show_retry': True,
                'show_details': True
            }
            
        else:
            # 對齊失敗或低品質
            score_text = f'（相關係數: {score:.3f}）' if score > 0 else ''
            return {
                'status': 'error',
                'icon': '✗',
                'title': '對齊失敗',
                'message': f'影像對齊未成功或品質過低{score_text}，無法進行可靠的影像融合。',
                'suggestions': (
                    '• 確認兩張影像拍攝的是相同區域\n'
                    '• 檢查影像是否存在嚴重的位移、旋轉或變形\n'
                    '• 嘗試手動裁切影像至共同視野範圍\n'
                    '• 調整影像亮度/對比度以增強特徵\n'
                    '• 考慮使用更高解析度的影像\n'
                    '• 如問題持續，請檢查影像採集參數設定'
                ),
                'show_retry': True,
                'show_details': True
            }
            
    def clear_feedback(self):
        """清除回饋內容並隱藏 widget"""
        self._current_result = None
        self.hide()
        
    def get_current_result(self) -> Optional[AlignResult]:
        """取得目前顯示的對齊結果"""
        return self._current_result
