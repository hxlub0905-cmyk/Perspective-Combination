"""ROI Visual Feedback System - Enhanced ROI selection experience.

This module provides visual feedback for ROI selection operations:
1. Drag preview with real-time rectangle display
2. Instant size display showing pixel dimensions
3. Highlight effects for selected ROI regions
4. Hover states and visual cues

Author: Claude AI
Date: 2026-01-23
Version: 1.0
"""
from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
import cv2
from PySide6 import QtCore, QtGui

from .design_tokens import Colors, Typography, Spacing


class ROIFeedbackRenderer:
    """ROI 視覺回饋渲染器，處理各種視覺效果繪製。"""
    
    def __init__(self):
        # 顏色配置（BGR 格式用於 OpenCV）
        self.colors = {
            'preview': self._hex_to_bgr(Colors.BRAND_PRIMARY),
            'preview_fill': self._hex_to_bgr(Colors.BRAND_PRIMARY_SOFT),
            'selected': self._hex_to_bgr(Colors.SUCCESS),
            'selected_fill': self._hex_to_bgr_alpha(Colors.SUCCESS, 0.2),
            'hover': self._hex_to_bgr(Colors.BRAND_PRIMARY_HOVER),
            'text_bg': (255, 255, 255),
            'text_fg': self._hex_to_bgr(Colors.TEXT_PRIMARY),
            'border_active': self._hex_to_bgr(Colors.BORDER_ACTIVE)
        }
        
        # 線條寬度配置
        self.line_widths = {
            'preview': 2,
            'selected': 3,
            'hover': 2
        }
    
    def _hex_to_bgr(self, color: str) -> Tuple[int, int, int]:
        """將十六進制顏色轉換為 BGR 元組。"""
        c = color.lstrip("#")
        return (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))
    
    def _hex_to_bgr_alpha(self, color: str, alpha: float) -> Tuple[int, int, int]:
        """將十六進制顏色轉換為帶透明度的 BGR。"""
        bgr = self._hex_to_bgr(color)
        # 這裡返回基本 BGR，透明度在繪製時處理
        return bgr
    
    def draw_preview_rect(self, image: np.ndarray, rect: Tuple[int, int, int, int], 
                         size_text: str = "") -> np.ndarray:
        """繪製拖拽預覽矩形。
        
        Args:
            image: 要繪製的影像
            rect: 矩形座標 (x, y, w, h)
            size_text: 尺寸顯示文字
            
        Returns:
            繪製後的影像
        """
        if len(rect) != 4:
            return image
            
        img = image.copy()
        x, y, w, h = rect
        
        # 繪製半透明填充
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), 
                     self.colors['preview_fill'], -1)
        img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)
        
        # 繪製邊框
        cv2.rectangle(img, (x, y), (x + w, y + h), 
                     self.colors['preview'], self.line_widths['preview'])
        
        # 繪製尺寸文字
        if size_text:
            self._draw_size_label(img, rect, size_text)
            
        return img
    
    def draw_selected_rois(self, image: np.ndarray, 
                          rois: List[Tuple[Tuple[int, int, int, int], str]]) -> np.ndarray:
        """繪製已選擇的 ROI 區域。
        
        Args:
            image: 要繪製的影像
            rois: ROI 列表，每個元素為 ((x, y, w, h), name)
            
        Returns:
            繪製後的影像
        """
        if not rois:
            return image
            
        img = image.copy()
        
        for rect, name in rois:
            x, y, w, h = rect
            
            # 繪製高亮邊框
            cv2.rectangle(img, (x, y), (x + w, y + h), 
                         self.colors['selected'], self.line_widths['selected'])
            
            # 繪製角落標記
            self._draw_corner_markers(img, rect)
            
            # 繪製名稱標籤
            if name:
                self._draw_roi_label(img, rect, name)
                
        return img
    
    def draw_hover_effect(self, image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
        """繪製懸停效果。
        
        Args:
            image: 要繪製的影像
            rect: 矩形座標 (x, y, w, h)
            
        Returns:
            繪製後的影像
        """
        if len(rect) != 4:
            return image
            
        img = image.copy()
        x, y, w, h = rect
        
        # 繪製懸停邊框
        cv2.rectangle(img, (x, y), (x + w, y + h), 
                     self.colors['hover'], self.line_widths['hover'])
        
        return img
    
    def _draw_size_label(self, image: np.ndarray, rect: Tuple[int, int, int, int], 
                        text: str) -> None:
        """繪製尺寸標籤。"""
        x, y, w, h = rect
        
        # 計算文字位置（矩形右下角）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # 獲取文字尺寸
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 計算標籤背景位置
        label_x = min(x + w - text_w - 8, image.shape[1] - text_w - 8)
        label_y = max(y + h - text_h - 8, text_h + 8)
        
        # 繪製背景
        cv2.rectangle(image, 
                     (label_x - 4, label_y - text_h - 4), 
                     (label_x + text_w + 4, label_y + 4),
                     self.colors['text_bg'], -1)
        
        # 繪製邊框
        cv2.rectangle(image, 
                     (label_x - 4, label_y - text_h - 4), 
                     (label_x + text_w + 4, label_y + 4),
                     self.colors['border_active'], 1)
        
        # 繪製文字
        cv2.putText(image, text, (label_x, label_y), font, font_scale, 
                   self.colors['text_fg'], thickness)
    
    def _draw_corner_markers(self, image: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
        """繪製角落標記。"""
        x, y, w, h = rect
        marker_size = 8
        thickness = 2
        color = self.colors['selected']
        
        # 左上角
        cv2.line(image, (x, y), (x + marker_size, y), color, thickness)
        cv2.line(image, (x, y), (x, y + marker_size), color, thickness)
        
        # 右上角
        cv2.line(image, (x + w, y), (x + w - marker_size, y), color, thickness)
        cv2.line(image, (x + w, y), (x + w, y + marker_size), color, thickness)
        
        # 左下角
        cv2.line(image, (x, y + h), (x + marker_size, y + h), color, thickness)
        cv2.line(image, (x, y + h), (x, y + h - marker_size), color, thickness)
        
        # 右下角
        cv2.line(image, (x + w, y + h), (x + w - marker_size, y + h), color, thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - marker_size), color, thickness)
    
    def _draw_roi_label(self, image: np.ndarray, rect: Tuple[int, int, int, int], 
                       name: str) -> None:
        """繪製 ROI 名稱標籤。"""
        x, y, w, h = rect
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # 獲取文字尺寸
        (text_w, text_h), baseline = cv2.getTextSize(name, font, font_scale, thickness)
        
        # 計算標籤位置（矩形左上角外側）
        label_x = x
        label_y = max(y - 8, text_h + 8)
        
        # 繪製背景
        cv2.rectangle(image, 
                     (label_x - 2, label_y - text_h - 2), 
                     (label_x + text_w + 2, label_y + 2),
                     self.colors['selected'], -1)
        
        # 繪製文字
        cv2.putText(image, name, (label_x, label_y), font, font_scale, 
                   (255, 255, 255), thickness)


class ROIFeedbackManager:
    """ROI 視覺回饋管理器，協調各種視覺效果。"""
    
    def __init__(self):
        self.renderer = ROIFeedbackRenderer()
        self.is_dragging = False
        self.drag_start = None
        self.drag_current = None
        self.selected_rois = []
        self.hover_roi = None
    
    def start_drag(self, start_pos: Tuple[int, int]) -> None:
        """開始拖拽操作。
        
        Args:
            start_pos: 拖拽起始位置 (x, y)
        """
        self.is_dragging = True
        self.drag_start = start_pos
        self.drag_current = start_pos
    
    def update_drag(self, current_pos: Tuple[int, int]) -> None:
        """更新拖拽位置。
        
        Args:
            current_pos: 當前滑鼠位置 (x, y)
        """
        if self.is_dragging:
            self.drag_current = current_pos
    
    def end_drag(self) -> Optional[Tuple[int, int, int, int]]:
        """結束拖拽操作。
        
        Returns:
            如果有效則返回矩形座標 (x, y, w, h)，否則返回 None
        """
        if not self.is_dragging or not self.drag_start or not self.drag_current:
            self._reset_drag()
            return None
            
        # 計算矩形
        x1, y1 = self.drag_start
        x2, y2 = self.drag_current
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        self._reset_drag()
        
        # 檢查最小尺寸
        if w < 10 or h < 10:
            return None
            
        return (x, y, w, h)
    
    def _reset_drag(self) -> None:
        """重置拖拽狀態。"""
        self.is_dragging = False
        self.drag_start = None
        self.drag_current = None
    
    def cancel_drag(self) -> None:
        """取消拖拽操作。"""
        self._reset_drag()
    
    def set_selected_rois(self, rois: List[Tuple[Tuple[int, int, int, int], str]]) -> None:
        """設定已選擇的 ROI 列表。
        
        Args:
            rois: ROI 列表，每個元素為 ((x, y, w, h), name)
        """
        self.selected_rois = rois.copy()
    
    def set_hover_roi(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """設定懸停的 ROI。
        
        Args:
            rect: 矩形座標 (x, y, w, h) 或 None
        """
        self.hover_roi = rect
    
    def get_current_drag_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """獲取當前拖拽矩形。
        
        Returns:
            如果正在拖拽則返回矩形座標 (x, y, w, h)，否則返回 None
        """
        if not self.is_dragging or not self.drag_start or not self.drag_current:
            return None
            
        x1, y1 = self.drag_start
        x2, y2 = self.drag_current
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        return (x, y, w, h)
    
    def render_feedback(self, image: np.ndarray) -> np.ndarray:
        """渲染所有視覺回饋效果。
        
        Args:
            image: 基礎影像
            
        Returns:
            包含所有視覺效果的影像
        """
        result = image.copy()
        
        # 繪製已選擇的 ROI
        if self.selected_rois:
            result = self.renderer.draw_selected_rois(result, self.selected_rois)
        
        # 繪製懸停效果
        if self.hover_roi:
            result = self.renderer.draw_hover_effect(result, self.hover_roi)
        
        # 繪製拖拽預覽
        drag_rect = self.get_current_drag_rect()
        if drag_rect:
            # 計算尺寸文字
            _, _, w, h = drag_rect
            size_text = f"{w} × {h} px"
            result = self.renderer.draw_preview_rect(result, drag_rect, size_text)
        
        return result
    
    def pixel_to_normalized(self, pixel_rect: Tuple[int, int, int, int], 
                           image_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """將像素座標轉換為標準化座標。
        
        Args:
            pixel_rect: 像素矩形 (x, y, w, h)
            image_shape: 影像尺寸 (height, width)
            
        Returns:
            標準化矩形 (nx, ny, nw, nh)
        """
        x, y, w, h = pixel_rect
        img_h, img_w = image_shape
        
        nx = x / img_w
        ny = y / img_h
        nw = w / img_w
        nh = h / img_h
        
        return (nx, ny, nw, nh)
    
    def normalized_to_pixel(self, norm_rect: Tuple[float, float, float, float], 
                           image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """將標準化座標轉換為像素座標。
        
        Args:
            norm_rect: 標準化矩形 (nx, ny, nw, nh)
            image_shape: 影像尺寸 (height, width)
            
        Returns:
            像素矩形 (x, y, w, h)
        """
        nx, ny, nw, nh = norm_rect
        img_h, img_w = image_shape
        
        x = int(nx * img_w)
        y = int(ny * img_h)
        w = int(nw * img_w)
        h = int(nh * img_h)
        
        return (x, y, w, h)


# 便捷函數
def create_roi_feedback_manager() -> ROIFeedbackManager:
    """建立 ROI 視覺回饋管理器。
    
    Returns:
        配置好的 ROI 視覺回饋管理器實例
    """
    return ROIFeedbackManager()


def apply_roi_feedback(image: np.ndarray, manager: ROIFeedbackManager) -> np.ndarray:
    """應用 ROI 視覺回饋效果到影像。
    
    Args:
        image: 基礎影像
        manager: ROI 視覺回饋管理器
        
    Returns:
        包含視覺回饋效果的影像
    """
    return manager.render_feedback(image)
