"""
PixelOpt UI Design Tokens
統一的設計系統常量定義

作者：Claude AI
日期：2026-01-23
版本：1.0 - 方案 A：優化橙色系
"""

# ============================================================================
# 配色系統 (Color Palette)
# ============================================================================

class Colors:
    """顏色常量"""

    # 品牌色系 (Brand Colors)
    BRAND_PRIMARY = "#F59E0B"
    BRAND_PRIMARY_HOVER = "#FBBF24"
    BRAND_PRIMARY_PRESSED = "#D97706"
    BRAND_PRIMARY_SOFT = "#FEF3C7"
    BRAND_PRIMARY_GLOW = "#F59E0B44"

    # 語義色系 (Semantic Colors)
    SUCCESS = "#16A34A"
    WARNING = "#DC2626"
    INFO = "#2563EB"

    # 背景色系 (Background Colors)
    BG_WINDOW = "#F5F6F8"
    BG_PANEL = "#FFFFFF"
    BG_CARD = "#FBFCFE"
    BG_SUBTLE = "#F3F4F6"
    BG_ALT = BG_SUBTLE
    BG_HEADER = "#FFFFFF"
    BG_INPUT = "#FFFFFF"
    BG_VIEWER = "#000000"

    # 文字色系 (Text Colors)
    TEXT_PRIMARY = "#1F2937"
    TEXT_SECONDARY = "#4B5563"
    TEXT_MUTED = "#9CA3AF"
    TEXT_INVERSE = "#111827"
    TEXT_ON_PRIMARY = "#FFFFFF"

    # 邊框色系 (Border Colors)
    BORDER_DEFAULT = "#E5E7EB"
    BORDER_HOVER = "#F6AD2B"
    BORDER_ACTIVE = "#F59E0B"
    BORDER_FOCUS = "#F59E0B"

    # 圖表色系 (Chart Colors)
    CHART_PRIMARY = "#F59E0B"
    CHART_SECONDARY = "#16A34A"
    CHART_TERTIARY = "#4B5563"
    CHART_QUATERNARY = "#2563EB"

    # 漸變色 (Gradients)
    GRADIENT_CARD_START = "#FFFFFF"
    GRADIENT_CARD_MID = "#FBFCFE"
    GRADIENT_CARD_END = "#F3F4F6"

    GRADIENT_BG_START = "#F8FAFC"
    GRADIENT_BG_MID = "#F5F6F8"
    GRADIENT_BG_END = "#EEF2F7"

    # 半透明色 (Transparent Colors)
    OVERLAY_DARK = "rgba(17, 24, 39, 0.55)"
    OVERLAY_LIGHT = "rgba(255, 255, 255, 0.75)"
    SELECTION_BG = "rgba(245, 158, 11, 0.2)"
    SLIDER_FILLED = "rgba(245, 158, 11, 0.35)"


# ============================================================================
# 字體系統 (Typography)
# ============================================================================

class Typography:
    """字體常量"""

    # 字體家族：Liberation Sans 在 Linux 預裝且與 Arial 度量相同；
    # Windows/macOS 優先使用 Arial / Helvetica Neue
    FONT_FAMILY = "'Liberation Sans', 'Arial', 'Helvetica Neue', 'Segoe UI', sans-serif"
    FONT_FAMILY_MONO = "'Liberation Mono', 'Consolas', 'Menlo', 'Courier New', monospace"

    # 字體大小（統一使用 px）
    FONT_SIZE_HERO = "52px"        # 超大標題（Welcome screen）
    FONT_SIZE_H1 = "22px"          # 大標題（Feature cards）
    FONT_SIZE_H2 = "18px"          # 二級標題
    FONT_SIZE_H3 = "14px"          # 三級標題（Section headers）
    FONT_SIZE_BODY = "13px"        # 正文
    FONT_SIZE_SMALL = "12px"       # 小字
    FONT_SIZE_CAPTION = "11px"     # 說明文字

    # 字重
    FONT_WEIGHT_REGULAR = "400"
    FONT_WEIGHT_MEDIUM = "500"
    FONT_WEIGHT_SEMIBOLD = "600"
    FONT_WEIGHT_BOLD = "700"

    # 行高
    LINE_HEIGHT_TIGHT = "1.2"
    LINE_HEIGHT_NORMAL = "1.5"
    LINE_HEIGHT_RELAXED = "1.8"

    # 字母間距
    LETTER_SPACING_TIGHT = "-0.02em"
    LETTER_SPACING_NORMAL = "0.01em"
    LETTER_SPACING_WIDE = "0.05em"


# ============================================================================
# 間距系統 (Spacing)
# ============================================================================

class Spacing:
    """間距常量（基於 4px 網格系統）"""

    XS = "4px"
    SM = "8px"
    MD = "12px"
    LG = "16px"
    XL = "24px"
    XXL = "32px"
    XXXL = "48px"

    # 常用組合
    BUTTON_PADDING = "8px 16px"
    GROUPBOX_PADDING = "16px 12px 12px 12px"
    CARD_PADDING = "16px"
    INPUT_PADDING = "7px 12px"
    DIALOG_PADDING = "20px"


# ============================================================================
# 圓角系統 (Border Radius)
# ============================================================================

class BorderRadius:
    """圓角半徑常量"""

    NONE = "0px"
    SM = "6px"     # 小圓角（Input, Tooltip）
    MD = "10px"    # 中圓角（GroupBox, Button）
    LG = "14px"    # 大圓角
    XL = "18px"    # 超大圓角（Feature cards）
    ROUND = "50%"  # 圓形


# ============================================================================
# 尺寸系統 (Sizing)
# ============================================================================

class Sizing:
    """元件尺寸常量"""

    # 按鈕
    BUTTON_HEIGHT_SM = "26px"
    BUTTON_HEIGHT_MD = "32px"
    BUTTON_HEIGHT_LG = "40px"
    BUTTON_MIN_WIDTH = "80px"

    # 輸入框
    INPUT_HEIGHT = "32px"
    INPUT_MIN_WIDTH = "120px"

    # 圖標
    ICON_SM = "16px"
    ICON_MD = "24px"
    ICON_LG = "32px"
    ICON_XL = "48px"

    # 滑塊
    SLIDER_TRACK_HEIGHT = "6px"
    SLIDER_HANDLE_WIDTH = "18px"
    SLIDER_HANDLE_HEIGHT = "18px"


# ============================================================================
# 陰影系統 (Shadows)
# ============================================================================

class Shadows:
    """陰影效果常量"""

    NONE = "none"
    SM = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    DEFAULT = "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
    MD = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    LG = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
    XL = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
    INNER = "inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)"
    GLOW_PRIMARY = f"0 0 20px {Colors.BRAND_PRIMARY_GLOW}"
    GLOW_SUCCESS = "0 0 20px rgba(14, 165, 164, 0.28)"


# ============================================================================
# 動畫系統 (Animations)
# ============================================================================

class Animations:
    """動畫時間常量"""

    DURATION_FAST = "150ms"
    DURATION_NORMAL = "250ms"
    DURATION_SLOW = "350ms"

    EASING_DEFAULT = "cubic-bezier(0.4, 0, 0.2, 1)"
    EASING_IN = "cubic-bezier(0.4, 0, 1, 1)"
    EASING_OUT = "cubic-bezier(0, 0, 0.2, 1)"
    EASING_IN_OUT = "cubic-bezier(0.4, 0, 0.2, 1)"


# ============================================================================
# Z-Index 層級 (Z-Index Layers)
# ============================================================================

class ZIndex:
    """元件層級常量"""

    BASE = 0
    DROPDOWN = 1000
    STICKY = 1100
    FIXED = 1200
    MODAL_BACKDROP = 1300
    MODAL = 1400
    POPOVER = 1500
    TOOLTIP = 1600


# ============================================================================
# 便捷方法 (Helper Methods)
# ============================================================================

def get_color_with_opacity(color: str, opacity: float) -> str:
    """
    將十六進制顏色轉換為帶透明度的 rgba 格式

    Args:
        color: 十六進制顏色碼（如 #F59E0B）
        opacity: 透明度（0.0 - 1.0）

    Returns:
        rgba 格式的顏色字符串
    """
    # 移除 # 符號
    color = color.lstrip('#')

    # 轉換為 RGB
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    return f"rgba({r}, {g}, {b}, {opacity})"


def create_gradient(start_color: str, end_color: str, angle: int = 180) -> str:
    """
    創建線性漸變

    Args:
        start_color: 起始顏色
        end_color: 結束顏色
        angle: 漸變角度（默認 180 度，從上到下）

    Returns:
        CSS 線性漸變字符串
    """
    return f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {start_color}, stop:1 {end_color})"


# ============================================================================
# 導出所有常量（方便導入）
# ============================================================================

__all__ = [
    'Colors',
    'Typography',
    'Spacing',
    'BorderRadius',
    'Sizing',
    'Shadows',
    'Animations',
    'ZIndex',
    'get_color_with_opacity',
    'create_gradient',
]
