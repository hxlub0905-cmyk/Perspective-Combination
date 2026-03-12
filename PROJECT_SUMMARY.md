# 專案摘要（Perspective-Combination / Fusi³）

## 1) 專案定位
- 這是一個以 **PySide6** 建立的桌面應用程式，用於 **SEM 影像融合與缺陷分析**。
- 核心目標是對不同條件下的影像進行對齊、運算（相減/混合/反相）與 SNR 分析，協助觀察可能缺陷。

## 2) 程式進入點
- `main.py` 為啟動入口，負責：
  - 建立 Qt Application。
  - 顯示品牌化 Splash Screen。
  - 建立並顯示主視窗 `PerspectiveCombinationDialog`。

## 3) 核心模組
- `perscomb/core/perspective_combine.py`
  - 定義主要資料結構與流程（如 `OperationType`、`SinglePairResult`、`CombineResult`）。
  - 負責影像對齊後的運算邏輯、統計資訊與結果整合。
- `perscomb/core/ebeam_snr.py`
  - 提供對齊與雜訊/訊號品質評估相關方法（含 `AlignResult` 與穩健對齊流程）。
- `perscomb/core/roi_set.py`
  - 管理 ROI（Region of Interest）資料、統計與結果封裝，支援缺陷分析的區域化比較。

## 4) UI 模組
- `perscomb/ui/dialog.py`
  - 主對話框與互動流程所在；整合載入、參數設定、運算觸發、結果顯示與輸出。
- `perscomb/ui/design_tokens.py`
  - 集中管理視覺設計 Token（色彩、字體、間距、圓角、陰影、動畫等），維持 UI 風格一致性。

## 5) 依賴套件
- 主要依賴定義於 `requirements.txt`：
  - `PySide6`：GUI
  - `numpy`：數值運算
  - `opencv-python`：影像處理
  - `matplotlib`：繪圖
  - `python-pptx`：簡報輸出

## 6) 專案結構總覽
- `main.py`：應用程式啟動入口
- `perscomb/core/`：演算法與資料模型
- `perscomb/ui/`：介面與設計系統
- `requirements.txt`：相依套件版本

## 7) 總結
- 本專案是一個「**影像對齊 + 影像運算 + SNR/ROI 分析 + 視覺化 UI**」的完整桌面工具。
- 架構上採用 **UI 與核心邏輯分層**，有助於後續維護與功能擴充。
