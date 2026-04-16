"""Live preview worker for PerspectiveCombinationDialog.

Architecture
------------
AlignmentCache
    Stores (aligned_compare, AlignResult) per pair+method key.
    After a full Compute the cache is populated from results so subsequent
    live-preview calls (Tier-2, non-alignment params) skip re-alignment and
    run in ~35-120 ms.

LivePreviewManager  (QObject, lives on main thread)
    Owns a single-shot QTimer for debouncing.
    Tier-1 change (alignment params): 600 ms debounce, cache invalidated.
    Tier-2 change (other params):     150 ms debounce, uses cached alignment.
    Spawns a background QThread per preview; cancels in-flight thread if a
    newer schedule arrives.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal

from ..core.perspective_combine import (
    compute_single_pair,
    compute_multi_synthesis,
    SinglePairResult,
    SynthesisResult,
)


# ---------------------------------------------------------------------------
# Alignment cache
# ---------------------------------------------------------------------------

class AlignmentCache:
    """Cache for (aligned_compare, AlignResult) keyed by pair + method.

    Key = (base_label, compare_label, align_method, search_radius)
    """

    def __init__(self) -> None:
        self._cache: Dict[tuple, Tuple[np.ndarray, Any]] = {}

    def make_key(
        self,
        base_label: str,
        compare_label: str,
        align_method: str,
        search_radius: int,
    ) -> tuple:
        return (base_label, compare_label, align_method, search_radius)

    def store(self, key: tuple, aligned_compare: np.ndarray, alignment: Any) -> None:
        self._cache[key] = (aligned_compare.copy(), alignment)

    def get(self, key: tuple) -> Optional[Tuple[np.ndarray, Any]]:
        return self._cache.get(key)

    def clear(self) -> None:
        self._cache.clear()

    def invalidate_pair(self, base_label: str, compare_label: str) -> None:
        """Remove all entries for a specific pair regardless of method."""
        keys_to_del = [
            k for k in self._cache
            if k[0] == base_label and k[1] == compare_label
        ]
        for k in keys_to_del:
            del self._cache[k]


# ---------------------------------------------------------------------------
# Background worker QObject
# ---------------------------------------------------------------------------

class _PreviewWorker(QtCore.QObject):
    finished = Signal(object)   # SinglePairResult
    error = Signal(str)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn
        self.abort_requested: bool = False

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:                        # noqa: BLE001
            self.error.emit(str(exc))
            return
        if not self.abort_requested:
            self.finished.emit(result)


# ---------------------------------------------------------------------------
# LivePreviewManager
# ---------------------------------------------------------------------------

class LivePreviewManager(QtCore.QObject):
    """Debounced live-preview orchestrator (main-thread QObject).

    Supports both Pair mode (compute_single_pair) and Synthesis mode
    (compute_multi_synthesis).  The mode is determined by the 'mode' key in
    the params dict: 'pair' (default) or 'synthesis'.

    Signals
    -------
    preview_ready(object)   — emitted with SinglePairResult or SynthesisResult.
    preview_error(str)      — emitted on computation error.
    preview_started()       — emitted just before thread starts.
    """

    preview_ready = Signal(object)   # SinglePairResult | SynthesisResult
    preview_error = Signal(str)
    preview_started = Signal()

    TIER1_DEBOUNCE_MS = 600   # alignment params changed
    TIER2_DEBOUNCE_MS = 150   # non-alignment params changed

    def __init__(self, alignment_cache: AlignmentCache, parent=None) -> None:
        super().__init__(parent)
        self._cache = alignment_cache
        self._pending_params: Optional[dict] = None
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_PreviewWorker] = None

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fire)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, params: dict, is_align_change: bool = False) -> None:
        """Schedule a preview.  Replaces any pending (not-yet-fired) schedule.

        Parameters
        ----------
        params:
            Dict produced by ``PerspectiveCombinationDialog._collect_preview_params()``.
        is_align_change:
            True when alignment-related parameters (method, search radius) changed.
            Uses longer debounce and already-cleared cache.
        """
        self._pending_params = params.copy()
        delay = self.TIER1_DEBOUNCE_MS if is_align_change else self.TIER2_DEBOUNCE_MS
        self._timer.start(delay)

    def cancel(self) -> None:
        """Cancel pending timer and signal in-flight worker to discard result."""
        self._timer.stop()
        if self._worker is not None:
            self._worker.abort_requested = True

    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(self) -> None:
        """Timer callback — kick off background computation."""
        if self._pending_params is None:
            return

        # If a thread is still running, abort it and re-schedule briefly.
        if self.is_busy():
            if self._worker is not None:
                self._worker.abort_requested = True
            self._timer.start(80)
            return

        params = self._pending_params
        self._pending_params = None

        mode = params.get("mode", "pair")

        if mode == "synthesis":
            _run = self._build_synthesis_run(params)
        else:
            _run = self._build_pair_run(params)

        # Spin up background thread
        thread = QtCore.QThread()
        worker = _PreviewWorker(_run)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(lambda _: thread.quit())
        worker.error.connect(lambda _: thread.quit())
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_done)

        self._thread = thread
        self._worker = worker
        thread.start()
        self.preview_started.emit()

    def _build_pair_run(self, params: dict):
        """Build the _run closure for standard Pair mode."""
        cache_key = self._cache.make_key(
            params["base_label"],
            params["compare_label"],
            params["align_method"],
            params["search_radius"],
        )
        cached = self._cache.get(cache_key)
        pre_aligned = cached[0] if cached is not None else None
        pre_alignment = cached[1] if cached is not None else None
        cache = self._cache

        def _run() -> SinglePairResult:
            result = compute_single_pair(
                base=params["base_img"],
                compare=params["compare_img"],
                base_label=params["base_label"],
                compare_label=params["compare_label"],
                operation=params["operation"],
                alpha=params["alpha"],
                beta=params["beta"],
                invert_base=params["invert_base"],
                invert_compare=params["invert_compare"],
                invert_result=params["invert_result"],
                normalize=params["normalize"],
                normalize_method=params["normalize_method"],
                glv_range=params.get("glv_range"),
                clahe_clip_limit=params.get("clahe_clip_limit", 2.0),
                search_radius=params["search_radius"],
                alignment_method=params["align_method"],
                preserve_positive_diff=params["preserve_positive_diff"],
                abs_no_gain=params["abs_no_gain"],
                snr_window_size=params.get("snr_window_size", 31),
                roi_rect=params.get("roi_rect"),
                roi_set=params.get("roi_set"),
                roi_match=params.get("roi_match", False),
                skip_snr_map=True,
                pre_aligned_compare=pre_aligned,
                pre_alignment_result=pre_alignment,
            )
            if pre_aligned is None and result.aligned_compare is not None:
                cache.store(cache_key, result.aligned_compare, result.alignment)
            return result

        return _run

    def _build_synthesis_run(self, params: dict):
        """Build the _run closure for Multi-Image Synthesis mode."""
        def _run() -> SynthesisResult:
            return compute_multi_synthesis(
                role_images=params["role_images"],
                algorithm=params["algorithm"],
                weights=params.get("weights"),
                shading_output=params.get("shading_output", "topo"),
                light_angle_deg=params.get("light_angle_deg", 135.0),
                normalize_method=params.get("normalize_method", "percentile"),
                glv_range=params.get("glv_range"),
                invert_result=params.get("invert_result", False),
                align_to_bse=params.get("align_to_bse", False),
                alignment_method=params.get("align_method", "phase"),
                snr_window_size=params.get("snr_window_size", 31),
                skip_snr_map=True,
            )

        return _run

    def _on_worker_finished(self, result: SinglePairResult) -> None:
        self.preview_ready.emit(result)

    def _on_worker_error(self, msg: str) -> None:
        self.preview_error.emit(msg)

    def _on_thread_done(self) -> None:
        self._thread = None
        self._worker = None
        # Fire immediately if a new schedule arrived while we were running
        if self._pending_params is not None:
            self._timer.start(50)
