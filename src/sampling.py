"""Deterministic stratified page sampling for the triage layer."""

from __future__ import annotations


class SmartSampler:
    """Selects representative page indices for analysis.

    Fixed strategy — no tuneable parameters — to guarantee
    deterministic, stable behaviour.
    """

    _FIXED_MAX = 5

    def sample_indices(self, page_count: int) -> list[int]:
        """Return sorted, deduplicated page indices to sample.

        * If *page_count* ≤ 5 → all pages.
        * Otherwise → ``[0, 1, page_count // 2, page_count - 2, page_count - 1]``.
        """
        if page_count <= 0:
            return []

        if page_count <= self._FIXED_MAX:
            return list(range(page_count))

        raw = [0, 1, page_count // 2, page_count - 2, page_count - 1]

        # Deduplicate, bounds-check, sort
        return sorted({i for i in raw if 0 <= i < page_count})
