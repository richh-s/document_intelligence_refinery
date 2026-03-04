"""Unit tests for SmartSampler."""

from document_intelligence_refinery.sampling import SmartSampler


class TestSmartSampler:
    def setup_method(self) -> None:
        self.sampler = SmartSampler()

    def test_zero_pages(self) -> None:
        assert self.sampler.sample_indices(0) == []

    def test_one_page(self) -> None:
        assert self.sampler.sample_indices(1) == [0]

    def test_three_pages(self) -> None:
        assert self.sampler.sample_indices(3) == [0, 1, 2]

    def test_five_pages(self) -> None:
        assert self.sampler.sample_indices(5) == [0, 1, 2, 3, 4]

    def test_six_pages(self) -> None:
        result = self.sampler.sample_indices(6)
        assert result == sorted(set(result))  # sorted and unique
        assert all(0 <= i < 6 for i in result)
        assert 0 in result  # first page
        assert 5 in result  # last page

    def test_100_pages(self) -> None:
        result = self.sampler.sample_indices(100)
        assert result == [0, 1, 50, 98, 99]

    def test_deduplication(self) -> None:
        """With 6 pages: [0,1,3,4,5] — indices overlap near boundaries."""
        result = self.sampler.sample_indices(6)
        assert len(result) == len(set(result))

    def test_determinism(self) -> None:
        """Same input must always produce same output."""
        a = self.sampler.sample_indices(42)
        b = self.sampler.sample_indices(42)
        assert a == b
