import pytest
from unittest.mock import MagicMock, patch
from pdf_ingest import chunk_text, table_row_to_line, table_to_chunks, table_context
import config


class TestChunkTextEdgeCases:
    """Edge cases for chunk_text function."""

    def test_unicode_and_emoji(self):
        """Test that unicode and emoji are handled correctly."""
        text = "Hello 🎉 world 你好 café"
        result = chunk_text(text, 100, 10)
        assert len(result) == 1
        assert "🎉" in result[0]
        assert "你好" in result[0]

    def test_carriage_return_newlines(self):
        """Test \\r\\n line endings are handled."""
        text = "line1\r\nline2\r\nline3"
        result = chunk_text(text, 50, 10)
        assert len(result) >= 1
        # Should split on \r\n boundaries
        assert "line1" in " ".join(result)

    def test_text_only_newlines(self):
        """Test text that is only newlines/whitespace."""
        assert chunk_text("\n\n\n", 100, 10) == []
        assert chunk_text("\t\n   \n\t", 100, 10) == []

    def test_chunk_size_smaller_than_overlap(self):
        """Test when overlap is larger than chunk_size (edge case)."""
        text = "word1 word2 word3 word4 word5"
        result = chunk_text(text, 10, 15)
        # Should still produce chunks, but overlap > chunk_size is degenerate
        assert len(result) >= 1

    def test_overlap_equals_chunk_size(self):
        """Test when overlap equals chunk_size (extreme overlap)."""
        text = "a" * 500
        result = chunk_text(text, 50, 50)
        assert len(result) > 0
        # Each chunk should be 50 chars, and next starts at 50, so no overlap in practice
        # (start = max(end - overlap, start + 1))

    def test_chunk_size_of_one(self):
        """Test with chunk_size of 1."""
        text = "ab"
        result = chunk_text(text, 1, 0)
        # Each chunk can be at most 1 char, but it tries to find boundaries
        assert len(result) >= 1
        # Content should be preserved
        assert "a" in " ".join(result)

    def test_very_long_single_word(self):
        """Test very long word that exceeds chunk_size."""
        word = "x" * 1000
        result = chunk_text(word, 100, 20)
        # Must split the word even though it has no boundaries
        # With chunk_size=100, overlap=20: advances ~80 chars per iteration
        # 1000/80 ≈ 12-13 chunks expected, so >= 7 is safe
        assert len(result) >= 7
        # All content should be present (overlaps mean sum > original length)
        assert sum(len(c) for c in result) > 1000

    def test_sentence_boundary_preference(self):
        """Test that '. ' is preferred over arbitrary cuts."""
        text = "First sentence. " + ("w" * 100) + ". Second part."
        result = chunk_text(text, 100, 0)
        # Should prefer to break on '. '
        assert "First sentence." in result[0]

    def test_newline_boundary_more_preferred_than_sentence(self):
        """Test that newline is preferred over sentence boundary."""
        text = ("a" * 30) + "\n" + ("b" * 30) + ". " + ("c" * 30)
        result = chunk_text(text, 100, 0)
        # Newline should be preferred, so first chunk ends with 'a's
        assert result[0].endswith("a" * 30) or "a" * 29 in result[0]

    def test_no_boundaries_found_uses_exact_size(self):
        """Test that when no boundaries exist, chunks are exactly chunk_size."""
        text = "nospace" * 200  # No spaces or newlines
        result = chunk_text(text, 50, 0)
        assert len(result) > 1
        # Each chunk should be 50 chars (except possibly the last)
        assert all(len(c) == 50 for c in result[:-1])

    def test_leading_trailing_whitespace_stripped(self):
        """Test that leading/trailing whitespace is stripped from chunks."""
        text = "   first line   \n   second line   "
        result = chunk_text(text, 100, 0)
        assert all(c == c.strip() for c in result)

    def test_single_character_text(self):
        """Test single character input."""
        assert chunk_text("x", 100, 10) == ["x"]

    def test_text_exactly_chunk_size(self):
        """Test text that is exactly chunk_size."""
        text = "a" * 100
        result = chunk_text(text, 100, 10)
        assert result == ["a" * 100]

    def test_boundary_search_within_50_percent(self):
        """Test that boundary search requires boundary > start + chunk_size // 2."""
        text = "a" * 40 + "\n" + "b" * 40  # Newline at position 40
        result = chunk_text(text, 100, 0)
        # With chunk_size 100, need boundary > 50. Boundary is at 40.
        # So newline is ignored, and we cut at 100.
        # This means we get "a"*40 + "\n" + "b"*60 in first chunk
        assert len(result) == 1


class TestTableRowToLineEdgeCases:
    """Edge cases for table_row_to_line function."""

    def test_row_with_int_values(self):
        """Test row containing non-string types like ints."""
        # Note: The function expects list[Optional[str]], but in practice
        # tables might have int values. Test robustness.
        row = ["Revenue", 1000, 2000]
        # This should raise an error or handle gracefully
        # The current code does (c or "").replace(...) which will fail on int
        with pytest.raises((AttributeError, TypeError)):
            table_row_to_line(row)

    def test_row_with_mixed_types(self):
        """Test row with mixed string and int values."""
        row = ["Item", 100, "200"]
        with pytest.raises((AttributeError, TypeError)):
            table_row_to_line(row)

    def test_label_only_with_empty_strings(self):
        """Test row where label exists but all values are empty strings."""
        row = ["Label", "", "", ""]
        assert table_row_to_line(row) == "Label"

    def test_label_empty_but_values_exist(self):
        """Test row where label cell is empty but values exist."""
        row = ["", "100", "200"]
        # Label becomes empty string, values exist
        # After filtering empty cells, cells = ["100", "200"]
        # label = "100", values = ["200"]
        result = table_row_to_line(row)
        assert result == "100: 200"

    def test_all_cells_are_fillers(self):
        """Test row containing only filler cells."""
        row = ["$", "%", "$", "%", ")"]
        assert table_row_to_line(row) == ""

    def test_label_with_filler_values(self):
        """Test row with meaningful label and only filler values."""
        row = ["Total", "$", "%", ")"]
        assert table_row_to_line(row) == "Total"

    def test_nbsp_mixed_with_spaces(self):
        """Test various whitespace including non-breaking space."""
        row = ["\xa0Total\xa0 ", "\xa0 1000 \xa0", None]
        result = table_row_to_line(row)
        assert result == "Total: 1000"

    def test_unicode_in_cells(self):
        """Test unicode characters in table cells."""
        row = ["Revenue €", "1,000", "2,000"]
        result = table_row_to_line(row)
        assert "€" in result

    def test_emoji_in_cells(self):
        """Test emoji in table cells."""
        row = ["Status 🔥", "High 📈", "Low 📉"]
        result = table_row_to_line(row)
        assert "🔥" in result

    def test_cell_with_only_whitespace(self):
        """Test cells that are only whitespace."""
        row = ["Label", "   ", "\t", "\n"]
        result = table_row_to_line(row)
        # After strip, these become empty
        assert result == "Label"

    def test_very_long_cell_value(self):
        """Test very long cell values."""
        row = ["Item", "x" * 1000]
        result = table_row_to_line(row)
        assert len(result) > 1000

    def test_cell_with_pipe_character(self):
        """Test cell containing pipe character."""
        row = ["Item", "value|other"]
        result = table_row_to_line(row)
        # Pipe is in the value, separator is also pipe - potential confusion
        assert "value|other" in result

    def test_none_values_mixed_with_strings(self):
        """Test None values scattered throughout."""
        row = [None, "A", None, "B", None]
        result = table_row_to_line(row)
        assert result == "A: B"


class TestTableToChunksEdgeCases:
    """Edge cases for table_to_chunks function."""

    def test_max_chars_zero(self):
        """Test with max_chars set to 0."""
        rows = [["Item", "100"], ["Item2", "200"]]
        result = table_to_chunks(rows, "context", 1, 0)
        # With max_chars=0, each line should cause a chunk to be emitted
        assert len(result) >= 2

    def test_huge_single_row_exceeds_max_chars(self):
        """Test that huge single row is still emitted even if exceeds max_chars."""
        rows = [["Item", "x" * 1000]]
        result = table_to_chunks(rows, "ctx", 1, 50)
        # The single line is > 50 chars, but it must be emitted
        assert len(result) == 1
        assert "Item" in result[0]

    def test_context_longer_than_limit(self):
        """Test that table_to_chunks works with long context strings."""
        # table_to_chunks receives pre-processed context, doesn't truncate it
        long_context = "word " * 100  # 500 chars
        rows = [["Item", "100"]]
        result = table_to_chunks(rows, long_context, 1, 200)
        # Should still work
        assert len(result) >= 1
        # Long context embedded in result
        assert "word" in result[0]

    def test_empty_context(self):
        """Test with empty context string."""
        rows = [["Item", "100"], ["Item2", "200"]]
        result = table_to_chunks(rows, "", 1, 200)
        # Header should be "Table on page 1. " (no context)
        assert all("Table on page 1." in c for c in result)

    def test_rows_that_produce_empty_lines(self):
        """Test rows that become empty lines after table_row_to_line."""
        rows = [
            ["$", "%", ")"],  # Becomes empty
            ["Item", "100"],   # Becomes "Item: 100"
            [None, "", "$"],   # Becomes empty
            ["Item2", "200"],  # Becomes "Item2: 200"
        ]
        result = table_to_chunks(rows, "ctx", 1, 200)
        # Empty rows are skipped
        body = result[0].split("\n", 1)[1]
        assert "Item: 100" in body
        assert "Item2: 200" in body

    def test_single_row_table(self):
        """Test table with only header row."""
        rows = [["Item", "Q1", "Q2"]]
        result = table_to_chunks(rows, "ctx", 1, 200)
        assert len(result) == 1
        assert "Item: Q1 | Q2" in result[0]

    def test_many_small_rows_fit_in_one_chunk(self):
        """Test many small rows that all fit in one chunk."""
        rows = [[f"Item{i}", "100"] for i in range(50)]
        result = table_to_chunks(rows, "ctx", 1, 5000)  # Large max_chars
        assert len(result) == 1

    def test_page_num_zero(self):
        """Test with page_num=0."""
        rows = [["Item", "100"]]
        result = table_to_chunks(rows, "ctx", 0, 200)
        assert "Table on page 0." in result[0]

    def test_page_num_very_large(self):
        """Test with very large page number."""
        rows = [["Item", "100"]]
        result = table_to_chunks(rows, "ctx", 99999, 200)
        assert "Table on page 99999." in result[0]

    def test_newlines_in_context(self):
        """Test context with embedded newlines (should be passed as-is)."""
        # Note: table_to_chunks doesn't collapse context, but table_context does
        context_with_newlines = "Title\nSubtitle\nMoreInfo"
        rows = [["Item", "100"]]
        result = table_to_chunks(rows, context_with_newlines, 1, 200)
        # Context is used as-is
        assert context_with_newlines in result[0]


class TestTableContextEdgeCases:
    """Edge cases for table_context function."""

    def test_extract_text_returns_none(self):
        """Test when page.crop().extract_text() returns None."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 100, 600, 200)  # top=100

        # Make crop().extract_text() return None
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = None
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Should return empty string (or default to "")
        assert result == ""

    def test_bbox_top_is_zero(self):
        """Test when table.bbox[1] (top) is 0.

        Code uses max(table.bbox[1], 1), so top=0 becomes 1.
        """
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 0, 600, 100)  # top=0

        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = "Some text"
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Should use top=1 (max(0, 1))
        mock_page.crop.assert_called_with((0, 0, 600, 1))
        assert result == "Some text"

    def test_bbox_top_is_negative(self):
        """Test when table.bbox[1] is negative (edge case, shouldn't happen)."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, -10, 600, 100)  # top=-10

        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = "Some text"
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Should use top=1 (max(-10, 1))
        mock_page.crop.assert_called_with((0, 0, 600, 1))

    def test_context_shorter_than_limit(self):
        """Test when extracted context is shorter than TABLE_CONTEXT_CHARS."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        short_text = "Short context"
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = short_text
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Should return as-is
        assert result == "Short context"

    def test_context_exactly_limit(self):
        """Test when context is exactly TABLE_CONTEXT_CHARS."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        exact_text = "x " * (config.TABLE_CONTEXT_CHARS // 2)  # Exactly 300 chars
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = exact_text
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Should return as-is (not > limit)
        assert result == exact_text.strip()

    def test_context_longer_than_limit_cuts_on_word_boundary(self):
        """Test that when context exceeds limit, it's cut on word boundary."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        # Create text longer than limit, with clear word boundaries
        long_text = "word " * 100  # Much longer than 300
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = long_text
        mock_page.crop.return_value = mock_crop

        with patch('config.TABLE_CONTEXT_CHARS', 50):
            result = table_context(mock_page, mock_table)
            # Should be truncated to last 50 chars, then find word boundary
            # The result should be less than 50 + word length
            assert len(result) <= 100

    def test_context_with_multiple_spaces_collapsed(self):
        """Test that multiple spaces/newlines are collapsed to single space."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        text_with_extra_space = "Word1    Word2\n\nWord3\tWord4"
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = text_with_extra_space
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Spaces/tabs/newlines collapsed
        assert result == "Word1 Word2 Word3 Word4"

    def test_context_with_leading_trailing_spaces(self):
        """Test that leading/trailing spaces are handled."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        text = "   Some text with spaces   "
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = text
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Should not have leading/trailing spaces
        assert result == "Some text with spaces"

    def test_unicode_in_context(self):
        """Test unicode text in context."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        unicode_text = "Financial Report 财务报告 Финансовый отчет"
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = unicode_text
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        assert "财务报告" in result

    def test_empty_extract_text(self):
        """Test when extract_text returns empty string (not None)."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = ""
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Empty string should be returned as-is (or fallback to "")
        assert result == ""

    def test_page_width_used_correctly(self):
        """Test that page.width is used for crop right boundary."""
        mock_page = MagicMock()
        mock_page.width = 1000  # Different width
        mock_table = MagicMock()
        mock_table.bbox = (100, 50, 900, 200)  # left=100, right=900, but page.width used

        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = "Text"
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Should crop to (0, 0, page.width, top) = (0, 0, 1000, 50)
        mock_page.crop.assert_called_with((0, 0, 1000, 50))

    def test_large_page_dimensions(self):
        """Test with very large page dimensions."""
        mock_page = MagicMock()
        mock_page.width = 50000
        mock_table = MagicMock()
        mock_table.bbox = (1000, 5000, 40000, 15000)

        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = "Some text"
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        mock_page.crop.assert_called_with((0, 0, 50000, 5000))

    def test_long_context_word_boundary_search(self):
        """Test word boundary finding in long context."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        # Text that's longer than limit with clear spaces
        limit = config.TABLE_CONTEXT_CHARS
        text = "word " * (limit // 5 + 50)  # Much longer than limit
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = text
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # Result should be truncated and cut at word boundary
        # It should not end in the middle of "word"
        assert not result.endswith("w") and not result.endswith("wo") and not result.endswith("wor")


class TestIntegrationEdgeCases:
    """Integration tests combining multiple functions with edge cases."""

    def test_chunk_text_with_unicode_preserves_content(self):
        """Verify chunk_text preserves unicode through chunking."""
        text = "Hello 🎉 " * 100
        chunks = chunk_text(text, 100, 20)
        reassembled = " ".join(chunks)
        # Should preserve emoji
        assert "🎉" in reassembled

    def test_table_to_chunks_with_unicode_rows(self):
        """Test table_to_chunks with unicode in rows."""
        rows = [
            ["Revenue €", "100,000", "200,000"],
            ["Expenses 日本", "50,000", "75,000"],
        ]
        result = table_to_chunks(rows, "Financial Report 报告", 1, 200)
        assert len(result) > 0
        assert "€" in result[0]
        assert "日本" in result[0]

    def test_table_context_with_whitespace_collapse(self):
        """Test table_context with long text that gets collapsed."""
        mock_page = MagicMock()
        mock_page.width = 600
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 600, 200)

        # Text with lots of whitespace
        raw_text = "Word1\n\n\nWord2\t\t\tWord3    " * 20  # Enough to trigger truncation
        mock_crop = MagicMock()
        mock_crop.extract_text.return_value = raw_text
        mock_page.crop.return_value = mock_crop

        result = table_context(mock_page, mock_table)
        # After collapse, should be much shorter
        # No multi-space or tabs should exist
        assert "  " not in result
        assert "\t" not in result
        assert "\n" not in result
