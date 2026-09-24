import pytest

from pdf_ingest import chunk_text, table_row_to_line, table_to_chunks


class TestChunkText:
    def test_empty_and_whitespace(self):
        assert chunk_text("", 100, 10) == []
        assert chunk_text("   \n  ", 100, 10) == []

    def test_short_text_is_single_chunk(self):
        assert chunk_text("hello world", 100, 10) == ["hello world"]

    def test_chunks_never_exceed_size(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = chunk_text(text, 100, 20)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_all_content_is_covered(self):
        words = [f"w{i}" for i in range(300)]
        chunks = chunk_text(" ".join(words), 120, 20)
        seen = set(" ".join(chunks).split())
        assert set(words) <= seen

    def test_consecutive_chunks_overlap(self):
        text = "\n".join(f"line number {i} with some padding text" for i in range(60))
        chunks = chunk_text(text, 200, 50)
        assert len(chunks) > 2
        for a, b in zip(chunks, chunks[1:]):
            assert b.split("\n")[0] in a or a.split("\n")[-1] in b

    def test_prefers_newline_boundary(self):
        text = ("a" * 70) + "\n" + ("b" * 70)
        chunks = chunk_text(text, 100, 0)
        assert chunks[0] == "a" * 70

    def test_terminates_on_unbroken_text(self):
        chunks = chunk_text("x" * 1000, 100, 30)
        assert len(chunks) >= 10

    @pytest.mark.parametrize("overlap", [0, 10, 99])
    def test_terminates_for_any_overlap(self, overlap):
        assert chunk_text("word " * 200, 100, overlap)


class TestTableRowToLine:
    def test_label_and_values(self):
        assert table_row_to_line(["Total net sales", "82,959", "81,434"]) == "Total net sales: 82,959 | 81,434"

    def test_drops_filler_and_empty_cells(self):
        assert table_row_to_line(["Products", "$", "63,355", None, "$", "63,948", ""]) == "Products: 63,355 | 63,948"

    def test_none_and_nbsp_handled(self):
        assert table_row_to_line(["Net income\xa0", None, "19,442\xa0"]) == "Net income: 19,442"

    def test_label_only_row(self):
        assert table_row_to_line(["Cost of sales:", "", None]) == "Cost of sales:"

    def test_blank_row(self):
        assert table_row_to_line([None, "", "$"]) == ""


class TestTableToChunks:
    ROWS = [["Item", "Q3"]] + [[f"Line item {i}", f"{i * 1000:,}"] for i in range(1, 40)]

    def test_every_chunk_carries_context_and_page(self):
        chunks = table_to_chunks(self.ROWS, "STATEMENT OF OPERATIONS Three Months Ended", 4, 300)
        assert len(chunks) > 1
        assert all(c.startswith("Table on page 4. STATEMENT OF OPERATIONS") for c in chunks)

    def test_chunks_respect_budget(self):
        chunks = table_to_chunks(self.ROWS, "ctx", 1, 300)
        header_len = len("Table on page 1. ctx\n")
        assert all(len(c) <= header_len + 300 + 60 for c in chunks)

    def test_no_row_lost_or_duplicated(self):
        chunks = table_to_chunks(self.ROWS, "ctx", 1, 300)
        body = "\n".join(c.split("\n", 1)[1] for c in chunks).splitlines()
        assert body == [table_row_to_line(r) for r in self.ROWS]

    def test_small_table_is_one_chunk(self):
        assert len(table_to_chunks(self.ROWS[:3], "ctx", 1, 600)) == 1

    def test_empty_table_gives_no_chunks(self):
        assert table_to_chunks([[None, ""], ["$"]], "ctx", 1, 600) == []
