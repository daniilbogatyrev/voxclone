from voxclone.eval.report import render_markdown, write_report

def test_render_markdown_has_header_and_rows():
    rows = [
        {"checkpoint": "ckpt_10", "similarity": 0.9, "naturalness": 4.2, "wer": 0.05, "score": 0.88},
    ]
    md = render_markdown(rows, ceiling=0.95)
    assert "| checkpoint | similarity | naturalness | wer | score |" in md
    assert "ckpt_10" in md
    assert "ceiling" in md.lower()

def test_write_report_creates_file(tmp_path):
    p = tmp_path / "r.md"
    write_report(p, [{"checkpoint": "c", "similarity": 0.9, "naturalness": 4.0,
                      "wer": 0.1, "score": 0.8}], ceiling=0.9)
    assert p.exists() and "c" in p.read_text()
