"""Tests for the glob and grep tools."""

from __future__ import annotations

import asyncio

from nanogent.tools import GlobFilesTool, GrepFilesTool, create_default_registry


def run(coro):
    return asyncio.run(coro)


def test_glob_finds_matching_files(tmp_path):
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("print(2)", encoding="utf-8")

    result = run(GlobFilesTool().execute("**/*.py", path=str(tmp_path)))

    assert "a.py" in result
    assert "c.py" in result
    assert "b.txt" not in result


def test_glob_skips_ignored_directories(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text("x", encoding="utf-8")
    (tmp_path / "real.py").write_text("y", encoding="utf-8")

    result = run(GlobFilesTool().execute("**/*.py", path=str(tmp_path)))

    assert "real.py" in result
    assert "junk.py" not in result


def test_glob_reports_missing_directory(tmp_path):
    result = run(GlobFilesTool().execute("*.py", path=str(tmp_path / "nope")))
    assert "Error" in result


def test_grep_returns_file_line_matches(tmp_path):
    (tmp_path / "app.py").write_text("import os\nTODO: fix me\nprint(1)\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("nothing here\n", encoding="utf-8")

    result = run(GrepFilesTool().execute("TODO", path=str(tmp_path)))

    assert "app.py:2" in result
    assert "TODO: fix me" in result


def test_grep_include_filter(tmp_path):
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("needle\n", encoding="utf-8")

    result = run(GrepFilesTool().execute("needle", path=str(tmp_path), include="*.py"))

    assert "a.py" in result
    assert "b.md" not in result


def test_grep_invalid_regex(tmp_path):
    result = run(GrepFilesTool().execute("([", path=str(tmp_path)))
    assert "invalid regular expression" in result


def test_grep_no_match(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    result = run(GrepFilesTool().execute("zzz", path=str(tmp_path)))
    assert "No matches" in result


def test_default_registry_includes_new_tools():
    registry = create_default_registry()
    assert "glob" in registry.tools
    assert "grep" in registry.tools
    for name, tool in registry.tools.items():
        spec = tool.to_openai_format()
        assert spec["function"]["name"] == name
