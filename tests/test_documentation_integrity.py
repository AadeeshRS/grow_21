"""Tests verifying documentation integrity, absence of invalid ASCII control characters,
absence of literal tabs in Markdown, and intact LaTeX escape sequences.
"""

from pathlib import Path
import re
import pytest

ROOT_DIR = Path(__file__).parent.parent
DOC_EXTENSIONS = {".md", ".txt"}
ALL_TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".py", ".toml"}


def _get_project_files(extensions):
    files = []
    for p in ROOT_DIR.rglob("*"):
        if not p.is_file():
            continue
        if any(part in p.parts for part in [".venv", "__pycache__", ".pytest_cache", ".git"]):
            continue
        if p.suffix.lower() in extensions:
            files.append(p)
    return files


def test_no_invalid_control_characters_in_text_files():
    """Scans all repository text files to verify absence of non-printing control characters
    (ASCII < 32 other than \t, \n, \r).
    """
    invalid_bytes = bytes(set(range(32)) - {9, 10, 13})
    violations = []
    for p in _get_project_files(ALL_TEXT_EXTENSIONS):
        raw = p.read_bytes()
        # Fast C-level scan: if none of the invalid bytes appear in raw, skip slow enumeration
        if not any(b in raw for b in invalid_bytes):
            continue
        for idx, b in enumerate(raw):
            if b in invalid_bytes:
                rel_p = p.relative_to(ROOT_DIR)
                violations.append((str(rel_p), idx, hex(b)))

    assert not violations, f"Found {len(violations)} invalid control characters in text files: {violations[:10]}"


def test_no_literal_tabs_in_markdown_and_text_docs():
    """Rejects literal tab characters embedded in Markdown and text documentation files."""
    tab_violations = []
    for p in _get_project_files(DOC_EXTENSIONS):
        raw = p.read_bytes()
        tab_count = raw.count(b"\t")
        if tab_count > 0:
            rel_p = p.relative_to(ROOT_DIR)
            tab_violations.append((str(rel_p), tab_count))

    assert not tab_violations, f"Found literal tab characters in documentation files: {tab_violations}"


def test_key_latex_and_identifier_expressions_intact():
    """Asserts that required core LaTeX fragments and key code identifiers appear intact in documents."""
    readme_text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    cl_text = (ROOT_DIR / "docs" / "research" / "phase_2" / "CHANGELOG_PHASE_2.md").read_text(encoding="utf-8")
    report_text = (ROOT_DIR / "docs" / "research" / "phase_2" / "PHASE_2_AUDIT_REPORT.md").read_text(encoding="utf-8")

    required_in_readme = [
        r"D^{\nu_D}",
        r"\gamma = 2.4, \nu_D = 1.0",
        r"\gamma = 3.5, \nu_D = 2.0",
        r"\text{logit}",
        r"p_{\text{clean}}",
        r"\theta_{\text{eff}}",
        "target_adaptive",
    ]
    for expr in required_in_readme:
        assert expr in readme_text, f"Expression '{expr}' missing from README.md"

    required_in_changelog = [
        r"D^{\nu_D}",
        r"\gamma = 2.4, \nu_D = 1.0",
        r"\gamma = 3.5, \nu_D = 2.0",
        r"\text{logit}",
        r"p_{\text{clean}}",
        r"\theta_{\text{eff}}",
        "test_persisted_calibration_bins_and_ece_consistency",
        "target_adaptive",
        "tests/test_audit_regressions.py",
    ]
    for expr in required_in_changelog:
        assert expr in cl_text, f"Expression '{expr}' missing from CHANGELOG_PHASE_2.md"

    required_in_report = [
        r"D^{\nu_D}",
        r"\gamma = 2.4, \nu_D = 1.0",
        r"\gamma = 3.5, \nu_D = 2.0",
        r"\text{logit}",
        r"p_{\text{clean}}",
        r"\theta_{\text{eff}}",
        "test_persisted_calibration_bins_and_ece_consistency",
        "target_adaptive",
        "tests/test_audit_regressions.py",
    ]
    for expr in required_in_report:
        assert expr in report_text, f"Expression '{expr}' missing from PHASE_2_AUDIT_REPORT.md"


def test_no_fractured_latex_lines():
    """Detects broken newline splits such as standalone u_D} or broken escape remnants in Markdown docs."""
    fracture_patterns = [
        re.compile(r"^\s*u_D\}", re.MULTILINE),
        re.compile(r"^\s*ext\{", re.MULTILINE),
        re.compile(r"^\s*heta\}", re.MULTILINE),
    ]

    fractures = []
    for p in _get_project_files({".md"}):
        text = p.read_text(encoding="utf-8")
        for pat in fracture_patterns:
            matches = list(pat.finditer(text))
            if matches:
                rel_p = p.relative_to(ROOT_DIR)
                fractures.append((str(rel_p), pat.pattern, len(matches)))

    assert not fractures, f"Found fractured LaTeX line splits: {fractures}"
