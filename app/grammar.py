"""
Lightweight wrapper around language_tool_python to provide grammar corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import language_tool_python
except ImportError:  # pragma: no cover - handled at runtime
    language_tool_python = None  # type: ignore


@dataclass
class GrammarResult:
    corrected: str
    issues_found: int


class GrammarHelper:
    def __init__(self, language: str = "en-US") -> None:
        self.language = language
        self.tool = None
        if language_tool_python is not None:
            try:
                self.tool = language_tool_python.LanguageTool(language)
            except Exception:
                self.tool = None

    @property
    def available(self) -> bool:
        return self.tool is not None

    def correct(self, text: str) -> GrammarResult:
        if not self.tool:
            raise RuntimeError(
                "Grammar engine not available. Install language-tool-python + Java."
            )
        matches = self.tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        return GrammarResult(corrected=corrected, issues_found=len(matches))


__all__ = ["GrammarHelper", "GrammarResult"]

