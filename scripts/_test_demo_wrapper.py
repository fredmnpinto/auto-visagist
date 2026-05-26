"""Test wrapper for demo_hairline_steps.py in headless environment.

Patches cv2.imshow and cv2.waitKey so the script can run without a display.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Patch cv2 BEFORE importing the demo script
import cv2

_imshow_calls: list[tuple[str, object]] = []
_waitkey_calls: list[int] = []


def _fake_imshow(winname: str, mat: object) -> None:
    _imshow_calls.append((winname, mat))


def _fake_waitkey(delay: int = 0) -> int:
    _waitkey_calls.append(delay)
    return ord(" ")  # simulate space key press


cv2.imshow = _fake_imshow
cv2.waitKey = _fake_waitkey

# Now import and run the demo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.demo_hairline_steps import main  # noqa: E402

if __name__ == "__main__":
    with patch.object(sys, "argv", ["demo_hairline_steps.py", "--visualize"]):
        main()

    print(f"\n[TEST WRAPPER] cv2.imshow called {_imshow_calls.__len__()} times")
    for winname, _ in _imshow_calls:
        print(f"  - Window: {winname}")
    print(f"[TEST WRAPPER] cv2.waitKey called {_waitkey_calls.__len__()} times")
    print(f"[TEST WRAPPER] Delays: {_waitkey_calls}")
