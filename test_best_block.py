#!/usr/bin/env python3
"""Quick verification script for the "best block" logic."""

from __future__ import annotations

from visagism.visagism_calculator import VisagismCalculator
from visagism.report_formatter import ReportFormatter


def main() -> None:
    # Example values from the spreadsheet / user request
    measurements = {
        "eye_width": 3.2,
        "inter_ocular_distance": 3.5,
        "nose_width": 4.4,
        "mouth_width": 5.5,
        "face_width": 13.5,
        "lower_third": 7.0,
        "middle_third": 7.1,
        "upper_third": 5.0,
    }

    calc = VisagismCalculator(**measurements)
    result = calc.calculate()

    print("=" * 60)
    print("BEST BLOCK SELECTION")
    print("=" * 60)
    print(f"Best block: {result.best_block_name}")
    print()

    # Explain *why* by showing total deviation for each block
    blocks = [result.block_1_eye_width, result.block_2_inter_ocular, result.block_3_nose_width]
    for block in blocks:
        total = sum(
            abs(dev.deviation_percent)
            for dev in block.deviations
            if dev.deviation_percent is not None
        )
        marker = " <-- BEST" if block.block_name == result.best_block_name else ""
        print(f"  {block.block_name}: total abs deviation = {total:.2f}%{marker}")
        for dev in block.deviations:
            if dev.deviation_percent is not None:
                print(
                    f"      {dev.measurement_name}: dev={dev.deviation_percent:.2f}%"
                )
    print()

    print("=" * 60)
    print("FORMATTED CONSOLE OUTPUT")
    print("=" * 60)
    print(ReportFormatter.format_console(result))


if __name__ == "__main__":
    main()
