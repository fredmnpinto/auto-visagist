#!/usr/bin/env python3
"""Verification script for woman_7 measurements."""

import sys
sys.path.insert(0, '/home/fred/Projects/MSc/VC/Project_backup-visagist-calculator-worktree')

from visagism.visagism_calculator import VisagismCalculator

# Measurements from woman_7
calc = VisagismCalculator(
    eye_width=134.44,
    inter_ocular_distance=180.14,
    nose_width=144.01,
    mouth_width=264.03,
    face_width=670.39,
    lower_third=290.00,
    middle_third=366.10,
    upper_third=377.90,
    hairline_fallback_used=False,
)

result = calc.calculate()

print("=" * 60)
print("VISAGISM ANALYSIS — woman_7")
print("=" * 60)

print(f"\nTotal face height: {result.measurements.total_face_height}")
print(f"Face width: {result.measurements.face_width}")
print()

print(f"BEST BLOCK: {result.best_block_name}")
print("-" * 40)

print(f"\nReference measurement: {result.best_block.reference_measurement}")
print(f"Reference value: {result.best_block.reference_value}")
print(f"Ideal face width: {result.best_block.ideal_face_width}")
print(f"Ideal face height: {result.best_block.ideal_face_height}")
print(f"Ideal mouth width: {result.best_block.ideal_mouth_width}")
print(f"Ideal length from width: {result.best_block.ideal_length_from_width}")

print("\nDeviations:")
for dev in result.best_block.deviations:
    flag = " [FLAGGED]" if dev.is_flagged else ""
    print(f"  {dev.measurement_name}: actual={dev.actual}, ideal={dev.ideal}, deviation={dev.deviation_percent}%{flag}")

print("\n" + "=" * 60)
print("FACE HEIGHT DEVIATION — DETAILED CALCULATION")
print("=" * 60)

actual_face_height = result.measurements.total_face_height
ideal_length_from_width = result.best_block.ideal_length_from_width

deviation = ((actual_face_height - ideal_length_from_width) / ideal_length_from_width) * 100

print(f"\n  actual_face_height = lower_third + middle_third + upper_third")
print(f"                     = 290.00 + 366.10 + 377.90")
print(f"                     = {actual_face_height}")
print(f"\n  ideal_length_from_width = face_width * GOLDEN_RATIO")
print(f"                          = {result.measurements.face_width} * 1.618")
print(f"                          = {ideal_length_from_width}")
print(f"\n  deviation_percent = ((actual - ideal) / ideal) * 100")
print(f"                    = (({actual_face_height} - {ideal_length_from_width}) / {ideal_length_from_width}) * 100")
print(f"                    = ({actual_face_height - ideal_length_from_width} / {ideal_length_from_width}) * 100")
print(f"                    = {deviation:.4f}%")
print(f"\n  Rounded to 2 decimals: {round(deviation, 2)}%")

print("\n" + "=" * 60)
print("ALL 3 BLOCKS COMPARISON")
print("=" * 60)

for block in [result.block_1_eye_width, result.block_2_inter_ocular, result.block_3_nose_width]:
    total = sum(abs(d.deviation_percent) for d in block.deviations if d.deviation_percent is not None)
    marker = " <-- BEST" if block == result.best_block else ""
    print(f"\n{block.block_name}{marker}")
    print(f"  Total absolute deviation: {total:.2f}%")
    for dev in block.deviations:
        print(f"    - {dev.measurement_name}: {dev.deviation_percent}%")
