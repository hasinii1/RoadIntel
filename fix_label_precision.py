from pathlib import Path

# ============================================================
# ROADINTEL — FINAL YOLO LABEL PRECISION FIX
# Fixes only boxes whose boundaries are outside [0, 1].
# Uses higher precision to avoid rounding overflow.
# ============================================================

DATASET_ROOT = Path(r"F:\RoadIntel\data\processed\pothole")

LABEL_DIRS = [
    DATASET_ROOT / "labels" / "train",
    DATASET_ROOT / "labels" / "val",
]

fixed_boxes = 0
modified_files = 0
checked_boxes = 0


def clip(value):
    return max(0.0, min(1.0, value))


for label_dir in LABEL_DIRS:

    print()
    print("=" * 70)
    print(f"Checking: {label_dir}")
    print("=" * 70)

    for label_file in sorted(label_dir.glob("*.txt")):

        lines = label_file.read_text(encoding="utf-8").splitlines()

        new_lines = []
        file_changed = False

        for line_number, line in enumerate(lines, start=1):

            if not line.strip():
                new_lines.append(line)
                continue

            parts = line.split()

            if len(parts) != 5:
                new_lines.append(line)
                continue

            try:
                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                new_lines.append(line)
                continue

            checked_boxes += 1

            # Original boundaries
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            # Clip boundaries
            nx1 = clip(x1)
            ny1 = clip(y1)
            nx2 = clip(x2)
            ny2 = clip(y2)

            nw = nx2 - nx1
            nh = ny2 - ny1

            # If the box becomes unusable, leave it alone.
            if nw <= 0 or nh <= 0:
                new_lines.append(line)
                continue

            nxc = (nx1 + nx2) / 2
            nyc = (ny1 + ny2) / 2

            # Use 10 decimal places to avoid boundary rounding problems.
            repaired_line = (
                f"{class_id} "
                f"{nxc:.10f} "
                f"{nyc:.10f} "
                f"{nw:.10f} "
                f"{nh:.10f}"
            )

            # Only change the file if the calculated box differs.
            if repaired_line != line.strip():
                file_changed = True
                fixed_boxes += 1

                print(
                    f"FIXED: {label_file.name} "
                    f"(line {line_number})"
                )

            new_lines.append(repaired_line)

        if file_changed:
            label_file.write_text(
                "\n".join(new_lines) + "\n",
                encoding="utf-8"
            )
            modified_files += 1


print()
print("=" * 70)
print("FINAL PRECISION FIX COMPLETE")
print("=" * 70)
print(f"Boxes checked   : {checked_boxes}")
print(f"Boxes rewritten : {fixed_boxes}")
print(f"Files modified  : {modified_files}")
print("=" * 70)