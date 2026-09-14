from pathlib import Path

# ============================================================
# ROADINTEL — REPAIR INVALID YOLO LABELS
# Only fixes bounding boxes that extend outside [0, 1].
# All valid annotations are left unchanged.
# ============================================================

DATASET_ROOT = Path(r"F:\RoadIntel\data\processed\pothole")

LABEL_DIRS = [
    DATASET_ROOT / "labels" / "train",
    DATASET_ROOT / "labels" / "val",
]

MIN_SIZE = 1e-6

total_files_checked = 0
total_boxes_checked = 0
total_boxes_repaired = 0
total_files_modified = 0


def repair_value(value):
    """Clip a normalized YOLO coordinate/value to [0, 1]."""
    return max(0.0, min(1.0, value))


for label_dir in LABEL_DIRS:

    if not label_dir.exists():
        print(f"ERROR: Folder not found: {label_dir}")
        continue

    print()
    print("=" * 70)
    print(f"Checking: {label_dir}")
    print("=" * 70)

    for label_file in sorted(label_dir.glob("*.txt")):

        total_files_checked += 1

        original_lines = label_file.read_text(
            encoding="utf-8"
        ).splitlines()

        repaired_lines = []
        file_changed = False

        for line_number, line in enumerate(original_lines, start=1):

            stripped = line.strip()

            # Preserve blank lines exactly
            if not stripped:
                repaired_lines.append(line)
                continue

            parts = stripped.split()

            # Expected YOLO format:
            # class x_center y_center width height
            if len(parts) != 5:
                print(
                    f"WARNING: {label_file.name}, line {line_number}: "
                    f"unexpected format — left unchanged."
                )
                repaired_lines.append(line)
                continue

            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(
                    f"WARNING: {label_file.name}, line {line_number}: "
                    f"non-numeric values — left unchanged."
                )
                repaired_lines.append(line)
                continue

            total_boxes_checked += 1

            # Calculate original box boundaries
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Check whether the box is already completely valid
            valid = (
                0.0 <= x1 <= 1.0
                and 0.0 <= y1 <= 1.0
                and 0.0 <= x2 <= 1.0
                and 0.0 <= y2 <= 1.0
                and width > 0
                and height > 0
            )

            if valid:
                # Keep valid annotation EXACTLY as it was.
                repaired_lines.append(line)
                continue

            # ------------------------------------------------
            # Repair:
            # Clip the bounding-box corners to image limits.
            # ------------------------------------------------

            new_x1 = repair_value(x1)
            new_y1 = repair_value(y1)
            new_x2 = repair_value(x2)
            new_y2 = repair_value(y2)

            new_width = new_x2 - new_x1
            new_height = new_y2 - new_y1

            # If clipping creates an unusable box,
            # do NOT silently create a bad annotation.
            if new_width <= MIN_SIZE or new_height <= MIN_SIZE:
                print(
                    f"WARNING: {label_file.name}, line {line_number}: "
                    f"box becomes invalid after clipping — left unchanged."
                )
                repaired_lines.append(line)
                continue

            new_x_center = (new_x1 + new_x2) / 2
            new_y_center = (new_y1 + new_y2) / 2

            repaired_line = (
                f"{class_id} "
                f"{new_x_center:.6f} "
                f"{new_y_center:.6f} "
                f"{new_width:.6f} "
                f"{new_height:.6f}"
            )

            repaired_lines.append(repaired_line)

            file_changed = True
            total_boxes_repaired += 1

            print(
                f"REPAIRED: {label_file.name} "
                f"(line {line_number})"
            )

        if file_changed:
            label_file.write_text(
                "\n".join(repaired_lines) + "\n",
                encoding="utf-8"
            )

            total_files_modified += 1


print()
print("=" * 70)
print("ROADINTEL LABEL REPAIR COMPLETE")
print("=" * 70)
print(f"Label files checked : {total_files_checked}")
print(f"Boxes checked       : {total_boxes_checked}")
print(f"Boxes repaired      : {total_boxes_repaired}")
print(f"Files modified      : {total_files_modified}")
print("=" * 70)