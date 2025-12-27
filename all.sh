#!/bin/bash
# Run stage2 on all plate.svg files

set -e

OUTPUT_DIR="${1:-output}"

# Find all plate.svg files and sort them naturally
plates=$(find "$OUTPUT_DIR" -name "plate.svg" | sort -V)

if [ -z "$plates" ]; then
    echo "No plate.svg files found in $OUTPUT_DIR"
    exit 1
fi

count=$(echo "$plates" | wc -l | tr -d ' ')
current=0

echo "Found $count plate.svg files"
echo "========================================"

for plate in $plates; do
    current=$((current + 1))
    echo ""
    echo "[$current/$count] Processing: $plate"
    echo "----------------------------------------"

    uv run python segment.py --no-overwrite stage2 "$plate"

    echo "----------------------------------------"
    echo "[$current/$count] Done: $plate"
done

echo ""
echo "========================================"
echo "All $count plates processed!"
