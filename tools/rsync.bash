#!/bin/bash

# Source and destination paths
# SOURCE_BASE="/mnt/lighthouseACD/augmented-curated-data/Public_Works_20241230_image_list_keep_0.95"
# DEST_BASE="b202:/raid/julian/lighthouseACD/augmented-curated-data/Public_Works_20241230_image_list_keep_0.95"
SOURCE_BASE="/mnt/lighthouseACD/augmented-curated-data/linker_4M_image_list_keep_0.95"
DEST_BASE="b202:/raid/julian/lighthouseACD/augmented-curated-data/linker_4M_image_list_keep_0.95"

# Rsync options
RSYNC_OPTS="-avz --ignore-existing --no-compress --whole-file --progress"

# Loop from split2 to split400
for i in $(seq 0 858); do
    echo "===== Starting transfer for split$i ====="
    rsync $RSYNC_OPTS "$SOURCE_BASE/split$i" "$DEST_BASE/"
    echo "===== Completed transfer for split$i ====="
done

echo "All transfers completed."