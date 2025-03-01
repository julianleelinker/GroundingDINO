#!/bin/bash

JSON_FILE="/mnt/data-home/mobility-multimodal/data-curation/Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95.json"
OUTPUT_ROOT='/mnt/lighthouseACD/augmented-curated-data'

#JSON_FILE="/home/julian/work/GroundingDINO/10001.json"
#OUTPUT_ROOT='/mnt/data-home/julian/test/'


OUTPUT_FOLDER="$OUTPUT_ROOT/$(basename "$JSON_FILE" .json)"

# Settings
NUM_JOBS=16
FILES_PER_SPLIT=5000

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

# Create helper script
HELPER_SCRIPT=$(mktemp)
chmod +x "$HELPER_SCRIPT"

cat > "$HELPER_SCRIPT" << 'EOF'
#!/bin/bash
IMAGE_PATH="$1"
SPLIT_INDEX="$2"
OUTPUT_FOLDER="$3"
SPLIT_DIR="$OUTPUT_FOLDER/split$SPLIT_INDEX"
SPLIT_MAPPING="$SPLIT_DIR/mapping.txt"

[ -z "$IMAGE_PATH" ] || [ ! -f "$IMAGE_PATH" ] && exit 0
EXT="${IMAGE_PATH##*.}"
HASH=$(md5sum "$IMAGE_PATH" | cut -d' ' -f1)
NEW_FILENAME="${HASH}.${EXT}"
NEW_PATH="$SPLIT_DIR/$NEW_FILENAME"

if [ -f "$NEW_PATH" ]; then
    # File exists, add to mapping without copying
    echo "$NEW_FILENAME,$IMAGE_PATH" >> "$SPLIT_MAPPING"
    exit 0
fi
cp "$IMAGE_PATH" "$NEW_PATH"
[ $? -eq 0 ] && echo "$NEW_FILENAME,$IMAGE_PATH" >> "$SPLIT_MAPPING"
EOF




# Extract paths
TEMP_PATHS=$(mktemp)
jq -r '.[]' "$JSON_FILE" > "$TEMP_PATHS"
TOTAL_FILES=$(wc -l < "$TEMP_PATHS")
TOTAL_SPLITS=$(( ($TOTAL_FILES + $FILES_PER_SPLIT - 1) / $FILES_PER_SPLIT ))
echo "Processing $TOTAL_FILES files across $TOTAL_SPLITS splits ($FILES_PER_SPLIT per split)"

# Process each split
for (( SPLIT_INDEX=0; SPLIT_INDEX<$TOTAL_SPLITS; SPLIT_INDEX++ )); do
    SPLIT_DIR="$OUTPUT_FOLDER/split$SPLIT_INDEX"
    mkdir -p "$SPLIT_DIR"
    echo "new_filename,original_path" > "$SPLIT_DIR/mapping.txt"
    
    # Get lines for this split
    START_LINE=$(( $SPLIT_INDEX * $FILES_PER_SPLIT + 1 ))
    END_LINE=$(( ($SPLIT_INDEX + 1) * $FILES_PER_SPLIT ))
    [ $END_LINE -gt $TOTAL_FILES ] && END_LINE=$TOTAL_FILES
    
    SPLIT_PATHS=$(mktemp)
    sed -n "${START_LINE},${END_LINE}p" "$TEMP_PATHS" > "$SPLIT_PATHS"
    
    echo "Processing split$SPLIT_INDEX ($START_LINE-$END_LINE)..."
    cat "$SPLIT_PATHS" | parallel --progress --bar --jobs $NUM_JOBS "$HELPER_SCRIPT" {} $SPLIT_INDEX "$OUTPUT_FOLDER"
    rm -f "$SPLIT_PATHS"
done

# Clean up
rm -f "$TEMP_PATHS" "$HELPER_SCRIPT"
echo "Processing complete. Files distributed across $TOTAL_SPLITS split folders."