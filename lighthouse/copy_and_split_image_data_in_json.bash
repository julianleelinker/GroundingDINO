#!/bin/bash

# JSON_FILE="/mnt/data-home/mobility-multimodal/data-curation/Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95.json"
# JSON_FILE='/mnt/data-home/mobility-multimodal/data-curation/Kaohsiung_Data_V2/Kaohsiung_Data_V2_image_list_keep_0.95.json'

# JSON_FILE='/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250115/Transportation_20250115_image_list_keep_0.95_rededuplicate.json'
JSON_FILE='/mnt/data-home/mobility-multimodal/data-curation/Ports_Corporation/20250124/Ports_Corporation_20250124_image_list_keep_0.95.json'
OUTPUT_ROOT="/mnt/lighthouseACD/augmented-curated-data"


# for testing
# JSON_FILE="/home/julian/work/GroundingDINO/10001.json"
# OUTPUT_ROOT='/mnt/data-home/julian/test/'

OUTPUT_FOLDER="$OUTPUT_ROOT/$(basename "$JSON_FILE" .json)"

# Settings
NUM_JOBS=128
FILES_PER_SPLIT=5000

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

# Define the process_image function (replaces helper script)
process_image() {
    local IMAGE_PATH="$1"
    local SPLIT_INDEX="$2"
    local OUTPUT_FOLDER="$3"
    local SPLIT_DIR="$OUTPUT_FOLDER/split$SPLIT_INDEX"
    local SPLIT_MAPPING="$SPLIT_DIR/mapping.txt"

    [ -z "$IMAGE_PATH" ] || [ ! -f "$IMAGE_PATH" ] && return 0
    local EXT="${IMAGE_PATH##*.}"
    
    # Generate a timestamp-based filename with a random suffix
    local TIMESTAMP=$(date +%Y%m%d%H%M%S%3N)  # Format: YYYYMMDDHHMMSSmmm (milliseconds)
    local RANDOM_SUFFIX=$((RANDOM % 10000))  # Random 4-digit number
    local NEW_FILENAME="${TIMESTAMP}_${RANDOM_SUFFIX}.${EXT}"
    local NEW_PATH="$SPLIT_DIR/$NEW_FILENAME"

    # Ensure the file does not exist
    # while [ -f "$NEW_PATH" ]; do
    #     RANDOM_SUFFIX=$((RANDOM % 10000))  # Generate new random number
    #     NEW_FILENAME="${TIMESTAMP}_${RANDOM_SUFFIX}.${EXT}"
    #     NEW_PATH="$SPLIT_DIR/$NEW_FILENAME"
    # done

    while [ -f "$NEW_PATH" ]; do
        NEW_FILENAME="${NEW_FILENAME%.*}a.${EXT}"  # Append 'a' before extension
        NEW_PATH="$SPLIT_DIR/$NEW_FILENAME"
    done


    cp "$IMAGE_PATH" "$NEW_PATH"
    [ $? -eq 0 ] && echo "$NEW_FILENAME,$IMAGE_PATH" >> "$SPLIT_MAPPING"
}


# Export the function so parallel can use it
export -f process_image

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
    parallel --progress --bar --jobs $NUM_JOBS process_image {} $SPLIT_INDEX "$OUTPUT_FOLDER" < "$SPLIT_PATHS"
    rm -f "$SPLIT_PATHS"
done

# Clean up
rm -f "$TEMP_PATHS"
echo "Processing complete. Files distributed across $TOTAL_SPLITS split folders."