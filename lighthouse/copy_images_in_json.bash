#!/bin/bash

#JSON_FILE="/mnt/data-home/mobility-multimodal/data-curation/Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95.json"
#OUTPUT_ROOT='/mnt/lighthouseACD/augmented-curated-data'


# Input JSON files and output location
JSON_FILE="/home/julian/work/GroundingDINO/10001.json"
OUTPUT_ROOT='/mnt/data-home/julian/test/'
OUTPUT_FOLDER="$OUTPUT_ROOT/$(basename "$JSON_FILE" .json)"

# Performance and organization settings
NUM_JOBS=16
FILES_PER_SPLIT=5000  # Configure number of files per split folder

# Create base output directory
mkdir -p "$OUTPUT_FOLDER"

# Create a temporary helper script that will be called by parallel
HELPER_SCRIPT=$(mktemp)
chmod +x "$HELPER_SCRIPT"

# Write the processing logic to the helper script
cat > "$HELPER_SCRIPT" << 'EOF'
#!/bin/bash
IMAGE_PATH="$1"
SPLIT_INDEX="$2"
OUTPUT_FOLDER="$3"
SPLIT_DIR="$OUTPUT_FOLDER/split$SPLIT_INDEX"
SPLIT_MAPPING="$SPLIT_DIR/mapping.txt"

# Skip if empty or file doesn't exist
[ -z "$IMAGE_PATH" ] || [ ! -f "$IMAGE_PATH" ] && exit 0

# Get file extension
EXT="${IMAGE_PATH##*.}"

# Create hash
HASH=$(md5sum "$IMAGE_PATH" | cut -d' ' -f1)
NEW_FILENAME="${HASH}.${EXT}"
NEW_PATH="$SPLIT_DIR/$NEW_FILENAME"

# Skip if destination already exists
[ -f "$NEW_PATH" ] && exit 0

# Copy the file
cp "$IMAGE_PATH" "$NEW_PATH"

# Output mapping info if copy succeeded
[ $? -eq 0 ] && echo "$NEW_FILENAME,$IMAGE_PATH" >> "$SPLIT_MAPPING"
EOF

# Extract paths from JSON file
echo "Extracting paths from JSON file..."
TEMP_PATHS=$(mktemp)
jq -r '.[]' "$JSON_FILE" > "$TEMP_PATHS"

# Get total files for reporting
TOTAL_FILES=$(wc -l < "$TEMP_PATHS")
echo "Processing $TOTAL_FILES files across multiple splits ($FILES_PER_SPLIT files per split)"

# Calculate total number of splits needed
TOTAL_SPLITS=$(( ($TOTAL_FILES + $FILES_PER_SPLIT - 1) / $FILES_PER_SPLIT ))
echo "Will create $TOTAL_SPLITS split folders"

# Process files by split
for (( SPLIT_INDEX=0; SPLIT_INDEX<$TOTAL_SPLITS; SPLIT_INDEX++ )); do
    SPLIT_DIR="$OUTPUT_FOLDER/split$SPLIT_INDEX"
    mkdir -p "$SPLIT_DIR"
    
    # Create mapping file header for this split
    echo "new_filename,original_path" > "$SPLIT_DIR/mapping.txt"
    
    # Calculate start and end lines for this split
    START_LINE=$(( $SPLIT_INDEX * $FILES_PER_SPLIT + 1 ))
    END_LINE=$(( ($SPLIT_INDEX + 1) * $FILES_PER_SPLIT ))
    
    # Make sure we don't exceed total files
    if [ $END_LINE -gt $TOTAL_FILES ]; then
        END_LINE=$TOTAL_FILES
    fi
    
    # Extract paths for this split
    SPLIT_PATHS=$(mktemp)
    sed -n "${START_LINE},${END_LINE}p" "$TEMP_PATHS" > "$SPLIT_PATHS"
    
    echo "Processing split$SPLIT_INDEX (files $START_LINE to $END_LINE)..."
    
    # Process this split in parallel using the helper script
    cat "$SPLIT_PATHS" | parallel --progress --jobs $NUM_JOBS "$HELPER_SCRIPT" {} $SPLIT_INDEX "$OUTPUT_FOLDER"
    
    # Clean up split paths file
    rm -f "$SPLIT_PATHS"
done

# Clean up
rm -f "$TEMP_PATHS"
rm -f "$HELPER_SCRIPT"

echo "Processing complete. Files distributed across $TOTAL_SPLITS split folders."