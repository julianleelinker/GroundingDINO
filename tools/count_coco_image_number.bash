#!/bin/bash

# Define root directory
ROOT_DIR="/mnt/lighthouseACD/ACD-gdino-COCO/"

# List of prefixes (Modify this list as needed)
PREFIXES=(
    "China_Steel"
    "Kaohsiung-full-dataset"
    "Mass_Rapid_Transit"
    "Ports_Corporation"
    "Public_Works"
    "Sports_Development"
    "Taiwan_Power"
    "Transportation"
    "Water_Resources"
)

# Loop through each prefix
for prefix in "${PREFIXES[@]}"; do
    total_images=0

    # Find folders matching the prefix
    for folder in "$ROOT_DIR"/"$prefix"*; do
        # Skip if no matching directories exist
        [ -d "$folder" ] || continue

        # Loop through splitX directories inside the folder
        for split_dir in "$folder"/split*; do
            # Check if "uploaded" is a file
            if [ -f "$split_dir/uploaded" ]; then
                # Count images in "images" directory
                if [ -d "$split_dir/images" ]; then
                    count=$(find "$split_dir/images" -type f | wc -l)
                    total_images=$((total_images + count))
                fi
            fi
        done
    done

    # Print the total count for this prefix
    echo "COCO images in $prefix folders: $total_images"
done
