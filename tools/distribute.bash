#!/bin/bash
SOURCE_DIR="/mnt/lighthouseACD/Public_Works_20241230_image_list_keep_0.95"
FILES_PER_FOLDER=5000  # Adjust as needed

# Ensure source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "❌ Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Change to source directory to avoid path issues
cd "$SOURCE_DIR" || exit


# Generate list of files and split them correctly
echo "📂 Generating list and chunks..."
find . -maxdepth 1 -type f | sort > /tmp/tmp_full_list.txt
# Count total files
total_files=$(wc -l < /tmp/tmp_full_list.txt)

if [[ $total_files -eq 0 ]]; then
    echo "❌ Error: No files found in '$SOURCE_DIR'."
    rm -f /tmp/tmp_full_list.txt
    exit 1
fi

# Ensure `split` properly creates all necessary chunks
split -l "$FILES_PER_FOLDER" -d -a 4 /tmp/tmp_full_list.txt tmp_list_parallel_
rm -f /tmp/tmp_full_list.txt

# Count how many split lists were created
split_count=$(ls tmp_list_parallel* 2>/dev/null | wc -l)
# Check if there are files to split
if [[ $split_count -eq 0 ]]; then
    echo "❌ Error: No files found in '$SOURCE_DIR' to split."
    exit 1
fi

# Create destination folders before moving files
seq 0 $((split_count-1)) | parallel mkdir -p "split{}"

# ✅ Move files into the correct folders using parallel processing
total_files=$(find . -maxdepth 1 -type f | wc -l | awk '{print $1}')

i=0
for file in tmp_list_parallel_*; do
    mv "$file" "tmp_list_parallel_$i"
    ((i++))
done

#for file in tmp_list_parallel_*; do
#    if [[ -f "$file" ]]; then
#        folder_num=$(echo "$file" | grep -oE '[0-9]+')
#        echo "📂 Moving files from $file to split$folder_num/"
#        cat "$file" | parallel --bar mv "{}" "split$folder_num/"
#    else
#        echo "⚠️ Warning: $file does not exist, skipping..."
#     fi
#done

# Start counter from 0
i=0
# Loop until all split files are processed
while [[ -f "tmp_list_parallel_$i" ]]; do
    split_folder="split$i"
    echo "📂 Moving files from tmp_list_parallel_$i to $split_folder/"
    # Using `--null` to handle filenames with spaces and special characters
    cat "tmp_list_parallel_$i" | parallel --jobs 0 --delimiter '\n' --bar mv "{}" "$split_folder/"
    # Increment counter
    ((i++))
done







# Cleanup temporary files
rm -f tmp_list_parallel*
echo "✅ Done! Files distributed into $split_count folders."
# Move back to original directory
cd - > /dev/null

