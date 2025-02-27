#!/usr/bin/env bash

# --- Make sure the conda command is available in this shell ---
# The exact method depends on how conda is set up on your system.
# Often you'll see something like:
# source /path/to/miniconda3/etc/profile.d/conda.sh
# or
# eval "$(conda shell.bash hook)"

# Example:
source /mnt/data-home/miniforge3/bin/activate

# --- Activate the environment you want ---
conda activate dataverse-sdk

# --- Now run your commands under that environment ---
#for i in {1..20}; do
#    python upload.py --folder "split_${i}"
# for i in {2..6}; do
#     for j in {1..4}; do
#         echo "python import_vqa.py -host https://staging.visionai.linkervision.ai/dataverse/curation -e  julianlee@linkervision.com -p Baoyun5820 -s  0ca454e7-5036-4945-8853-6f96589ba9dc -project 101 --folder /mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/Kaoshsiung_76152_retrieval_curated_t220_part${i}_${j}/"
#         python import_vqa.py -host https://staging.visionai.linkervision.ai/dataverse/curation -e  julianlee@linkervision.com -p Baoyun5820 -s  0ca454e7-5036-4945-8853-6f96589ba9dc -project 101 --folder /mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/Kaoshsiung_76152_retrieval_curated_t220_part${i}_${j}/
#     done
# done

for i in {0..3}; do
    echo "python import_vqa.py -host https://visionai.linkervision.ai/dataverse/curation -e  julianlee@linkervision.com -p Baoyun5820 -s 2bd928e5-a98f-4aae-a093-8545c57c103f -project 230 --folder /mnt/data-home/mobility-multimodal/vlm-annotations/Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split${i}/"
    python import_vqa.py -host https://visionai.linkervision.ai/dataverse/curation -e  julianlee@linkervision.com -p Baoyun5820 -s 2bd928e5-a98f-4aae-a093-8545c57c103f -project 230 --folder /mnt/data-home/mobility-multimodal/vlm-annotations/Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split${i}/
done

# python import_vqa.py -host https://visionai.linkervision.ai/dataverse/curation -e  julianlee@linkervision.com -p Baoyun5820 -s 2bd928e5-a98f-4aae-a093-8545c57c103f -project 230 --folder /mnt/data-home/mobility-multimodal/vlm-annotations/Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split0/


# Optionally, deactivate the environment at the end
conda deactivate

