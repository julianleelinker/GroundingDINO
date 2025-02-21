import subprocess

# Define the Conda environment
conda_env = "lighthouse"

# Base command template
base_command = (
    "CUDA_VISIBLE_DEVICES=1 nohup conda run -n {conda_env} python demo/inference_gpt_on_images_in_folder.py "
    "-c groundingdino/config/GroundingDINO_SwinT_OGC.py "
    "-p weights/groundingdino_swint_ogc.pth "
    "-o /mnt/data-home/mobility-multimodal/gdino-coco "
    "--box_threshold 0.4 "
    "--text_threshold 0.3 "
    "--high_threshold 0.35 "
    "-t 'all' "
    "--enlarge_scale 1.4 "
    "--ios_threshold 0.3 "
    # "-i /mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95-part_{i} "
    "-i /mnt/data-home/julian/lighthouse/augmented-curated-data/Transportation_20250109_image_list_keep_0.95-split_{i}"
    # "| tee trans1230-{i}.log 2>&1 & disown"
    "> trans0109-{i}.log 2>&1 &"
)

# Loop through i = 1 to 12
for i in range(20, 24):
    command = base_command.format(i=i, conda_env=conda_env)
    print(f"Executing: {command}")
    subprocess.run(command, shell=True, executable="/bin/bash")