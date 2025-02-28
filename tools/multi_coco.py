import subprocess
import argparse


# Define the Conda environment
CONDA_ENV = "lighthouse"


def make_parser():
    parser = argparse.ArgumentParser("Resize Coco Dataset")
    parser.add_argument(
        "-i",
        "--input_dataset",
        required=True,
        type=str,
        help="dataset path"
    )
    parser.add_argument(
        "-s",
        "--start_id",
        required=True,
        type=int,
        help="start id",
    )
    parser.add_argument(
        "-e",
        "--end_id",
        required=True,
        type=int,
        help="end id",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        required=True,
        type=int,
        help="gpu id",
    )
    return parser.parse_args()

if __name__=="__main__":
    args = make_parser()
    # Base command template
    base_command = (
        "CUDA_VISIBLE_DEVICES={gpu_id} nohup conda run -n {conda_env} python demo/inference_gpt_on_images_in_folder.py "
        "-c groundingdino/config/GroundingDINO_SwinT_OGC.py "
        "-p weights/groundingdino_swint_ogc.pth "
        # "-o /mnt/data-home/mobility-multimodal/gdino-coco/{dataset} "
        "-o /mnt/lighthouseACD/ACD-gdino-COCO/{dataset} "
        "--box_threshold 0.4 "
        "--text_threshold 0.3 "
        "--high_threshold 0.35 "
        "-t 'all' "
        "--enlarge_scale 1.4 "
        "--ios_threshold 0.3 "
        # "-i /mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95-part_{i} "
        # "-i /mnt/data-home/julian/lighthouse/augmented-curated-data/Transportation_20250109_image_list_keep_0.95-split_{i}"
        # "-i /mnt/data-home/julian/lighthouse/augmented-curated-data/Sports_Development_20241223_image_list_keep_0.95-split_{i}"
        # "-i /mnt/data-home/julian/lighthouse/augmented-curated-data/{dataset}/split{i} "
        "-i /mnt/lighthouseACD/augmented-curated-data/{dataset}/split{i} "
        "> {dataset}-{i}.log 2>&1 &"
    )


    # Loop through i = 1 to 12
    for i in range(args.start_id, args.end_id+1):
        # dataset = 'Sports_Development_20250109_image_list_keep_0.95'
        command = base_command.format(i=i, conda_env=CONDA_ENV, dataset=args.input_dataset, gpu_id=args.gpu_id)
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, executable="/bin/bash")
