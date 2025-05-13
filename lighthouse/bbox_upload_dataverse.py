import pathlib
import time
import subprocess
import os
import tqdm
import fire
from common import get_depart_date, DATAVERSE_PASSWORD, DATAVERSE_BBOX_GOV_PROJECT_ID, DATAVERSE_BBOX_QA_PROJECT_ID
from common import DINO_COCO_FOLDERS_0508, DINO_COCO_SPLITS_0508


def execute(data_path, command):
    print(command)
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable="/bin/bash")
            
        # Print both stdout and stderr
        print("=== Command Output ===")
        print(result.stdout)
        if result.stderr:
            print("=== Error Output ===")
            print(result.stderr)
        # subprocess.run(command, shell=True, check=True)
        print(f"COCO {data_path} uploaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


def check_and_execute(data_path, command):
    done_path = pathlib.Path(data_path) / 'done'
    upload = pathlib.Path(data_path) / 'uploaded'
    if done_path.exists() and not upload.exists():
        print(f"COCO {data_path} done and not uploaded, Running command:")
        print(command)
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable="/bin/bash")
                
            # Print both stdout and stderr
            print("=== Command Output ===")
            print(result.stdout)
            if result.stderr:
                print("=== Error Output ===")
                print(result.stderr)
            # subprocess.run(command, shell=True, check=True)

            upload.touch()
            print(f"COCO {data_path} uploaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
    elif done_path.exists() and upload.exists():
        print(f"COCO {data_path} done and already uploaded")
    else:
        print(f"COCO {data_path} not done yet.")


def main(conda_env, prefix):
    print("Starting auto upload to dataverse")
    wait_time = 600
    # coco_root ='/mnt/lighthouseACD/ACD-gdino-COCO'
    # coco_root = '/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand'
    # coco_root = "/mnt/data-home/mobility-multimodal/revised_bbox/datasets"
    # coco_root = "/mnt/data-home/mobility-multimodal/revised_bbox/deduplicated"
    # coco_root = "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0422"
    # coco_root = "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0423"
    coco_root = "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0428"
    file_path_list =[
        # p for p in pathlib.Path(coco_root).glob('*/*') if p.is_dir()
        "/mnt/lighthouseACD/ACD-gdino-COCO-new/Transportation_20250319_image_list_keep_0.95/split37"
    ]
    file_path_list = DINO_COCO_SPLITS_0508
    
    # file_path_list = file_path_list[:1]
    import ipdb; ipdb.set_trace()
    while True:
        for i, file_path in tqdm.tqdm(enumerate(file_path_list), total=len(file_path_list)):
            print(f"dataset number {i}")
            depart, split = get_depart_date(file_path, ch=True)
            dataset_name = f"{prefix}_{depart}_{split}"
            # import ipdb; ipdb.set_trace()

            # for checkpoint
            # command = f'conda run -n {conda_env} python tools/import_dataset_from_local.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {DATAVERSE_PASSWORD} -s 697aa90b-00d0-4455-8863-bd2ad70a93e7 -project 121 --folder {file_path} -name {dataset_name} -type annotated_data -anno coco'
            # command = f'conda run -n {conda_env} python tools/import_dataset_from_local.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {DATAVERSE_PASSWORD} -s 697aa90b-00d0-4455-8863-bd2ad70a93e7 -project {DATAVERSE_BBOX_GOV_PROJECT_ID} --folder {file_path} -name {dataset_name} -type annotated_data -anno coco'

            # for QA
            command = f'conda run -n {conda_env} python tools/import_dataset_from_local.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {DATAVERSE_PASSWORD} -s 2bd928e5-a98f-4aae-a093-8545c57c103f  -project {DATAVERSE_BBOX_QA_PROJECT_ID} --folder {file_path} -name {dataset_name} -type annotated_data -anno coco'

            # execute(file_path, command)
            check_and_execute(file_path, command)

        print(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)


if __name__ == "__main__":
    fire.Fire(main)