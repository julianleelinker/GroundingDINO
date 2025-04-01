import pathlib
import time
import subprocess
import os
import tqdm
from common import get_depart_date


PASSWORD = os.environ.get('DATAVERSE_PASSWORD')

def execute(data_path, command):
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


if __name__ == "__main__":
    print("Starting auto upload to dataverse")
    wait_time = 600
    # coco_root ='/mnt/lighthouseACD/ACD-gdino-COCO'
    coco_root = '/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand'
    file_path_list =[
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Transportation_20250109_image_list_keep_0.95/split15_0.30_0.35",
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split30_0.30_0.35",
        # p for p in pathlib.Path(coco_root).glob('*/*') if p.is_dir()
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Sports_Development_20241223_image_list_keep_0.95/split2_0.30_0.35",
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split18_0.30_0.35",
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split31_0.30_0.35",
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split49_0.30_0.35",
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split50_0.30_0.35",
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split51_0.30_0.35",
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split58_0.30_0.35",
        # "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split63_0.30_0.35",

        "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split21_0.30_0.35",
        "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand/Public_Works_20241230_image_list_keep_0.95/split55_0.30_0.35",
    ]
    file_path_list = [pathlib.Path(x) for x in file_path_list]
    import ipdb; ipdb.set_trace()
    while True:
        i = 0
        for file_path in tqdm.tqdm(file_path_list):
            print(f"xxxxxxxxxxxx {i}")
            i += 1
            # token = str(file_path).split('/checkpoint/bbox/hand/')[1].split('/')[0]
            # depart = token_list[0]
            # date = token_list[1]
            depart, date = get_depart_date(file_path, ch=True)
            dataset_name = f"{depart}_{date}"
            # for checkpoint
            command = f'conda run -n dataverse-sdk python tools/import_dataset_from_local.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {PASSWORD} -s 697aa90b-00d0-4455-8863-bd2ad70a93e7 -project 121 --folder {file_path} -name {dataset_name} -type annotated_data -anno coco'
            # command = f'conda run -n dataverse-sdk python tools/import_dataset_from_local.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {PASSWORD} -s 2bd928e5-a98f-4aae-a093-8545c57c103f  -project 225 --folder {file_path} -name {file_path.parent.name}/{file_path.name} -type annotated_data -anno coco'

            # command = [
            #     'conda', 'run', '-n' lighthouse python tools/import_dataset_from_local.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {PASSWORD} -s 2bd928e5-a98f-4aae-a093-8545c57c103f  -project 225 --folder {file_path} -name {file_path.parent.name}/{file_path.name} -type annotated_data -anno coco'
            # ]
            execute(file_path, command)
            # check_and_execute(file_path, command)
        print(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)
