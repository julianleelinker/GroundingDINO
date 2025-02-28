import pathlib
import time
import subprocess
import os


PASSWORD = os.environ.get('DATAVERSE_PASSWORD')

def check_and_execute(data_path, command):
    label_path = pathlib.Path(data_path) / 'annotations' / 'labels.json'
    upload = pathlib.Path(data_path) / 'uploaded'
    if label_path.exists() and not upload.exists():
        print(f"COCO {data_path} exists and not uploaded, Running command:")
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
    elif label_path.exists() and upload.exists():
        print(f"COCO {data_path} exists and already uploaded")
    else:
        print(f"COCO {data_path} does not exist.")


if __name__ == "__main__":
    wait_time = 600
    print(PASSWORD)
    while True:
        file_root = pathlib.Path(f'/mnt/lighthouseACD/ACD-gdino-COCO/Public_Works_20241230_image_list_keep_0.95/')
        file_path_list = [p for p in file_root.glob('*') if p.is_dir()]
        for file_path in file_path_list:
            print(str(file_path))
            command = f'conda run -n lighthouse python tools/import_dataset_from_local.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {PASSWORD} -s 2bd928e5-a98f-4aae-a093-8545c57c103f  -project 225 --folder {file_path} -name {file_path.parent.name}/{file_path.name} -type annotated_data -anno coco'
            check_and_execute(file_path, command)
        print(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)
