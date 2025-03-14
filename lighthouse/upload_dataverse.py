import pathlib
import subprocess
import os
from common import DEPART_MAP, VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS


DATAVERSE_PASSWORD = os.environ.get('DATAVERSE_PASSWORD')
PROJECT_ID_CKPT1 = 230 
PROJECT_ID_CKPT2 = 464 

folder_list = VLM_CKPT2_FOLDERS
project_id = PROJECT_ID_CKPT2

# check folder exit
for folder in folder_list:
    assert pathlib.Path(folder).exists(), f"{folder} not exist"

for folder in folder_list:
    if not (folder / 'done').exists():
        continue
    if (folder / 'uploaded').exists():
        continue
    token_list = str(folder).split('/vlm-annotations/')[1].split('/')
    depart = token_list[0]
    date = token_list[1]
    subfolder = token_list[-1]
    dataset_name = f"{DEPART_MAP[depart]}_{date}-{subfolder}"
    print(f'{dataset_name=}')

    command = [
        "conda", "run", "-n", "dataverse-sdk", "python", "tools/import_vqa.py",
        "-host", "https://visionai.linkervision.ai/dataverse/curation",
        "-e", "julianlee@linkervision.com",
        "-p", DATAVERSE_PASSWORD,
        "-s", "2bd928e5-a98f-4aae-a093-8545c57c103f",
        "-project", str(project_id),
        "--folder", folder,
        "-name", dataset_name
    ]
    print(command)
    print('\n')
    subprocess.run(command, check=True)
    (folder / 'uploaded').touch()
