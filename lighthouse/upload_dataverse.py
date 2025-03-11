import pathlib
import subprocess
import os
from common import DEPART_MAP, CKPT1_VLM_FOLDERS


DATAVERSE_PASSWORD = os.environ.get('DATAVERSE_PASSWORD')

# check folder exit
for folder in CKPT1_VLM_FOLDERS:
    assert pathlib.Path(folder).exists(), f"{folder} not exist"

for folder in CKPT1_VLM_FOLDERS:
    token_list = folder.split('/vlm-annotations/')[1].split('/')
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
        "-project", "230",
        "--folder", folder,
        "-name", dataset_name
    ]
    print(command)
    print('\n')
    subprocess.run(command, check=True)
