import pathlib
import subprocess
import os
from common import DEPART_MAP, VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, VLM_ANNOTATION_ROOT, DATAVERSE_PASSWORD, DATAVERSE_CKPT1_PROJECT_ID, DATAVERSE_CKPT2_PROJECT_ID


# DATAVERSE_PASSWORD = os.environ.get('DATAVERSE_PASSWORD')
# PROJECT_ID_CKPT1 = 230 
# PROJECT_ID_CKPT2 = 464 
CKPT_PROJECT_ID = 122

folder_list = VLM_CKPT2_FOLDERS
# project_id = DATAVERSE_CKPT2_PROJECT_ID
project_id = CKPT_PROJECT_ID
CKPT_SERVICE_ID = "697aa90b-00d0-4455-8863-bd2ad70a93e7"

# folder_list =[
#     'Sports_Development/20241223/Sports_Development_20241223_curated_t4-revised',
# ]
# folder_list = [pathlib.Path(f'{VLM_ANNOTATION_ROOT}/{folder}') for folder in folder_list]
# folder_list = [
#     "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part6_1"
# ]
# folder_list = [pathlib.Path(folder) for folder in folder_list]
folder_list = VLM_CKPT1_FOLDERS
# folder_list = VLM_CKPT1_FOLDERS + VLM_CKPT2_FOLDERS
# "/mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part5_2'"
# "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part4_1"
folder_list = [pathlib.Path(str(x).replace("/vlm-annotations/", "/checkpoint/vlm/hand/")) for x in folder_list]
print(len(folder_list))
folder_list = [x for x in folder_list if str(x) == "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part6_1"]
print(len(folder_list))
folder_list = [x for x in folder_list if x.exists()]
print(len(folder_list))

remaining_list = [
    # ckpt2
    "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Water_Resources/20250213/Water_Resources_20250213_curated_t1",
    # ckpt1 remaining
    "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Water_Resources/20250106/Water_Resources_20250106_curated_t8_VLM_100000_patch",
    "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Transportation/20250120/Transportation_20250120_llava-onevision-0.5b-full",
    "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Sports_Development/20241223/Sports_Development_20241223_curated_t6_VLM_100000_patch",
    # patch ckpt2
    "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand/Transportation/20250115/Transportation_20250115_curated_t4"
]
folder_list = [pathlib.Path(x) for x in remaining_list]


# check folder exit
for folder in folder_list:
    assert pathlib.Path(folder).exists(), f"{folder} not exist"

count = 0
iii = 0
print(len(folder_list))
for folder in folder_list[iii:]:
    print(f"xxxxxxxxxxxxxxx={iii}")
    # if not (folder / 'done').exists():
    #     continue
    # if (folder / 'uploaded').exists():
    #     continue
    # token_list = str(folder).split('/vlm-annotations/')[1].split('/')
    # for ckpt
    token_list = str(folder).split('/checkpoint/vlm/hand/')[1].split('/')
    depart = token_list[0]
    date = token_list[1]
    subfolder = token_list[-1]

    dataset_name = f"{DEPART_MAP[depart]}_{date}-{subfolder}"
    print(folder)
    print(f'{dataset_name=}\n')
    # continue

    command = [
        "conda", "run", "-n", "dataverse-sdk", "python", "tools/import_vqa.py",
        "-host", "https://visionai.linkervision.ai/dataverse/curation",
        "-e", "julianlee@linkervision.com",
        "-p", DATAVERSE_PASSWORD,
        # "-s", "2bd928e5-a98f-4aae-a093-8545c57c103f",
        "-s", CKPT_SERVICE_ID,
        "-project", str(project_id),
        "--folder", folder,
        "-name", dataset_name
    ]
    print(command)
    print('\n')
    subprocess.run(command, check=True)
    (folder / 'uploaded').touch()
    count += 1
    iii += 1
print(f'total {count} data upload')
