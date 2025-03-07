import pathlib
import subprocess
import os


DATAVERSE_PASSWORD = os.environ.get('DATAVERSE_PASSWORD')

depart_map = {
    "Sports_Development": "運發局",
    "Water_Resources":    "水利局",
    "Transportation":     "交通局", 
    "Mass_Rapid_Transit": "捷運局", 
    "Taiwan_Power":       "台電",  
    "Public_Works":       "工務局", 
    "China_Steel":        "中鋼",
    "Ports_Corporation":  "港務局",
}
folder_list = [
    "Sports_Development/20241223/Sports_Development_20241223_curated_t7-revised",
    "Sports_Development/20241223/Sports_Development_20241223_curated_t5_VLA_makeup-revised",
    "Sports_Development/20250109/Sports_Development_20250109_llava-onevision-0.5b-full-revised",
    "Water_Resources/20250106/Water_Resources_20250106_curated_t5-revised",
    "Transportation/20250109/Transportation_20250109_curated_t7-revised",
    "Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95-revised",
    "Transportation/20241230/Transportation_20241230_curated_t7-revised",
    "Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95-revised",
    "Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95-revised",
    "Public_Works/20241230/Public_Works_20241230_curated_t5_part-revised",
]
root_path = "/mnt/data-home/mobility-multimodal/vlm-annotations"
folder_list = [f'{root_path}/{folder}' for folder in folder_list]

# check folder exit
for folder in folder_list:
    assert pathlib.Path(folder).exists(), f"{folder} not exist"

# command = f'conda run -n dataverse-sdk python tools/import_dataset_from_local.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {PASSWORD} -s 2bd928e5-a98f-4aae-a093-8545c57c103f  -project 225 --folder {file_path} -name {file_path.parent.name}/{file_path.name} -type annotated_data -anno coco'

for folder in folder_list:
    token_list = folder.split('/vlm-annotations/')[1].split('/')
    depart = token_list[0]
    date = token_list[1]
    subfolder = token_list[-1]
    dataset_name = f"{depart_map[depart]}_{date}-{subfolder}"
    print(f'{dataset_name=}')
    # command = f'conda run -n dataverse-sdk python tools/import_vqa.py -host https://visionai.linkervision.ai/dataverse/curation -e  julianlee@linkervision.com -p {DATAVERSE_PASSWORD} -s 2bd928e5-a98f-4aae-a093-8545c57c103f -project 230 --folder {folder} -name {dataset_name}'

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
    subprocess.run(command, check=True)
