import os
import tqdm
import pathlib


JSON_LIST_JAN = [
    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20241230/Transportation_20241230_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250109/Transportation_20250109_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95.json',
    # below include too much images
    '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Kaohsiung_Data_V2/Kaohsiung_Data_V2_image_list_keep_0.95.json',
]
REMOTE_HOST = 'julian@10.1.24.90'
for remote_path in tqdm.tqdm(JSON_LIST_JAN):
    pathlib.Path(remote_path).parent.mkdir(exist_ok=True, parents=True)
    local_path = remote_path
    command = f"rsync -avz --checksum {REMOTE_HOST}:{remote_path} {local_path}"
    # subprocess.run(command, shell=True, check=True)
    os.system(command)