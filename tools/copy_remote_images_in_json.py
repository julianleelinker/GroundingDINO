# %%
import json
import pathlib
import tqdm
import subprocess


SOURCES_BY_DEPART = {
    'Public_Works': 0,
    'Water_Resources': 0,
    'Transportation': 0,
    'Mass_Rapid_Transit': 0,
    'Sports_Development': 0,
    'Taiwan_Power': 0,
    'Ports_Corporation': 0,
    'China_Steel': 0,
    'Linker_Vision_Data_V2': 0,
    'Kaohsiung_Data_V2': 0,
}
prefix = '/mnt/data-home/mobility-multimodal/data-curation'
JSON_LIST_JAN = [
    # 'Transportation/20241230/Transportation_20241230_image_list_keep_0.95.json',
    # 'Transportation/20250109/Transportation_20250109_image_list_keep_0.95.json',
    'Transportation/20250115/Transportation_20250115_image_list_keep_0.95_rededuplicate.json',
    'Transportation/20250120/Transportation_20250120_image_list_keep_0.95.json',

    # 'Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95.json',
    'Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95.json',
    'Public_Works/20250206/Public_Works_20250206_image_list_keep_0.95.json',

    # 'Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json',
    'Water_Resources/20250213/Water_Resources_20250213_image_list_keep_0.95.json',

    # 'Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95.json',

    # 'Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95.json',
    # 'Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95.json',

    # 'Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95.json',

    'Ports_Corporation/20250124/Ports_Corporation_20250124_image_list_keep_0.95.json'


    # below include too much images
    # 'Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95.json',
]

# JSON_LIST_JAN = [
    # '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250109/Transportation_20250109_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95.json',
# ]
JSON_LIST_JAN = [f'{prefix}/{json_path}' for json_path in JSON_LIST_JAN]

ROOT_LIST = {
    'Dec',
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
}


def get_concat_file_name(file_path, root_list):
    root = file_path.parent
    while root.name not in root_list:
        root = root.parent
    concat_name = str(file_path.relative_to(root)).replace('/', '-')
    return concat_name


if __name__ == '__main__':
    # sample_size = 10000
    output_root = '/mnt/data-home/julian/lighthouse/augmented-curated-data'


    for json_path in JSON_LIST_JAN:
        with open(json_path, 'r') as f:
            image_list = json.load(f)
        print(f'{json_path=}')
        print(f'{len(image_list)=}\n')