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
JSON_LIST_JAN = [
    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20241230/Transportation_20241230_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250109/Transportation_20250109_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95.json',
    # below include too much images
    # '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Kaohsiung_Data_V2/Kaohsiung_Data_V2_image_list_keep_0.95.json',
]

JSON_LIST_JAN = [
    # '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250109/Transportation_20250109_image_list_keep_0.95.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95.json',
]

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

REMOTE_HOST = 'julian@10.1.24.90'


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
        print(f'{len(image_list)=}')
        for key, val in SOURCES_BY_DEPART.items():
            if key in json_path:
                SOURCES_BY_DEPART[key] += len(image_list)
                # count += len(image_list)
                break
    print('images numbers from different sources:')
    for key, val in SOURCES_BY_DEPART.items():
        print(f'{key}, {val}')
    print('continue to copy?')

    
    for json_path in JSON_LIST_JAN:
        print(json_path)
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        image_path_list = [pathlib.Path(data) for data in json_data]
        # if len(image_path_list) > sample_size:
        #     image_path_list = random.sample(image_path_list, sample_size)
        output_folder = pathlib.Path(output_root) / pathlib.Path(json_path).stem
        output_folder.mkdir(parents=True, exist_ok=True)

        files_per_folder = 5000
        for i in range(1, (len(image_path_list)-1)//files_per_folder + 2):
            (output_folder / f'split{i}').mkdir(parents=True, exist_ok=True)
            # os.makedirs(os.path.join(dest_dir, f"split_{i}"), exist_ok=True)
    
        # for index, filename in enumerate(image_path_list):
        #     # Determine the folder index (round-robin)

        #     src_path = os.path.join(source_dir, filename)
        #     dest_path = os.path.join(dest_dir, f"split_{folder_index}", filename)


        duplicated_count = 0
        for index, remote_path in tqdm.tqdm(enumerate(image_path_list), total=len(image_path_list)):
            image_name = get_concat_file_name(remote_path, ROOT_LIST)
            folder_index = (index // files_per_folder) + 1
            local_path = output_folder / f'split{folder_index}' / image_name
            if not local_path.exists():
                command = f"rsync -avz {REMOTE_HOST}:{remote_path} {local_path}"
                subprocess.run(["rsync", "-avz", f"{REMOTE_HOST}:{remote_path}", local_path], check=False)
            else:
                duplicated_count += 1
                print(f'{remote_path} already exist, skip')
        print(f'{duplicated_count=}')
        print(f'file saved to {str(output_folder)}')


# Water_Resources_20250106_image_list_keep_0.95 many
# Transportation_20250109_image_list_keep_0.95 
# Sports_Development_20241223_image_list_keep_0.95