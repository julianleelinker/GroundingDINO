import pathlib
import json
import tqdm


def change_image_id(data_folder, start_id, dry=True):
    annotation_root = pathlib.Path(data_folder) / 'annotations'
    json_path_list = sorted(annotation_root.rglob('*.json'))
    i = start_id
    for json_path in json_path_list:
        with json_path.open('r') as f:
            json_data = json.load(f)
        for data in tqdm.tqdm(json_data):
            if not dry:
                data['id'] = i
            i += 1
        if not dry:
            with json_path.open('w') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f'number of annotations: {i - start_id}')
    image_root = pathlib.Path(data_folder) / 'images'
    file_list = list(image_root.rglob('*'))
    print(f'number of files in images: {len(file_list)}')
    return i


if __name__ == "__main__":
    depart_folder_list = [
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
    depart_folder_list = [f'{root_path}/{folder}' for folder in depart_folder_list]


    linker_data_prefix = '/mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part'
    linker_folder_list = []
    for i in range(1, 6):
        for j in range(1, 5):
            linker_folder_list.append(f'{linker_data_prefix}{i}_{j}/')
    i = 6
    for j in range(1, 4):
        linker_folder_list.append(f'{linker_data_prefix}{i}_{j}/')

    # folder_list, start_id = depart_folder_list, 1
    folder_list, start_id = linker_folder_list, 43635
    #next 99920
    for folder in folder_list:
        start_id = change_image_id(folder, start_id, dry=False)
        print(f'{folder=}')
        print(f'next start_id={start_id}')