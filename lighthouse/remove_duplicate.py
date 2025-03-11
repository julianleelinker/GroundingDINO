import json
import pathlib
import tqdm


DEPART_LIST = [
    'China_Steel',
    'Mass_Rapid_Transit',
    'Ports_Corporation',
    'Public_Works',
    'Sports_Development',
    'Taiwan_Power',
    'Transportation',
    'Water_Resources',
]


def remove_duplicated(dataverse_path, rededuplicated_json, old_anno_name='new_anno_0306.json', new_anno_name='anno_0311.json', dry_run=True, verbose=True):
    anno_path = dataverse_path / 'annotations'
    anno_jsons = list(anno_path.rglob(old_anno_name))
    anno_data = []
    for anno_json in anno_jsons:
        with open(anno_json, 'r') as f:
            data = json.load(f)
        anno_data.extend(data)

    revised_annos = []
    image_root = dataverse_path / 'images'

    with open(rededuplicated_json, 'r') as f:
        image_path = json.load(f)

    # temp
    rede_image_set = set()
    for image in image_path:
        src_file = pathlib.Path(image).resolve()
        for depart in DEPART_LIST:
            if depart in str(src_file):
                break
        folder_list = str(src_file).split(f'/{depart}/')
        dst_name = ('-').join(folder_list[1:]).replace('/', '-')
        dst_name = ('.').join(dst_name.split('.')[:-1]) + '.' + dst_name.split('.')[-1].lower()
        rede_image_set.add(dst_name)
    

    # rede_image_set = {pathlib.Path(image).stem for image in image_path}
    print(f'{len(rede_image_set)=}')

    removed_count, new_count = 0, 0
    for anno in tqdm.tqdm(anno_data):
        # image_stem = ('.').join(anno['image'].split('.')[:-1])
        
        if anno['image'] in rede_image_set:
            revised_annos.append(anno)
            new_count += 1
            continue

        removed_count += 1
        if verbose:
            print(f'found duplicated {anno["image"]}')
        if (image_root / anno['image']).exists() and not dry_run:
            (image_root / anno['image']).unlink()

    if dry_run:
        print(f'dry run, removed {removed_count} images from {dataverse_path}')
        print(f'dry run, new count {new_count} images from {dataverse_path}')
    else:
        revised_anno_path = dataverse_path / 'annotations' / new_anno_name
        with open(revised_anno_path, 'w') as f:
            json.dump(revised_annos, f, ensure_ascii=False, indent=4)
        print(f'removed {removed_count} images from {dataverse_path}')
        print(f'new count {new_count} images from {dataverse_path}')


def check_json(old_json, rede_json):
    with open(old_json, 'r') as f:
        old_data = json.load(f)
    with open(rede_json, 'r') as f:
        rede_data = json.load(f)
    rede_set = set(rede_data)
    count = 0
    for anno in old_data:
        if anno['image_path'] not in rede_set:
            count += 1
    import ipdb; ipdb.set_trace()
    print(f'found {count} annotations not in rededuplicated json')


if __name__ == '__main__':
    dataverse_path = '/mnt/data-home/mobility-multimodal/vlm-annotations/Sports_Development/20250109/Sports_Development_20250109_llava-onevision-0.5b-full/'
    rededuplicated_json = '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95_rededuplicate.json'
    # rededuplicated_json = '/mnt/data-home/mobility-multimodal/data-curation/Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95_full.json'
    # dataverse_path_prefix = '/mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part'

    # dataverse_path_list = []
    # for i in range(1, 6):
    #     for j in range(1, 5):
    #         dataverse_path_list.append(f'{dataverse_path_prefix}{i}_{j}/')
    # i = 6
    # for j in range(1, 4):
    #     dataverse_path_list.append(f'{dataverse_path_prefix}{i}_{j}/')

    # for dataverse_path in dataverse_path_list:
    #     # remove_duplicated(pathlib.Path(dataverse_path), rededuplicated_json, verbose=False, dry_run=False)
    #     remove_duplicated(pathlib.Path(dataverse_path), rededuplicated_json, verbose=False, dry_run=True)

    curation_prefix = '/mnt/data-home/mobility-multimodal/data-curation/'
    vlm_prefix = '/mnt/data-home/mobility-multimodal/vlm-annotations'

    path_map = {
        # "Sports_Development/20241223/Sports_Development_20241223_curated_t7-revised":           'Sports_Development/20241223/Sports_Development_20241223_curated_new.json',
        # "Sports_Development/20241223/Sports_Development_20241223_curated_t5_VLA_patch-revised": 'Sports_Development/20241223/Sports_Development_20241223_curated_new.json',
        # "Transportation/20250109/Transportation_20250109_curated_t7-revised":                   'Transportation/20250109/Transportation_20250109_curated_t7_new.json',
        # "Transportation/20241230/Transportation_20241230_curated_t7-revised":                   'Transportation/20241230/Transportation_20241230_curated_t7_new.json',           
        # "Public_Works/20241230/Public_Works_20241230_curated_t5_part-revised":                  'Public_Works/20241230/Public_Works_20241230_curated_t5_part_new.json',
        "Water_Resources/20250106/Water_Resources_20250106_curated_t5-revised":                   'Water_Resources/20250106/Water_Resources_20250106_curated_t5_new.json'
    }
    for anno_path, rededu_path in path_map.items():
        remove_duplicated(pathlib.Path(f'{vlm_prefix}/{anno_path}'), f'{curation_prefix}/{rededu_path}', new_anno_name='new_anno_0306.json', verbose=False, dry_run=False)

    # json_map = {key.replace('-revised', '.json'): value for key, value in path_map.items()}
    # for json_old, json_rede in json_map.items():
    #     check_json(pathlib.Path(f'{curation_prefix}/{json_old}'), f'{curation_prefix}/{json_rede}')


#"Sports_Development/20241223/Sports_Development_20241223_curated_t7.json":                 'Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95_rededuplicate.json',
#"Sports_Development/20241223/Sports_Development_20241223_curated_t5_VLA_patch.json":       'Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95_rededuplicate.json',
#"Transportation/20250109/Transportation_20250109_curated_t7.json":                         'Transportation/20250109/Transportation_20250109_image_list_keep_0.95_rededuplicate.json',
#"Transportation/20241230/Transportation_20241230_curated_t7.json":                         'Transportation/20241230/Transportation_20241230_image_list_keep_0.95_rededuplicate.json',
#"Public_Works/20241230/Public_Works_20241230_curated_t5_part.json":                        'Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95_rededuplicate.json',
