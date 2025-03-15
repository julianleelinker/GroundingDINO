import json
import pathlib
import tqdm
from collections import defaultdict
from common import VLM_CKPT1_FOLDERS

final_json_path = '/mnt/data-home/chungan/curation/checkpoint1_image_list_new.json'

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
        image_path_list = json.load(f)

    rede_image_dict = defaultdict(set)
    for image_path in image_path_list:
        image_path = pathlib.Path(image_path)
        rede_image_dict[image_path.parent].add(image_path.name)

    removed_count, new_count = 0, 0
    # import ipdb; ipdb.set_trace()
    for anno in tqdm.tqdm(anno_data):
        if anno['image'] in rede_image_dict[image_root]:
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
    return new_count



import ipdb; ipdb.set_trace()
count = 0
for folder in VLM_CKPT1_FOLDERS:
    assert pathlib.Path(folder).exists(), f'{folder} not exists'
    # count += remove_duplicated(pathlib.Path(folder), final_json_path, old_anno_name='new_anno_0306.json', new_anno_name='new_anno_0306.json', verbose=False, dry_run=True)
    # count += remove_duplicated(pathlib.Path(folder), final_json_path, old_anno_name='vlm_annotations_*', new_anno_name='new_anno_0306.json', verbose=False, dry_run=False)
    # count += remove_duplicated(pathlib.Path(folder), final_json_path, old_anno_name='revised_anno.json', new_anno_name='new_anno_0306.json', verbose=False, dry_run=False)
print(f'total {count} images')