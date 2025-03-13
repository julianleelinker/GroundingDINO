import pathlib
import json
import tqdm

coco_root ='/mnt/lighthouseACD/ACD-gdino-COCO'
dataverse_list =[
    'Public_Works_20250106_image_list_keep_0.95',
    'Sports_Development_20250109_image_list_keep_0.95',
    'Taiwan_Power_20250106_image_list_keep_0.95_0.30_0.35',
    'Water_Resources_20250106_image_list_keep_0.95',
]
revision_list = [
    # 'Mass_Rapid_Transit_20250109_image_list_keep_0.95_0.30_0.35', # done upload
    'Public_Works_20241230_image_list_keep_0.95',

    # below seems no union
    # 'Sports_Development_20241223_image_list_keep_0.95', # done remove
    # 'Transportation_20241230_image_list_keep_0.95', # done remove
    # 'Transportation_20250109_image_list_keep_0.95', # done remove
]
revision_list = [f'{coco_root}/{x}' for x in revision_list]
revision_json_prefix = '/mnt/data-home/mobility-multimodal/data-curation'
revision_json = [
    # 'Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95_rededuplicate.json',
    'Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95_rededuplicate.json',

    # 'Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95_rededuplicate.json', # done remove
    # 'Transportation/20241230/Transportation_20241230_image_list_keep_0.95_rededuplicate.json', # done remove
    # 'Transportation/20250109/Transportation_20250109_image_list_keep_0.95_rededuplicate.json',  # done remove
]
revision_json = [f'{revision_json_prefix}/{x}' for x in revision_json]
ok_list = [
    'Transportation_20250115_image_list_keep_0.95_rededuplicate',
    'Ports_Corporation_20250124_image_list_keep_0.95',
]


def remove_coco_duplicated(split, image_list, verbose=True, dry_run=True):
    print(split)
    
    anno_path = (split / 'annotations' / 'labels.json')
    image_root = (split / 'images')
    with open(anno_path) as f:
        try:
            anno_data = json.load(f)
        except json.decoder.JSONDecodeError:
            raise

    remove_count = 0
    keep_count = 0
    
    new_images_annos = []
    keep_images_ids = []
    image_set = set(image_list)
    iterable = anno_data['images']

    if verbose:
        print('checking images')
        print('image path sample')
        print(anno_data['images'][0]['file_name'])
        print(image_list[0])
        iterable = tqdm.tqdm(iterable)

    for anno in iterable:
        if anno['file_name'] in image_set:
            new_images_annos.append(anno)
            keep_images_ids.append(anno['id'])
            keep_count += 1
        else:
            remove_count += 1
            image_path = image_root / anno['file_name']
            if image_path.exists():
                if not dry_run:
                    image_path.unlink()
    
    print('checking annotations')
    new_annotations_annos = []
    for anno in tqdm.tqdm(anno_data['annotations']):
        if anno['image_id'] in keep_images_ids:
            new_annotations_annos.append(anno)
    
    anno_data['images'] = new_images_annos
    anno_data['annotations'] = new_annotations_annos

    print(f'{remove_count=}, {keep_count=}\n')
    if not dry_run:
        with open(anno_path, 'w') as f:
            json.dump(anno_data, f)
    # import ipdb; ipdb.set_trace()
    return keep_count, remove_count


all_keep, all_removed = 0, 0
failed_split = []
for revision_folder, json_file in zip(revision_list, revision_json): 
    splits = list(pathlib.Path(revision_folder).glob('split*'))

    with open(json_file) as f:
        image_list = json.load(f)
    # image_list = [pathlib.Path(x).name for x in image_list]
    image_list = [
        ('-').join(x.split('/')[3:]) for x in image_list
    ]
    print(image_list[0])

    for split in tqdm.tqdm(splits):
        try:
            keep, remove = remove_coco_duplicated(split, image_list, verbose=False, dry_run=True)
            all_keep += keep
            all_removed += remove
        except json.decoder.JSONDecodeError:
            print(f'Error processing {split}')
            failed_split.append(split)
    print(f'{all_keep=}')
    print(f'{all_removed=}')

for split in failed_split:
    print(f'failed split: {split}')