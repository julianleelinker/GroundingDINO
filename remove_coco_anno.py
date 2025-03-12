import pathlib
import json

coco_root ='/mnt/lighthouseACD/ACD-gdino-COCO'
dataverse_list =[
    'Public_Works_20250106_image_list_keep_0.95',
    'Sports_Development_20250109_image_list_keep_0.95',
    'Taiwan_Power_20250106_image_list_keep_0.95_0.30_0.35',
    'Water_Resources_20250106_image_list_keep_0.95',
]
revision_list = [
    'Mass_Rapid_Transit_20250109_image_list_keep_0.95_0.30_0.35',
    'Public_Works_20241230_image_list_keep_0.95',
    # below seems no union
    'Sports_Development_20241223_image_list_keep_0.95',
    'Transportation_20241230_image_list_keep_0.95',
    'Transportation_20250109_image_list_keep_0.95',
]
revision_list = [f'{coco_root}/{x}' for x in revision_list]
revision_json_prefix = '/mnt/data-home/mobility-multimodal/data-curation'
revision_json = [
    'Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95_rededuplicate.json',
    'Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95_rededuplicate.json',
    'Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95_rededuplicate.json',
    'Transportation/20241230/Transportation_20241230_image_list_keep_0.95_rededuplicate.json',
    'Transportation/20250109/Transportation_20250109_image_list_keep_0.95_rededuplicate.json',
]
revision_json = [f'{revision_json_prefix}/{x}' for x in revision_json]
ok_list = [
    'Transportation_20250115_image_list_keep_0.95_rededuplicate',
    'Ports_Corporation_20250124_image_list_keep_0.95',
]

# def remove_duplicate(split_root, image_list, dry_run=True):
splits = list(pathlib.Path(revision_list[0]).glob('split*'))

json_file = revision_json[0]
split = splits[0]

def remove_coco_duplicated(split, json_file, dry_run=True):
    image_list = list((split / 'images').glob('*'))
    anno_path = (split / 'annotations' / 'labels.json')
    with open(anno_path) as f:
        anno_data = json.load(f)
    remove_count = 0
    keep_count = 0
    
    new_images_annos = []
    keep_images_ids = []
    for anno in anno_data['images']:
        if anno['file_name'] in image_list:
            new_images_annos.append(anno)
            keep_images_ids.append(anno['id'])
            keep_count += 1
        else:
            remove_count += 1
    
    new_annotations_annos = []
    for anno in anno_data['annotations']:
        if anno['image_id'] in keep_images_ids:
            new_annotations_annos.append(anno)
    
    anno_data['images'] = new_images_annos
    anno_data['annotations'] = new_annotations_annos

import ipdb; ipdb.set_trace()