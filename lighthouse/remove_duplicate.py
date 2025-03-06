import json
import pathlib

# load new anno.json , save to revised anno
new_anno_list = [
    # only this two use this path to do vlm annotations
    '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95_rededuplicate.json',
    '/mnt/data-home/mobility-multimodal/data-curation/Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95_rededuplicate.json',
    # 
    # '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95_rededuplicate.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95_rededuplicate.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95_rededuplicate.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20241230/Transportation_20241230_image_list_keep_0.95_rededuplicate.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250109/Transportation_20250109_image_list_keep_0.95_rededuplicate.json',
    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250115/Transportation_20250115_image_list_keep_0.95_rededuplicate.json',
]

# TODO deal with this tomorrow after gpt done load vlm_annoataions.json, save to revised anno
vlm_anno_list = [
    '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95_rededuplicate.json',
]


for image_path_file in new_anno_list:
    with open(image_path_file, 'r') as f:
        image_path = json.load(f)
    image_set = {pathlib.Path(image).stem for image in image_path}
    data_path = image_path_file.replace('data-curation', 'vlm-annotations').replace('image_list_keep_0.95_rededuplicate.json', 'image_list_keep_0.95')
    data_path = pathlib.Path(data_path)
    anno_path = data_path / 'annotations'
    anno_jsons = list(anno_path.rglob('new_anno_0306.json'))
    annos = []
    for anno_json in anno_jsons:
        with open(anno_json, 'r') as f:
            data = json.load(f)
        annos.extend(data)

    revised_annos = []
    image_root = data_path / 'images'
    for anno in annos:
        image_stem = anno['image'].split('.')[0]
        if image_stem in image_set:
            revised_annos.append(anno)
            continue
        print(anno['image'])
        if (image_root / anno['image']).exists():
            print(f'removing {anno["image"]}')
            (image_root / anno['image']).unlink()

    revised_anno_path = data_path / 'annotations' / f'revised_anno.json'
    with open(revised_anno_path, 'w') as f:
        json.dump(revised_annos, f, ensure_ascii=False, indent=4)


 