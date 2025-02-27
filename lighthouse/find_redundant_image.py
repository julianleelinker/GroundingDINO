import pathlib
import json

data_path = pathlib.Path('/mnt/data-home/mobility-multimodal/vlm-annotations/Public_Works/20241230/Public_Works_20241230_curated_t5_part')
anno_root = data_path / 'annotations'
anno_list = sorted(anno_root.rglob('*.json'))

all_images = set()
for anno in anno_list:
    with open(anno, 'r') as f:
        anno_data = json.load(f)
    for anno in anno_data:
        image_name = anno['image']
        all_images.add(image_name)

image_path_list = list((data_path/'images').rglob('*.jpg'))
for image_path in image_path_list:
    if image_path.name in all_images:
        continue
    print(image_path)
import ipdb; ipdb.set_trace()