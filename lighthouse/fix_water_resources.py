import json
import pathlib
import os


anno_path = '/mnt/data-home/mobility-multimodal/vlm-annotations/Water_Resources/20250106/Water_Resources_20250106_curated_t5/annotations'
image_json = '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_curated_t5.json'
newoutput = pathlib.Path('water_output')
newoutput.mkdir(parents=True, exist_ok=True)
os.chmod(newoutput, 0o777)
output_images_root = newoutput / 'images'
output_anno_root = newoutput / 'annotations'
output_images_root.mkdir(parents=True, exist_ok=True)
os.chmod(output_images_root, 0o777)
output_anno_root.mkdir(parents=True, exist_ok=True)
os.chmod(output_anno_root, 0o777)


with open(image_json) as f:
    image_path_list = json.load(f)
image_path_list = [pathlib.Path(data['image_path']) for data in image_path_list]
# chcek if image path exist
for image_path in image_path_list:
    if not image_path.exists():
        print(f'{image_path} does not exist')

image_path_set = set()
count = 0
for image_path in image_path_list:
    if image_path.name in image_path_set:
        print(f'{image_path} is duplicated')
        count += 1
    else:
        image_path_set.add(image_path.name)
import ipdb; ipdb.set_trace()


anno_file_list = pathlib.Path(anno_path).rglob('*.json')
annos = []
for anno_file in anno_file_list:
    with anno_file.open() as f:
        anno_data = json.load(f)
    annos.extend(anno_data)

tmp_set = set()
for anno in annos:
    tmp_set.add(anno['image'])

import ipdb; ipdb.set_trace()
