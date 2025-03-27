import pathlib
import json
from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS


folder_list = VLM_CKPT1_FOLDERS

folder_list = [pathlib.Path(str(x).replace("/vlm-annotations/", "/checkpoint/vlm/hand/")) for x in folder_list]
for x in folder_list:
    if not x.exists():
        print(f"{x} not exist")

image_count = 0
anno_count = 0
for folder in folder_list:
    images_path = pathlib.Path(folder) / 'images'
    if not images_path.exists():
        print(f'"{images_path}" not found')
        continue
    image_num = len(list(images_path.glob('*')))
    print(images_path)
    image_count += image_num
    # anno_path = pathlib.Path(folder) / 'annotations' / 'vlm_annotations.json'
    anno_path = pathlib.Path(folder) / 'annotations' / 'vlm_annotation.json'
    if not anno_path.exists():
        print(f'"{anno_path}" not found')
        continue
    with anno_path.open() as f:
        anno_num = len(json.load(f))
    anno_count += anno_num
    if image_num != anno_num: 
        print(f"{folder} found {image_num} images while {anno_num} annotations")
import ipdb; ipdb.set_trace()
