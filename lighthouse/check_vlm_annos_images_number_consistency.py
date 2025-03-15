import pathlib
import json
from common import VLM_CKPT2_FOLDERS


folder_list = VLM_CKPT2_FOLDERS

for folder in folder_list:
    images_path = pathlib.Path(folder) / 'images'
    if not images_path.exists():
        print(f'"{images_path}" not found')
        continue
    image_num = len(list(images_path.glob('*')))
    anno_path = pathlib.Path(folder) / 'annotations' / 'vlm_annotations.json'
    if not anno_path.exists():
        print(f'"{anno_path}" not found')
        continue
    with anno_path.open() as f:
        anno_num = len(json.load(f))
    if image_num != anno_num: 
        print(f"{folder} found {image_num} images while {anno_num} annotations")
