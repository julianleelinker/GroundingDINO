import os
import pathlib
from common import AUGMENTED_CURATED_NEW_JSONS, DINO_COCO_TARGET_ROOT, copy_images_in_json


for json_path in AUGMENTED_CURATED_NEW_JSONS:
    dst = pathlib.Path(DINO_COCO_TARGET_ROOT) / json_path.stem
    dst.mkdir(exist_ok=True, parents=True)
    os.chmod(dst, 0o777)
    print(dst)
    copy_images_in_json(json_path, dst, is_image_list=True, split_size=5000)