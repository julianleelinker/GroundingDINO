import os
import pathlib
import tqdm
from common import AUGMENTED_CURATED_RUNNING_JSONS, DINO_COCO_SOURCE_ROOT, copy_images_in_json


def main():
    json_list = AUGMENTED_CURATED_RUNNING_JSONS

    dst_root = pathlib.Path(DINO_COCO_SOURCE_ROOT)

    print("Copying images from json files to target folder...")
    for json_path in tqdm.tqdm(json_list):
        dst = dst_root / json_path.stem
        print(dst)
        dst.mkdir(exist_ok=True, parents=True)
        os.chmod(dst, 0o777)
        copy_images_in_json(json_path, dst, is_image_list=True, split_size=10000)


if __name__ == "__main__":
    main()