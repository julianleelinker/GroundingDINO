import pathlib
import tqdm
import json
import os
from PIL import Image


image_name_list = [
    "31_152805497_s480.jpg",
    "NO.72_B2F(D通道口)_2025_2_14 下午 (UTC+08_00) 04_49_59_s2780.jpg",
    "24_203352778_s360.jpg",
    "24_014414563_s110.jpg",
    "Public_Works-20241230-video-S11CC-B8DF6B001459-20241218-video_S11CC-B8DF6B001459_20241218_20241218071130_20241218073305_s790.jpg",
]

data_root = pathlib.Path("/mnt/lighthouseACD/QAed-data/bbox/hand0526")
image_file_list = list(data_root.rglob("*.jpg"))
image_file_dict = {x.name: x for x in image_file_list}
target_image_dict = {}
for image_name in tqdm.tqdm(image_name_list):
    if image_name not in image_file_dict:
        print(f"Image {image_name} not found in {data_root}")
    else:
        target_image_dict[image_name] = image_file_dict[image_name]


output_root = pathlib.Path("threshold_test")
output_images_folder = output_root / "images"
output_images_folder.mkdir(parents=True, exist_ok=True)
os.chmod(output_images_folder, 0o777)
output_annotations_folder = output_root / "annotations"
output_annotations_folder.mkdir(parents=True, exist_ok=True)
os.chmod(output_annotations_folder, 0o777)


test_json = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "object"}]
}
for new_id, (image_name, image_path) in enumerate(target_image_dict.items()):
    json_path = image_path.parent.parent / "annotations" / "labels.json"
    with open(json_path, "r") as f:
        anno_data = json.load(f)
    for image_anno in tqdm.tqdm(anno_data["images"]):
        if image_anno["file_name"] == image_name:
            old_id = image_anno["id"]
            image_anno["id"] = new_id
            image_anno.pop('coco_url')
            test_json["images"].append(image_anno)
            break
    for anno_anno in tqdm.tqdm(anno_data["annotations"]):
        if anno_anno["image_id"] == old_id:
            anno_anno["id"] = len(test_json["annotations"])
            anno_anno["image_id"] = new_id
            anno_anno["category_id"] = 1
            test_json["annotations"].append(anno_anno)
with open(output_annotations_folder / "labels.json", "w") as f:
    json.dump(test_json, f, indent=4, ensure_ascii=False)
for image_name, image_path in target_image_dict.items():
    new_image_path = output_images_folder / image_name
    image_pil = Image.open(image_path).convert("RGB")
    image_pil.save(new_image_path)
    # image_pil = image_path.open("rb").read()
    # with open(new_image_path, "wb") as f:
    #     f.write(image_pil)
print(f"Saved test dataset with {len(test_json['images'])} images and {len(test_json['annotations'])} annotations.")
import ipdb; ipdb.set_trace()
