import json
import pathlib
import os
import webdataset as wds
from PIL import Image
import tqdm
import io


def check_coco_caption_data(image_root, json_path):
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    image_list = list(pathlib.Path(image_root).glob("*.jpg"))
    image_name_set = set(image.name for image in image_list)
    print(f"{len(image_list)=}")
    for image in coco_data["images"]:
        assert image["file_name"] in image_name_set, f"Image {image['file_name']} not found in {image_root}"

    image_id_to_path_caption = {image["id"]: (f"{image_root}/{image['file_name']}", []) for image in coco_data["images"]}
    print(f"{len(image_id_to_path_caption)=}")
    for anno in coco_data["annotations"]:
        image_id_to_path_caption[anno["image_id"]][1].append(anno["caption"])
        assert anno["image_id"] in image_id_to_path_caption, f"Annotation {anno['id']} refers to non-existent image ID {anno['image_id']}"
    print(f"{len(coco_data['annotations'])=}")
    return image_id_to_path_caption

def save_coco_to_webdataset(val_id_to_path_caption, output_path, base_name="shard", caption_idx=0):
    """
    Save COCO validation data to WebDataset format.
    """
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    os.chmod(output_path, 0o777)
    max_per_shard = 1000
    pattern = os.path.join(output_path, f"{base_name}-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=max_per_shard) as sink:
        for (image_path, caption_text) in tqdm.tqdm(val_id_to_path_caption.values()):
            if not pathlib.Path(image_path).exists():
                print(f"Image {image_path} not found, skipping.")
                continue
            image_pil = Image.open(image_path)
            img_buffer = io.BytesIO()
            image_pil.convert("RGB").save(img_buffer, format="jpeg")
            img_bytes = img_buffer.getvalue()

            # Create sample
            sample = {
                "__key__": pathlib.Path(image_path).stem,
                "jpg": img_bytes,
                "txt": caption_text[caption_idx],
            }
            sink.write(sample)


val_image_root = "/mnt/data-home/julian/coco/val2017"
val_json_path = "/mnt/data-home/julian/coco/annotations/captions_val2017.json"
train_image_root = "/mnt/data-home/julian/coco/train2017"
train_json_path = "/mnt/data-home/julian/coco/annotations/captions_train2017.json"
val_id_to_path_caption = check_coco_caption_data(val_image_root, val_json_path)
train_id_to_path_caption = check_coco_caption_data(train_image_root, train_json_path)

output_path = "/mnt/data-home/julian/coco/val"
# save_coco_to_webdataset(val_id_to_path_caption, output_path, caption_idx=0)
output_path = "/mnt/data-home/julian/coco/train"
save_coco_to_webdataset(train_id_to_path_caption, output_path, caption_idx=0)
import ipdb; ipdb.set_trace()