import pathlib
import json
from PIL import Image
import webdataset as wds
import os
import io
import fire
import tqdm


def save_to_webdataset_auto(pairs, output_dir, base_name="shard", max_per_shard=1000):
    """
    Save image-text pairs to WebDataset shards with automatic shard rotation.

    Args:
        pairs: Iterable of (image_pil, text) pairs.
        output_dir: Directory to save .tar shards.
        base_name: Base name for shards (e.g., 'shard' -> shard-000000.tar).
        max_per_shard: Max samples per shard (auto-rotate beyond this).
    """
    # os.makedirs(output_dir, exist_ok=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o777)
    pattern = os.path.join(output_dir, f"{base_name}-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=max_per_shard) as sink:
        for i, (image_pil, caption_text, image_name) in enumerate(pairs):
            # Convert image to JPEG bytes
            img_buffer = io.BytesIO()
            image_pil.convert("RGB").save(img_buffer, format="jpeg")
            img_bytes = img_buffer.getvalue()
            caption_bytes = caption_text.encode("utf-8")

            # Create sample
            sample = {
                # "__key__": pathlib.Path(image_name).stem,
                "__key__": f"{i:09d}",
                "jpg": img_bytes,
                "txt": caption_bytes,
            }
            sink.write(sample)


def yield_image_text_name(curated_path):
    with open(curated_path, "r") as f:
        curated_data = json.load(f)
    for data in curated_data:
        image_path = pathlib.Path(data["image_path"])
        image_pil = Image.open(image_path)
        text = data["text"]
        yield image_pil, text, image_path.name


def main(output_path, train_json, val_json):
    # output_path = "/mnt/lighthouseACD/image_text-back/wds"
    # split_path = {
    #     "train": "/home/julian/work/MetaCLIP/image_text-back/image_text-back_curated_t30_20250620.json",
    #     "val": "/home/julian/work/MetaCLIP/image_text-back/image_text-back_curated_t1_20250620.json",
    # }
    split_path = {
        "train": train_json,
        "val": val_json,
    }

    for split, json_path in tqdm.tqdm(split_path.items()):
        image_text_name = yield_image_text_name(json_path)
        save_to_webdataset_auto(image_text_name, output_path, base_name=split, max_per_shard=1000)


if __name__ == "__main__":
    fire.Fire(main)