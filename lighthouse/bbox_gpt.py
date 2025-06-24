import pathlib
import json
import torch
import tqdm
from PIL import Image
import os
import fire
import io
import shutil
import webdataset as wds

from infer_settings import AZURE_OPENAI_API_KEY
from inference_on_a_image import load_image, load_model, get_grounding_output, plot_boxes_to_image, infer_an_image, infer_an_image_text_list
from box_utils import xywh_to_xyxy, fix_boundary, merge_by_iou
from chatgpt import encode_image, ask_chatgpt_describe_image, ask_chatgpt_describe_image_find_suitable_answer, convert_pil_to_base64, generate_vlm_pretraining_annotation


def parse_coco_anno(coco_root):
    coco_root = pathlib.Path(coco_root)
    anno_path = coco_root / "annotations" / "labels.json"
    image_path_list = list((coco_root / "images").glob("*"))

    with open(anno_path, "r") as f:
        anno_data = json.load(f)
    annotations = anno_data["annotations"]
    images = anno_data["images"]

    image_id_to_name_and_anno = {}
    for image in images:
        image_id_to_name_and_anno[image["id"]] = (image["file_name"], [])
    for anno in annotations:
        image_id_to_name_and_anno[anno["image_id"]][1].append(anno)
    return image_id_to_name_and_anno


def get_yolo_bboxes_from_coco_anno(image_path, anno_list):
    bbox_list = [anno["bbox"] for anno in anno_list]
    image_pil = Image.open(image_path)
    W, H = image_pil.width, image_pil.height
    bboxes = torch.tensor(bbox_list, dtype=torch.float32)
    bboxes = bboxes / torch.Tensor([W, H, W, H])
    # shift top left corner to center
    bboxes[:, :2] += 0.5*bboxes[:, 2:]
    return bboxes, (W, H)


def get_cocoo_bboxes_from_yolo_bboxes(bboxes, H, W):
    # bbox_list = [anno["bbox"] for anno in anno_list]
    # image_pil = Image.open(image_path)
    # W, H = image_pil.width, image_pil.height
    # bboxes = torch.tensor(bbox_list, dtype=torch.float32)
    # shift top left corner to center
    bboxes[:, :2] -= 0.5*bboxes[:, 2:]
    bboxes = bboxes * torch.Tensor([W, H, W, H])
    return bboxes


def coco_bbox_gpt_generate_image_text(image_path, bboxes, image_size, output_root, scale=4.0, merge_threshold=0.1, plot_mode=False, full_image_prompt = "Provide a one-sentence caption​ for the scene, time, and weather​ in the provided image.​", cropped_image_prompt = "Provide a one-sentence caption​ for the provided image."):
    image_pil = Image.open(image_path)
    W, H = image_size

    if plot_mode:
        pred_dict = {
            "size": (H, W),
            "boxes": bboxes,
            "labels": [x for x in range(len(bboxes))],
        }
        image_tmp = plot_boxes_to_image(image_pil, pred_dict, color=(255, 0, 0))[0] # Uses imported plot_boxes_to_image

    bboxes[:, 2:] *= scale
    bboxes = fix_boundary(bboxes)
    merged_bboxes, merged_labels = merge_by_iou(bboxes, image_size=(H, W), threshold=merge_threshold)

    if plot_mode:
        pred_dict["boxes"] = merged_bboxes
        pred_dict['labels'] = merged_labels
        image_tmp = plot_boxes_to_image(image_tmp, pred_dict, color=(0, 255, 0))[0] # Uses imported plot_boxes_to_image
        image_tmp.save(pathlib.Path(output_root) / f"{image_path.name}")
        return

    gpt_bboxes = xywh_to_xyxy(merged_bboxes)
    gpt_bboxes = gpt_bboxes * torch.Tensor([W, H, W, H])


    # import ipdb; ipdb.set_trace()
    # # save image_pil to output_root_dir
    for i, bbox in enumerate(gpt_bboxes):
        instance_name = f"{image_path.stem}-{i}"
        cropped_image_path = pathlib.Path(output_root) / f"{instance_name}.jpg"
        json_path = pathlib.Path(output_root) / f"{instance_name}.json"
        if cropped_image_path.exists() and json_path.exists():
            continue
        bbox_int = torch.ceil(bbox)
        cropped_image = image_pil.crop((int(bbox_int[0]), int(bbox_int[1]), int(bbox_int[2]), int(bbox_int[3])))
        cropped_image.save(cropped_image_path)
        response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, cropped_image_path, prompt = cropped_image_prompt)
        if response is None:
            cropped_image_path.unlink()
            continue
        with open(json_path, "w") as f:
            json.dump(response, f, indent=4, ensure_ascii=False)

    instance_name = image_path.stem
    image_path = pathlib.Path(output_root) / f"{instance_name}.jpg"
    json_path = pathlib.Path(output_root) / f"{instance_name}.json"
    if image_path.exists() and json_path.exists():
        return
    image_pil.save(image_path)
    response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, image_path, prompt = full_image_prompt)
    if response is not None:
        with open(json_path, "w") as f:
            json.dump(response, f, indent=4, ensure_ascii=False)
    else:
        image_path.unlink()


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
                "__key__": pathlib.Path(image_name).stem,
                "jpg": img_bytes,
                # "txt": caption_text,
                "txt": caption_bytes,
            }
            sink.write(sample)


def yield_image_text_name(folder_path):
    image_list = pathlib.Path(folder_path).glob("*.jpg")
    for image_path in image_list:
    # for i in range(10000000):
        image_pil = Image.open(image_path)
        json_path = image_path.with_suffix(".json")
        with open(json_path, "r") as f:
            text = json.load(f)
        yield image_pil, text, image_path.name


DATA_ROOTS_0 = {
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split1_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split29_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split2_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split31_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split3_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split4_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split5_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split6_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split7_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split8_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split9_0.30_0.35": None,
    "hand0428/Public_Works_20250106_image_list_keep_0.95/split1_0.30_0.35": None,
}

DATA_ROOTS_1 = {
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split12_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split13_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split1_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split2_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split3_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split4_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split5_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split6_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split7_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split8_0.30_0.35": None,
    "hand0428/Sports_Development_20241223_image_list_keep_0.95/split9_0.30_0.35": None,
    "hand0428/Sports_Development_20250109_image_list_keep_0.95/split1_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split11_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split12_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split13_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split14_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split15_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split1_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split2_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split3_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split4_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split5_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split6_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split7_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split8_0.30_0.35": None,
    "hand0428/Water_Resources_20250106_image_list_keep_0.95/split9_0.30_0.35": None,
}

DATA_ROOTS_2 = {
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split19_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split20_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split21_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split23_0.30_0.35": None,
}

DATA_ROOTS_3 = {
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split25_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split26_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split27_0.30_0.35": None,
    "hand0428/Public_Works_20241230_image_list_keep_0.95/split28_0.30_0.35": None,
}


def main(scale=2.5, merge_threshold=0.35, plot_mode=False, split=0):
    data_root_list = [
        DATA_ROOTS_0,
        DATA_ROOTS_1,
        DATA_ROOTS_2,
        DATA_ROOTS_3,
    ]
    split_root_list = data_root_list[split]

    for split_root, split_range in tqdm.tqdm(split_root_list.items()):
        split_root = pathlib.Path(f"/mnt/lighthouseACD/QAed-data/bbox/{split_root}")
        data_root = split_root.parent
        output_root = pathlib.Path(f"/mnt/lighthouseACD/image_text/{data_root.parent.name}")

        # for split_root in tqdm.tqdm(split_list):
        output_folder = pathlib.Path(f"{output_root}/{split_root.parent.name}/{split_root.name}_s{scale}_mt{merge_threshold}")
        if output_folder.exists():
            print(f"Output folder {output_folder} already exists, skipping...")
            continue

        image_id_to_name_and_anno = parse_coco_anno(split_root)

        tmp_root = pathlib.Path(f"/tmp/{split_root.parent.name}/{split_root.name}_s{scale}_mt{merge_threshold}")
        tmp_root.mkdir(parents=True, exist_ok=True)
        os.chmod(tmp_root, 0o777)

        # generate temp file for saving to webdataset
        for image_name, anno_list in tqdm.tqdm(image_id_to_name_and_anno.values()):
            image_path = pathlib.Path(split_root) / "images" / image_name
            bboxes, (W, H) = get_yolo_bboxes_from_coco_anno(image_path, anno_list)
            coco_bbox_gpt_generate_image_text(image_path, bboxes, (W, H), tmp_root, scale=scale, merge_threshold=merge_threshold, plot_mode=plot_mode)

        image_text_name = yield_image_text_name(tmp_root)
        save_to_webdataset_auto(image_text_name, output_folder, base_name="shard", max_per_shard=1000)
        shutil.rmtree(tmp_root)
# "deprecated"


if __name__ == "__main__":
    fire.Fire(main)


# data for metaclip
# a json store the fowllowing information:
# a list of dict, each dict contains:
# "image_path": "/mnt/Jan/Sports_Development/20250109/NO.22_H車道/20250109_NO.22_H車道_2024_10_3 上午 (UTC+08_00) 09_59_59_s0.jpg",
# "text":  "Provide a one-sentence caption​ for the provided image.", 