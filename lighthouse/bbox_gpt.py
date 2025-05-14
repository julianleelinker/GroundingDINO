import pathlib
import json
import torch
import tqdm
from PIL import Image
import os
import fire
import io
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
        bbox_int = torch.ceil(bbox)
        cropped_image = image_pil.crop((int(bbox_int[0]), int(bbox_int[1]), int(bbox_int[2]), int(bbox_int[3])))
        cropped_image_path = pathlib.Path(output_root) / f"{instance_name}.jpg"
        cropped_image.save(cropped_image_path)
        response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, cropped_image_path, prompt = cropped_image_prompt)
        if response is None:
            cropped_image_path.unlink()
            continue
        json_path = pathlib.Path(output_root) / f"{instance_name}.json"
        with open(json_path, "w") as f:
            json.dump(response, f, indent=4, ensure_ascii=False)

    response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, image_path, prompt = full_image_prompt)
    if response is not None:
        instance_name = image_path.stem
        image_pil.save(pathlib.Path(output_root) / f"{instance_name}.jpg")
        json_path = pathlib.Path(output_root) / f"{instance_name}.json"
        with open(json_path, "w") as f:
            json.dump(response, f, indent=4, ensure_ascii=False)


def save_to_webdataset_auto(pairs, output_dir, base_name="shard", max_per_shard=10):
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

            # Create sample
            sample = {
                "__key__": image_name,
                "jpg": img_bytes,
                "txt": caption_text,
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


def main(scale=2.5, merge_threshold=0.26, plot_mode=False):
    coco_root = pathlib.Path("/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0422/Taiwan_Power_20250106_image_list_keep_0.95_0.30_0.35/split0/")
    image_id_to_name_and_anno = parse_coco_anno(coco_root)

    output_root = f"{coco_root.parent.name}/{coco_root.name}_s{scale}_mt{merge_threshold}"
    pathlib.Path(output_root).mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o777)

    # count = 0
    # for image_name, anno_list in tqdm.tqdm(image_id_to_name_and_anno.values()):
    #     image_path = pathlib.Path(coco_root) / "images" / image_name
    #     bboxes, (W, H) = get_yolo_bboxes_from_coco_anno(image_path, anno_list)
    #     coco_bbox_gpt_generate_image_text(image_path, bboxes, (W, H), output_root, scale=scale, merge_threshold=merge_threshold, plot_mode=plot_mode)
    #     count += 1
    #     if count > 10:
    #         break

    output_web_root = f"{coco_root.parent.name}/web_{coco_root.name}_s{scale}_mt{merge_threshold}"
    image_text_name = yield_image_text_name(output_root)
    save_to_webdataset_auto(image_text_name, output_web_root, base_name="shard", max_per_shard=10)


if __name__ == "__main__":
    fire.Fire(main)