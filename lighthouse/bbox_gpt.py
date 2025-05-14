import pathlib
import json
import torch
import tqdm
from PIL import Image
import os
import fire

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


def coco_bbox_gpt_generate_image_text(image_path, bboxes, image_size, output_root, scale=4.0, merge_threshold=0.1, plot_mode=False, full_image_prompt = "Provide a one-sentence caption​ for the scene, time, and weather​ in the provided image.​"):
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


    json_path = pathlib.Path(output_root) / "response.json"
    response_json = []
    # import ipdb; ipdb.set_trace()
    # # save image_pil to output_root_dir
    for i, bbox in enumerate(gpt_bboxes):
        bbox_int = torch.ceil(bbox)
        cropped_image = image_pil.crop((int(bbox_int[0]), int(bbox_int[1]), int(bbox_int[2]), int(bbox_int[3])))
        cropped_image_path = pathlib.Path(output_root) / f"{image_path.stem}-{i}.jpg"
        cropped_image.save(cropped_image_path)
        response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, cropped_image_path, prompt = 'Provide a one-sentence​ caption for​ the provided image.')
        if response is None:
            cropped_image_path.unlink()
            continue
        response_json.append({"file": f"{image_path.stem}-{i}.jpg", "response": response})
        with open(json_path, "w") as f:
            json.dump(response, f, indent=4, ensure_ascii=False)

    response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, image_path, prompt = full_image_prompt)
    if response is not None:
        image_pil.save(pathlib.Path(output_root) / f"{image_path.name}")
        response_json.append({"file": f"{image_path.stem}.jpg", "response": response})
    with open(json_path, "w") as f:
        json.dump(response_json, f, indent=4, ensure_ascii=False)


def main(scale=2.5, merge_threshold=0.26, plot_mode=False):
    coco_root = pathlib.Path("/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0422/Taiwan_Power_20250106_image_list_keep_0.95_0.30_0.35/split0/")
    image_id_to_name_and_anno = parse_coco_anno(coco_root)

    output_root = f"{coco_root.parent.name}/{coco_root.name}_s{scale}_mt{merge_threshold}"
    pathlib.Path(output_root).mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o777)

    for image_name, anno_list in tqdm.tqdm(image_id_to_name_and_anno.values()):
        image_path = pathlib.Path(coco_root) / "images" / image_name
        bboxes, (W, H) = get_yolo_bboxes_from_coco_anno(image_path, anno_list)
        coco_bbox_gpt_generate_image_text(image_path, bboxes, (W, H), output_root, scale=scale, merge_threshold=merge_threshold, plot_mode=plot_mode)


if __name__ == "__main__":
    fire.Fire(main)