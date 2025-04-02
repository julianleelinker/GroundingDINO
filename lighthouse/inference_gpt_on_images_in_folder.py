import argparse
import os
import pathlib
import json
import base64
import io
import json
import tqdm
import shutil

import numpy as np
import torch

# Import shared functions
from inference_on_a_image import load_image, load_model, get_grounding_output, plot_boxes_to_image, infer_an_image, infer_an_image_text_list
from chatgpt import encode_image, ask_chatgpt_describe_image, ask_chatgpt_describe_image_find_suitable_answer, convert_pil_to_base64, generate_vlm_pretraining_annotation
from infer_settings import AZURE_OPENAI_API_KEY, DINO_INFER_CLASSES, EASY_CLASSES_LIST, HARD_CLASSES_LIST
from box_utils import xywh_to_xyxy, fix_boundary, merge_by_ios


FULL_IMAGE_PROMPT = 'Provide a one-sentence caption​ for the scene, time, and weather​ in the provided image.'


def infer_images_text_list_save_gdino_coco_result(image_path_list, model, text_prompt_list, box_threshold, text_threshold, higher_class_list, high_threshold, token_spans, output_root_dir):
    coco_anno = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i+1, "name": name} for i, name in enumerate(text_prompt_list)]
    }
    start_image_id = 1
    cat_to_id = {text: i+1 for i, text in enumerate(text_prompt_list)}
    image_root_dir = pathlib.Path(output_root_dir) / 'images'
    image_root_dir.mkdir(mode=0o777, exist_ok=True, parents=True)
    os.chmod(image_root_dir, 0o777)

    coco_label_path = pathlib.Path(output_root_dir) / 'annotations' / 'labels.json'
    # resume function
    if coco_label_path.exists():
        with open(coco_label_path, 'r') as f:
            coco_anno = json.load(f)
        image_path_set = {image_anno['file_name'] for image_anno in coco_anno['images']}
        new_image_path_list = []
        print('resuming...')
        for image_path in tqdm.tqdm(image_path_list):
            if image_path.name in image_path_set:
                dst = image_root_dir / f"{image_path.name}"
                if not dst.exists():
                    shutil.copy(image_path, dst)
            else:
                new_image_path_list.append(image_path)
        image_path_list = new_image_path_list
        start_image_id = len(image_path_set) + 1

    for path_id, image_path in enumerate(image_path_list):
        image_id = path_id + start_image_id
        print(f'{image_id=}, {len(image_path_list)=}')
        image_pil, pred_dict = infer_an_image_text_list(image_path, model, text_prompt_list, box_threshold, text_threshold, higher_class_list, high_threshold, token_spans)
        H, W = image_pil.size[1], image_pil.size[0]
        image_anno = {
            "id": image_id,
            "file_name": image_path.name,
            "width": W,
            "height": H,
            "date_captured": "2022-09-01 00:00:00",
        }
        coco_anno['images'].append(image_anno)
        boxes = pred_dict['boxes'] * torch.Tensor([W, H, W, H])
        boxes[:, :2] -= boxes[:, 2:]*0.5
        # print(pred_dict['labels'])
        # print(cat)
        for j in range(len(pred_dict['boxes'])):
            cat = pred_dict['labels'][j].split('(')[0]
            if cat not in cat_to_id:
                continue
            bbox = [int(v) for v in boxes[j]]
            box_anno = {
                "image_id": image_id,
                "category_id": cat_to_id[cat],
                "bbox": bbox,
            }
            coco_anno['annotations'].append(box_anno)
        # print(coco_anno)
        # import ipdb; ipdb.set_trace()
        # image_with_box = plot_boxes_to_image(image_tmp, pred_dict, show_id=False)[0] # Uses imported plot_boxes_to_image
        # output_image_path = pathlib.Path(output_root_dir) / f"{image_path.name}"
        # output_text_path = pathlib.Path(output_root_dir) / f"{image_path.stem}.json"
        # save coco_anno to output_text_path

        image_pil.save(image_root_dir / f"{image_path.name}")

        coco_root_dir = pathlib.Path(output_root_dir) / 'annotations'
        coco_root_dir.mkdir(mode=0o777, exist_ok=True, parents=True)
        os.chmod(coco_root_dir, 0o777)
        with open(coco_root_dir / 'labels.json', 'w') as f:
            json.dump(coco_anno, f, indent=4, ensure_ascii=False)

    pathlib.Path(output_root_dir/'done').touch()

    # image_with_box.save(output_image_path)
    # # write label_txt to output_image_path
    # with open(output_text_path, 'w') as f:
    #     f.write('\n'.join(label_text))
    # import ipdb; ipdb.set_trace()

# Kept local function: infer_images_text_list_save_gpt_result
def infer_images_text_list_save_gpt_result(image_path_list, model, text_prompt_list, box_threshold, text_threshold, higher_class_list, high_threshold, token_spans, scale=1.5, merge_threshold=0.5):
    for image_path in image_path_list:
        image_pil, pred_dict = infer_an_image_text_list(image_path, model, text_prompt_list, box_threshold, text_threshold, higher_class_list, high_threshold, token_spans)

        image_base64 = convert_pil_to_base64(image_pil)
        annotation_list = []
        response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, image_base64, prompt = FULL_IMAGE_PROMPT)
        if response is None:
            continue
        annotation_list.append(generate_vlm_pretraining_annotation(1, image_path.name, FULL_IMAGE_PROMPT, response))
        print(f'{response=}')
        image_tmp = plot_boxes_to_image(image_pil, pred_dict, color=(255, 0, 0))[0] # Uses imported plot_boxes_to_image
        print(pred_dict['labels'])
        print(f'raw    {len(pred_dict["boxes"])=}')
        pred_dict["boxes"][:, 2:] *= scale
        pred_dict["boxes"] = fix_boundary(pred_dict["boxes"])
        pred_dict["boxes"], pred_dict['labels'] = merge_by_ios(pred_dict["boxes"], pred_dict['size'], threshold=merge_threshold)
        print(f'merged {len(pred_dict["boxes"])=}')

        bboxes = xywh_to_xyxy(pred_dict["boxes"])
        H, W = pred_dict["size"]
        bboxes = bboxes * torch.Tensor([W, H, W, H])
        # deep copy pred_dict['labels']
        label_text = pred_dict['labels'].copy()

        for i, bbox in enumerate(bboxes):
            bbox_int = torch.ceil(bbox)
            cropped_image = image_pil.crop((int(bbox_int[0]), int(bbox_int[1]), int(bbox_int[2]), int(bbox_int[3])))
            cropped_base64 = convert_pil_to_base64(cropped_image)
            response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, cropped_base64, prompt = 'Provide a one-sentence​ caption for​ the provided image.')
            if response is None:
                continue
            label_text[i] += ' ' + response
            print(label_text[i])
            bbox_image_name = pathlib.Path(output_root_dir) / f"{image_path.stem}-{i}.jpg"
            cropped_image.save(bbox_image_name)
            annotation_list.append(generate_vlm_pretraining_annotation(1, bbox_image_name.name, 'Provide a one-sentence​ caption for​ the provided image.', response))

        image_pil.save(pathlib.Path(output_root_dir) / f"{image_path.name}")
        output_image_path = pathlib.Path(output_root_dir) / f"{image_path.stem}-result.jpg"
        image_with_box = plot_boxes_to_image(image_tmp, pred_dict, show_id=False)[0] # Uses imported plot_boxes_to_image
        output_text_path = pathlib.Path(output_root_dir) / f"{image_path.stem}.txt"
        image_with_box.save(output_image_path)
        # write label_txt to output_image_path
        with open(output_text_path, 'w') as f:
            f.write('\n'.join(label_text))

        output_anno_path = pathlib.Path(output_root_dir) / f"{image_path.stem}.json"
        with open(output_anno_path, "w") as json_file:
            json.dump(annotation_list, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--high_threshold", type=float, default=0.28, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument("--ios_threshold", type=float, required=True, help="box threshold")
    parser.add_argument("--enlarge_scale", type=float, required=True, help="box threshold")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    # parser.add_argument("--prefix", type=str, default='pred', help="prefix of saved predicted filename")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_root = args.image_path
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    high_threshold = args.high_threshold
    if 'SwinB' in config_file:
        model_name = 'SwinB'
    else:
        model_name = 'SwinT'
    # TEXT_PROMPT_LIST = [args.text_prompt]

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only) # Uses imported load_model

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    print(f'{image_root=}')
    root_path = pathlib.Path(image_root)


    if root_path.is_dir():
        # image_path_list = list(root_path.rglob("*.jpg")) + list(root_path.rglob("*.png"))
        # output_root_dir = pathlib.Path(output_dir).resolve() / model_name / (root_path.name + '_' + args.text_prompt + f'_en{args.enlarge_scale:3.2f}_io{args.ios_threshold:3.2f}')
        image_path_list = list(root_path.rglob("*"))
        import ipdb; ipdb.set_trace()
        image_path_list = [x for x in image_path_list if x.suffix != '.txt']
        output_root_dir = pathlib.Path(output_dir).resolve() / (root_path.name + f'_{args.text_threshold:3.2f}_{args.high_threshold:3.2f}')
    elif root_path.suffix == '.json':
        with open(root_path, "r") as file:
            image_path_list = json.load(file)
        image_path_list = [pathlib.Path(image_path) for image_path in image_path_list]
        output_root_dir = pathlib.Path(output_dir).resolve() / (root_path.name + f'_{args.text_threshold:3.2f}_{args.high_threshold:3.2f}')
    else:
        print(f'unsupported {root_path=}')
        exit(-1)
    output_root_dir.mkdir(mode=0o777, exist_ok=True, parents=True)
    os.chmod(output_root_dir, 0o777)
    # print(TEXT_PROMPT_LIST)
    # infer_images_text_list_save_gpt_result(image_path_list, model, TEXT_PROMPT_LIST, box_threshold, text_threshold, HIGHER_CLASS_LIST, high_threshold, token_spans, scale=args.enlarge_scale, merge_threshold=args.ios_threshold)
    image_path_list = image_path_list[:2]
    infer_images_text_list_save_gdino_coco_result(image_path_list, model, DINO_INFER_CLASSES, box_threshold, text_threshold, HARD_CLASSES_LIST, high_threshold, token_spans, output_root_dir)
    # dino_coco_loaded = pd.read_csv(f"{DINO_COCO_ROOT}/dino_coco_stats.csv", dtype=STATS_COLUMN_DTYPES)

    src_map_file = root_path / 'name_to_path.txt'
    dst_map_file = output_root_dir / 'name_to_path.txt'
    shutil.copy(src_map_file, dst_map_file)
