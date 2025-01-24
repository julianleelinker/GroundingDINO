import argparse
import os
import pathlib
import json
import base64
import io

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from openai import AzureOpenAI
from openai import BadRequestError, InternalServerError


AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')

CLASS_LIST = [
    'algae',
    'animal',
    'barricade',
    'boat',
    'branch',
    'bridge',
    'building',
    'bus',
    'bus stop',
    'cane',
    'car',
    'collapse',
    'detention basin',
    'dog',
    'door',
    'driveway',
    'driveway entracne',
    'driveway exit',
    'electric tower',
    'electric wire',
    'emergency exit',
    'entrance',
    'equipment',
    'exit',
    'fallen leaves',
    'faregate',
    'fence',
    'fire',
    'firearm',
    'fish',
    'flooding',
    'flowmeter',
    'garbage',
    'graffiti',
    'guardrail',
    'helmet',
    'human',
    'intersection',
    'jersey barrier',
    'knife',
    'landslide',
    'lane',
    'levee',
    'life vest',
    'lighting',
    'litter',
    'lock',
    'manhole cover',
    'motorcycle',
    'no smoking sign',
    'obstacle',
    'oil leakage',
    'oil stain',
    'palanquin',
    'park',
    'parking lot',
    'passage',
    'paved shoulder',
    'pavement crack',
    'pavement defect',
    'pavement patch',
    'people',
    'pipeline',
    'platform',
    'pole',
    'ponding',
    'port',
    'road marking',
    'rock',
    'rope',
    'safety belt',
    'safety harness',
    'seat crutch',
    'ship',
    'sidewalk',
    'sign',
    'silt',
    'smoke',
    'solar panel',
    'steel',
    'streetlight',
    'stroller',
    'substation',
    'tank',
    'thermometer',
    'tire',
    'tools',
    'traffic cone',
    'traffic light',
    'traffic sign',
    'transformer',
    'transformer box',
    'trash',
    'trees',
    'truck',
    'twig',
    'vehicle',
    'vehicle door',
    'waiting area',
    'warning sign',
    'waste',
    'water gauge',
    'waterway',
    'weapon',
    'wheelchair',
    'window',
    'wounds',
]
HARD_CLASS_LIST = [
    'algae',
    'detention basin', 
    'electric tower',
    'electric wire',
    'flowmeter',
    'graffiti',
    'guardrail',
    'safety harness',
    'seat crutch',
    'tank',
    'transformer',
    'transformer box',
    'palanquin,'
]
# CLASS_LIST = [
#     'human',
#     'people',
#     'pedestrian',
#     'fence',
# ]
PROMPT_WORDS = 12
N_PROMPTS = len(CLASS_LIST)//PROMPT_WORDS + 1
TEXT_PROMPT_LIST = [' . '.join(CLASS_LIST[i:i+PROMPT_WORDS]) for i in range(N_PROMPTS-1)]
TEXT_PROMPT_LIST.append(' . '.join(CLASS_LIST[N_PROMPTS*(PROMPT_WORDS-1):]))
TEXT_PROMPT_LIST = [
    'excavator',
	'excavator',
	'jersey',
	'streetlight',
	'cone',
	'truck',
	'vest',
	'helmet',
	'traffic sign',
	'sign',
	'traffic light',
]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ask_chatgpt_describe_image(azure_openai_api_key, image_base64, prompt="Please briefly describe the image.\n"):
    config = {
                "azure_endpoint": "https://azure-openai-vision-platform.openai.azure.com/",
                "api_key": azure_openai_api_key,
                "api_version":"2024-02-15-preview"
            }
    client = AzureOpenAI(**config)

    messages = [
        {
            "role": "user",
            "content": [
                {
                  "type": "text",
                  "text": prompt,
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                  },
                },
            ],
        }
    ]
    try:
        completion = client.chat.completions.create(
                        model= "4o",
                        messages=messages,
                        temperature=0.5,
                        top_p=1,
                        max_tokens=500,
                        # stop='\n',
                    )
    except BadRequestError as e:
        print(e)
        return None
    except InternalServerError as e:
        print(e)
        return None
    return(completion.choices[0].message.content)


def plot_boxes_to_image(image_pil, tgt, color=(0, 0, 0)):
    image_result = image_pil.copy()
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_result)
    mask = Image.new("L", image_result.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for id, (box, label) in enumerate(zip(boxes, labels)):
        label_text = f'{id}|{str(label)}'
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        # color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), label_text, font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), label_text, fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_result, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


def infer_an_image(image_path, model, text_prompt, box_threshold, text_threshold, token_spans):
    # load image
    image_pil, image = load_image(image_path)

    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    return image_pil, pred_dict


def infer_an_image_text_list(image_path, model, text_prompt_list, box_threshold, text_threshold, token_spans):
    # load image
    image_pil, image = load_image(image_path)

    # run model
    boxes_filt_list, pred_phrases_concat = [], []
    for text_prompt in text_prompt_list:
        # print(f'infering {image_path} with {text_prompt}')
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
        )
        boxes_filt_list.append(boxes_filt)
        pred_phrases_concat.extend(pred_phrases)
    boxes_filt = torch.vstack(boxes_filt_list)

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases_concat,
    }
    return image_pil, pred_dict


def xywh_to_xyxy(bboxes):
    results = bboxes.clone()
    results[:, :2] -= results[:, 2:] / 2
    results[:, 2:] += results[:, :2]
    return results


def xyxy_to_xywh(bboxes):
    results = bboxes.clone()
    results[:, 2:] -= results[:, :2]
    results[:, :2] += results[:, 2:] / 2
    return results


def fix_boundary(bboxes):
    results = bboxes.clone()
    results = xywh_to_xyxy(results)
    results = torch.clamp(results, 0., 1.)
    results = xyxy_to_xywh(results)
    return results


def compute_intersection_over_self(bboxes1, bboxes2=None):
    if bboxes2 is None:
        bboxes2 = bboxes1.clone()

    area1 = ((bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])).unsqueeze(-1)
    area2 = ((bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])).unsqueeze(0)
    area_min = torch.min(area1, area2)

    # add dummy dimension 1 to bbox1
    bboxes1 = bboxes1.unsqueeze(-1)
    # add dummy dimension 1 to bbox2 to last dimension
    bboxes2 = bboxes2.unsqueeze(-1)
    # reverse dimension of bbox2
    bboxes2 = bboxes2.permute(2, 1, 0)

    ma = torch.max(bboxes1, bboxes2)
    mi = torch.min(bboxes1, bboxes2)
    width = torch.max(torch.tensor([0]), mi[:, 2, :] - ma[:, 0, :])
    height = torch.max(torch.tensor([0]), mi[:, 3, :] - ma[:, 1, :])
    intersection_area = width * height
    ios = intersection_area / area_min
    return ios


def merge_two_bbox(bbox1, bbox2):
    result = torch.max(bbox1, bbox2)
    result[:2] = torch.min(bbox1, bbox2)[:2]
    return result


def merge_by_ios(bboxes, image_size, threshold):
    labels = [str(i) for i in range(len(bboxes))]
    bboxes = xywh_to_xyxy(bboxes)
    H, W = image_size
    bboxes = bboxes * torch.Tensor([W, H, W, H])
    while True:
        ios = compute_intersection_over_self(bboxes)
        ios = ios - 2.0*torch.eye(ios.size(0))
        max_pos = torch.unravel_index(torch.argmax(ios), ios.shape)
        if ios[max_pos]<threshold:
            break
        bboxes[max_pos[0]] = merge_two_bbox(bboxes[max_pos[0]], bboxes[max_pos[1]])
        bboxes = torch.cat((bboxes[:max_pos[1], :], bboxes[max_pos[1]+1:, :]), dim=0)
        labels[max_pos[0]] = labels[max_pos[0]] + ' ' + labels.pop(max_pos[1])
    bboxes = bboxes / torch.Tensor([W, H, W, H])
    bboxes = xyxy_to_xywh(bboxes)
    return bboxes, labels
        

def convert_pil_to_base64(image_pil):
    image_byte_array = io.BytesIO()
    image_pil.save(image_byte_array, format='PNG')
    image_byte_array = image_byte_array.getvalue()
    image_base64 = base64.b64encode(image_byte_array).decode('utf-8')
    return image_base64


def infer_images_text_list_save_result(image_path_list, model, text_prompt_list, box_threshold, text_threshold, token_spans, scale=1.5, threshold=0.5):
    for image_path in image_path_list:
        image_pil, pred_dict = infer_an_image_text_list(image_path, model, text_prompt_list, box_threshold, text_threshold, token_spans)
        image_base64 = convert_pil_to_base64(image_pil)
        response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, image_base64, prompt = 'Provide a one-sentence caption​ for the scene, time, and weather​ in the provided image.')
        print(f'{response=}')
        image_tmp = plot_boxes_to_image(image_pil, pred_dict, color=(255, 0, 0))[0]
        print(pred_dict['labels'])
        print(f'raw    {len(pred_dict["boxes"])=}')
        pred_dict["boxes"][:, 2:] *= scale
        pred_dict["boxes"] = fix_boundary(pred_dict["boxes"])
        pred_dict["boxes"], pred_dict['labels'] = merge_by_ios(pred_dict["boxes"], pred_dict['size'], threshold=threshold)
        print(f'merged {len(pred_dict["boxes"])=}')

        bboxes = xywh_to_xyxy(pred_dict["boxes"])
        H, W = pred_dict["size"]
        bboxes = bboxes * torch.Tensor([W, H, W, H])
        for i, bbox in enumerate(bboxes):
            bbox_int = torch.ceil(bbox)
            cropped_image = image_pil.crop((int(bbox_int[0]), int(bbox_int[1]), int(bbox_int[2]), int(bbox_int[3])))
            cropped_base64 = convert_pil_to_base64(cropped_image)
            response = ask_chatgpt_describe_image(AZURE_OPENAI_API_KEY, cropped_base64, prompt = 'Provide a one-sentence​ caption for​ the provided image.')
            pred_dict['labels'][i] += ' ' + response
            print(pred_dict['labels'][i])

        image_with_box = plot_boxes_to_image(image_tmp, pred_dict)[0]
        print(os.path.join(output_root_dir, f"{image_path.name}"))
        # import ipdb; ipdb.set_trace()
        image_with_box.save(os.path.join(output_root_dir, f"{image_path.name}"))


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
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")

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
    if 'SwinB' in config_file:
        model_name = 'SwinB'
    else:
        model_name = 'SwinT'
    # TEXT_PROMPT_LIST = [args.text_prompt]

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    print(f'{image_root=}')
    root_path = pathlib.Path(image_root)
    if root_path.is_dir():
        output_root_dir = pathlib.Path(output_dir).resolve() / model_name / (root_path.name + '_' + args.text_prompt)
        output_root_dir.mkdir(exist_ok=True, parents=True)
        image_path_list = list(root_path.rglob("*.jpg")) + list(root_path.rglob("*.png"))
        infer_images_text_list_save_result(image_path_list[:1], model, TEXT_PROMPT_LIST, box_threshold, text_threshold, token_spans, scale=2.0)
        # for image_path in image_path_list:
            # image_pil, pred_dict = infer_an_image_text_list(image_path, model, TEXT_PROMPT_LIST, box_threshold, text_threshold, token_spans)
            # print(pred_dict['labels'])
            # image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
            # print(os.path.join(output_root_dir, f"{args.prefix}_{image_path.name}"))
            # # import ipdb; ipdb.set_trace()
            # image_with_box.save(os.path.join(output_root_dir, f"{args.prefix}_{image_path.name}"))
    elif root_path.suffix == '.json':
        output_root_dir = pathlib.Path(output_dir).resolve() / model_name / (root_path.stem + '_all' + args.text_prompt)
        output_root_dir.mkdir(exist_ok=True, parents=True)
        with open(root_path, "r") as file:
            image_path_list = json.load(file)
        image_path_list = [pathlib.Path(image_path) for image_path in image_path_list]
        infer_images_text_list_save_result(image_path_list, model, TEXT_PROMPT_LIST, box_threshold, text_threshold, token_spans)
    else:
        image_pil, pred_dict = infer_an_image_text_list(root_path, model, TEXT_PROMPT_LIST, box_threshold, text_threshold, token_spans)
        print(pred_dict['labels'])
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(os.path.join(output_dir, f"{model_name}_{root_path.name}"))

