import pathlib
import json
import os
import shutil
import tqdm

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt, show_id=True, color=None):
    """
    Plots bounding boxes on an image.
    """
    image_result = image_pil.copy()
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_result)
    mask = Image.new("L", image_result.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    random_color = color is None
    for id, (box, label) in enumerate(zip(boxes, labels)):
        if show_id:
            label_text = f'{id}|{str(label)}'
        else:
            label_text = str(label)
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        if random_color:
            color = tuple(np.random.randint(0, 128, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        # print(x0, y0, x1, y1) # Removed print

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
    """
    Loads and transforms an image for model input.
    """
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


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    """
    Runs the GroundingDINO model and filters the output based on thresholds.
    """
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu" # Check cuda availability
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    scores = [] # Initialize scores list
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
        # scores = [] # Moved initialization up
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            score = logit.max().item() # Calculate score once
            scores.append(score)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(score)[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        # Assuming text_prompt is passed correctly if using token_spans outside this function context
        # This part might need adjustment depending on how token_spans are generated/used
        tokenized = model.tokenizer(caption) # Tokenize the caption here
        positive_maps = create_positive_map_from_span(
            tokenized, # Use the tokenized caption
            token_span=token_spans
        ).to(device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        # scores = [] # Moved initialization up
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            filtered_boxes = boxes[filt_mask]
            all_boxes.append(filtered_boxes)
            # filt logits
            logit_phr_num = logit_phr[filt_mask]
            all_logits.append(logit_phr_num)

            current_scores = logit_phr_num.cpu().tolist() # Get scores for the current phrase
            scores.extend(current_scores) # Add scores to the main list

            if with_logits:
                all_phrases.extend([phrase + f"({str(logit_item)[:4]})" for logit_item in current_scores])
            else:
                # Ensure the length matches the number of boxes found for this phrase
                all_phrases.extend([phrase for _ in range(len(filtered_boxes))])

        if all_boxes: # Check if any boxes were found
             boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        else:
             boxes_filt = torch.empty((0, 4), dtype=torch.float32).cpu() # Return empty tensor if no boxes
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases, scores


def infer_an_image_text_list(image_path, model, text_prompt_list, box_threshold, text_threshold, higher_class_list, high_threshold, token_spans, cpu_only=False):
    """
    Infers objects in an image based on a list of text prompts.
    Applies a higher confidence threshold for specified classes.
    """
    # load image
    image_pil, image = load_image(image_path)

    # run model
    boxes_filt_list, pred_phrases_concat, scores_concat = [], [], []
    for text_prompt in text_prompt_list:
        # print(f'infering {image_path} with {text_prompt}') # Removed print
        # Note: get_grounding_output expects a single caption, not a list.
        # Assuming the intention is to run inference for each prompt individually.
        boxes_filt, pred_phrases, scores = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only, token_spans=token_spans # Pass token_spans if needed per prompt
        )

        if boxes_filt.nelement() == 0: # Check if boxes_filt is empty
            continue

        # Apply higher threshold logic
        indices_to_keep = []
        for i in range(len(boxes_filt)):
            # Extract base phrase name (remove score part if present)
            base_phrase = pred_phrases[i].split('(')[0].strip()
            # Check if the base phrase (which should correspond to the text_prompt) is in higher_class_list
            if text_prompt in higher_class_list:
                if scores[i] > high_threshold:
                    indices_to_keep.append(i)
            else:
                # Keep all boxes if the prompt is not in the higher_class_list
                indices_to_keep.append(i)

        if indices_to_keep:
            boxes_filt_list.append(boxes_filt[indices_to_keep])
            pred_phrases_concat.extend([pred_phrases[i] for i in indices_to_keep])
            scores_concat.extend([scores[i] for i in indices_to_keep])


    if not boxes_filt_list: # Handle case where no boxes are found after filtering
        final_boxes = torch.empty((0, 4), dtype=torch.float32)
        final_labels = []
    else:
        final_boxes = torch.vstack(boxes_filt_list)
        final_labels = pred_phrases_concat

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": final_boxes,
        "size": [size[1], size[0]],  # H,W
        "labels": final_labels, # Use the filtered labels
    }
    return image_pil, pred_dict


def infer_images_text_list_save_gdino_coco_result(image_path_list, model, text_prompt_list, box_threshold, text_threshold, higher_class_list, high_threshold, token_spans, output_root_dir, cpu_only=False):
    """
    Processes a list of images, performs inference using GroundingDINO with multiple text prompts,
    applies class-specific thresholds, and saves the results in COCO format.
    Includes resume functionality.
    """
    coco_anno = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i+1, "name": name} for i, name in enumerate(text_prompt_list)]
    }
    start_image_id = 1
    start_anno_id = 1 # Initialize annotation id counter
    cat_to_id = {text: i+1 for i, text in enumerate(text_prompt_list)}

    output_root_path = pathlib.Path(output_root_dir)
    image_root_dir = output_root_path / 'images'
    anno_root_dir = output_root_path / 'annotations'
    coco_label_path = anno_root_dir / 'labels.json'

    image_root_dir.mkdir(mode=0o777, exist_ok=True, parents=True)
    anno_root_dir.mkdir(mode=0o777, exist_ok=True, parents=True)
    # Setting permissions might require sudo or specific user group memberships
    try:
        os.chmod(image_root_dir, 0o777)
        os.chmod(anno_root_dir, 0o777)
    except PermissionError:
        print(f"Warning: Could not set permissions for {image_root_dir} or {anno_root_dir}. Check user permissions.")


    # Resume function
    processed_image_filenames = set()
    if coco_label_path.exists():
        print('Resuming...')
        try:
            with open(coco_label_path, 'r') as f:
                coco_anno = json.load(f)
            processed_image_filenames = {img['file_name'] for img in coco_anno['images']}
            if coco_anno['images']:
                start_image_id = max(img['id'] for img in coco_anno['images']) + 1
            if coco_anno['annotations']:
                 # Ensure unique annotation IDs across resumes
                start_anno_id = max(ann.get('id', 0) for ann in coco_anno['annotations']) + 1
            else:
                 coco_anno['annotations'] = [] # Ensure annotations list exists

            print(f"Resuming from image ID: {start_image_id}, annotation ID: {start_anno_id}")
            # Filter out already processed images
            original_count = len(image_path_list)
            image_path_list = [p for p in image_path_list if p.name not in processed_image_filenames]
            print(f"Skipped {original_count - len(image_path_list)} already processed images.")

            # Copy missing images (optional, could be slow for large datasets)
            # print("Checking for missing image files in output directory...")
            # for image_info in tqdm.tqdm(coco_anno['images']):
            #     src_path = next((p for p in image_path_list_original if p.name == image_info['file_name']), None) # Find original path
            #     dst_path = image_root_dir / image_info['file_name']
            #     if src_path and not dst_path.exists():
            #          try:
            #              shutil.copy(src_path, dst_path)
            #          except Exception as e:
            #              print(f"Warning: Could not copy {src_path} to {dst_path}: {e}")

        except json.JSONDecodeError:
            print(f"Warning: Could not decode existing annotation file {coco_label_path}. Starting fresh.")
            coco_anno = {"images": [], "annotations": [], "categories": [{"id": i+1, "name": name} for i, name in enumerate(text_prompt_list)]}
            start_image_id = 1
            start_anno_id = 1
        except Exception as e:
            print(f"Error during resume: {e}. Starting fresh.")
            coco_anno = {"images": [], "annotations": [], "categories": [{"id": i+1, "name": name} for i, name in enumerate(text_prompt_list)]}
            start_image_id = 1
            start_anno_id = 1


    current_anno_id = start_anno_id
    for path_id, image_path in enumerate(tqdm.tqdm(image_path_list, desc="Processing images")):
        image_id = path_id + start_image_id
        # print(f'{image_id=}, total={len(image_path_list)}') # Reduced verbosity

        try:
            image_pil, pred_dict = infer_an_image_text_list(
                image_path, model, text_prompt_list, box_threshold, text_threshold,
                higher_class_list, high_threshold, token_spans, cpu_only=cpu_only
            )
        except FileNotFoundError:
            print(f"Warning: Image file not found: {image_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Error processing image {image_path}: {e}. Skipping.")
            continue


        H, W = image_pil.size[1], image_pil.size[0]
        image_anno = {
            "id": image_id,
            "file_name": image_path.name,
            "width": W,
            "height": H,
            "date_captured": "N/A", # Or use actual date if available
            "license": 1, # Add default license/flicker_url if needed by COCO format
            "flickr_url": "",
            "coco_url": ""
        }
        coco_anno['images'].append(image_anno)

        boxes_xywh = pred_dict['boxes'].clone() # Operate on a clone
        # Convert XYWH (center) to XYWH (top-left) for COCO
        boxes_xywh[:, :2] -= boxes_xywh[:, 2:] * 0.5
        boxes_xywh = boxes_xywh * torch.Tensor([W, H, W, H])

        num_boxes_added = 0
        for j in range(len(pred_dict['boxes'])):
            # Extract base category name (remove score like '(0.345)')
            cat_name = pred_dict['labels'][j].split('(')[0].strip()
            if cat_name in cat_to_id:
                bbox_coco = [round(v.item()) for v in boxes_xywh[j]] # Use .item() and round
                 # Basic check for valid bbox dimensions
                if bbox_coco[2] > 0 and bbox_coco[3] > 0:
                    box_anno = {
                        "id": current_anno_id, # Assign unique annotation ID
                        "image_id": image_id,
                        "category_id": cat_to_id[cat_name],
                        "bbox": bbox_coco, # COCO format: [x_min, y_min, width, height]
                        "area": bbox_coco[2] * bbox_coco[3], # Calculate area
                        "iscrowd": 0, # Default iscrowd
                        "segmentation": [] # Add empty segmentation if needed
                    }
                    coco_anno['annotations'].append(box_anno)
                    current_anno_id += 1 # Increment annotation ID
                    num_boxes_added += 1
            # else: # Optional: Log unrecognized categories
            #     print(f"Warning: Category '{cat_name}' from prediction not in provided text_prompt_list for image {image_path.name}")


        # Save image copy
        try:
            dest_image_path = image_root_dir / f"{image_path.name}"
            if not dest_image_path.exists(): # Avoid unnecessary copies if resuming
                 shutil.copy(image_path, dest_image_path)
            # image_pil.save(dest_image_path) # Alternative: save PIL object if transformations were applied
        except Exception as e:
            print(f"Warning: Could not copy/save image {image_path.name} to output directory: {e}")


        # Save annotations periodically (e.g., every 100 images) and at the end
        if (path_id + 1) % 100 == 0 or (path_id + 1) == len(image_path_list):
            try:
                with open(coco_label_path, 'w') as f:
                    json.dump(coco_anno, f, indent=4, ensure_ascii=False)
            except Exception as e:
                 print(f"Error saving annotations to {coco_label_path}: {e}")


    # Final save after loop finishes
    try:
        with open(coco_label_path, 'w') as f:
            json.dump(coco_anno, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error performing final save of annotations to {coco_label_path}: {e}")

    # Create 'done' file to indicate completion
    try:
        (output_root_path / 'done').touch()
    except Exception as e:
        print(f"Warning: Could not create 'done' file in {output_root_path}: {e}")

    print(f"Processing complete. Annotations saved to {coco_label_path}")
