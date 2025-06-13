import pathlib
import tqdm
from bbox_gpt import parse_coco_anno, get_yolo_bboxes_from_coco_anno, fix_boundary, merge_by_iou, get_cocoo_bboxes_from_yolo_bboxes
import os
from PIL import Image, ImageDraw
from infer_settings import DINO_INFER_CLASSES
from common import DINO_COCO_SPLITS_0508
import json
import fire
from inference_on_a_image import plot_boxes_to_image


def main(scale=1.0, merge_threshold=0.26):
    data_root = "/mnt/lighthouseACD/QAed-data/bbox/hand0526/"
    output_root = pathlib.Path("test_threshold")
    split_root = pathlib.Path("threshold_test")

    # import ipdb; ipdb.set_trace()
    # for split_root in tqdm.tqdm(split_root_list):
    for merge_threshold in [0.50, 0.26, 0.38]:

        output_folder = pathlib.Path(f"{output_root}_iou{merge_threshold:.2f}")

        if (output_folder / "done").exists():
            print(f"Output folder {output_folder} already exists, skipping...")
            continue
        output_label_folder = output_folder / "annotations"
        output_images_folder = output_folder / "images"
        pathlib.Path(output_label_folder).mkdir(parents=True, exist_ok=True)
        os.chmod(output_label_folder, 0o777)
        pathlib.Path(output_images_folder).mkdir(parents=True, exist_ok=True)
        os.chmod(output_images_folder, 0o777)

        image_id_to_name_and_anno = parse_coco_anno(split_root)

        # image_list = list(split_root.glob("images/*"))
        # if len(image_id_to_name_and_anno) != len(image_list):
        #     print(f"Number of images in {split_root} does not match the number of annotations: {len(image_list)=} vs {len(image_id_to_name_and_anno)=}")
        #     count += 1

        coco_anno = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}]
        }
        count = 0
        for image_id, (image_name, anno_list) in tqdm.tqdm(image_id_to_name_and_anno.items()):
            image_path = pathlib.Path(split_root) / "images" / image_name

            if len(anno_list) == 0:
                merged_bboxes = []
                count += 1
            else:
                bboxes, (W, H) = get_yolo_bboxes_from_coco_anno(image_path, anno_list)
                new_bboxes = bboxes.clone()
                new_bboxes[:, 2:] *= scale
                new_bboxes = fix_boundary(new_bboxes)
                merged_bboxes, _ = merge_by_iou(new_bboxes, image_size=(H, W), threshold=merge_threshold)
                # merged_bboxes = get_cocoo_bboxes_from_yolo_bboxes(merged_bboxes, H, W)

            merged_labels = [1] * len(merged_bboxes)  
            pred_dict = {
                "boxes": merged_bboxes,
                "labels":  merged_labels,
                "size": (H, W),
            }
            image_pil = Image.open(image_path)
            image_pil = plot_boxes_to_image(image_pil, pred_dict, color=(0, 255, 0))[0] # Uses imported plot_boxes_to_image
            image_pil.save(pathlib.Path(output_folder) / f"{image_path.name}")
            continue

            image_anno = {
                "id": image_id,
                "file_name": image_path.name,
                "width": W,
                "height": H,
                "date_captured": "2022-09-01 00:00:00",
            }
            coco_anno['images'].append(image_anno)
            image_pil = Image.open(image_path).convert("RGB")
            image_pil.save(output_images_folder / f"{image_path.name}")
            for i in range(len(merged_bboxes)):
                bbox = [int(v) for v in merged_bboxes[i]]
                box_anno = {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                }
                coco_anno['annotations'].append(box_anno)
        if count != 0:
            print(f"Number of images without annotations in {split_root}: {count}")

        coco_label_path = pathlib.Path(output_label_folder) / 'labels.json'
        with open(coco_label_path, 'w') as f:
            json.dump(coco_anno, f, indent=4)

        # assert len(coco_anno['images']) == len(image_id_to_name_and_anno), f"n images in new {len(coco_anno['images'])} does not match n images in old image_id_to_name_and_anno {len(image_id_to_name_and_anno)}"
        # (output_folder / "done").touch()


if __name__ == "__main__":
    fire.Fire(main)