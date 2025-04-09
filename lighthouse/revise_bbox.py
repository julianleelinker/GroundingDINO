import json
import pathlib
import tqdm
import shutil
import fire


def main(dry_run=False):
    slice_root = "/mnt/data-home/mobility-multimodal/revised_bbox/dataslices"
    new_bbox_root = "/mnt/data-home/mobility-multimodal/revised_bbox/datasets"

    bbox_json = "/mnt/data-home/chungan/acd_ckpt1_image_list_keep_0.95.json"
    with open(bbox_json, 'r') as f:
        bbox_image_list = json.load(f)
    bbox_image_dict = {pathlib.Path(x).name: pathlib.Path(x) for x in bbox_image_list}

    slice_folder_list = pathlib.Path(slice_root).glob("*")
    slice_folder_list = [x for x in slice_folder_list if x.is_dir()]
    # slice_folder_list = slice_folder_list[:1]


    count = {
        "original": 0,
        "removed": 0,
        "new": 0,
    }

    for slice_folder in tqdm.tqdm(slice_folder_list):
        # image_list = list((slice_folder / "images").glob("*"))
        anno_path = slice_folder / "annotations" / "labels.json"

        with open(anno_path, 'r') as f:
            old_coco_anno = json.load(f)
        # import ipdb; ipdb.set_trace()

        image_annos = old_coco_anno["images"]
        anno_annos = old_coco_anno["annotations"]

        count["original"] += len(image_annos)

        image_keep_id_set = {x['id'] for x in image_annos if x['file_name'] in bbox_image_dict}
        image_final_id_set = image_keep_id_set

        image_new_annos = [x for x in image_annos if x['id'] in image_final_id_set]
        anno_new_annos = [x for x in anno_annos if x['image_id'] in image_final_id_set]

        count["new"] += len(image_new_annos)
        count["removed"] += len(image_annos) - len(image_new_annos)

        image_final_id_map = {x: i for i, x in enumerate(image_final_id_set)}
        for x in image_new_annos:
            x['id'] = image_final_id_map[x['id']]
        for x in anno_new_annos:
            x['image_id'] = image_final_id_map[x['image_id']]

        if not dry_run:
            new_coco_anno = {
                "info": old_coco_anno["info"],
                "licenses": old_coco_anno["licenses"],
                "categories": old_coco_anno["categories"],
                "images": image_new_annos,
                "annotations": anno_new_annos,
            }

            new_bbox_data_root = str(bbox_image_dict[image_new_annos[0]["file_name"]].parent.parent).replace("/mnt/lighthouseACD/ACD-gdino-COCO", new_bbox_root)
            new_bbox_image_root = pathlib.Path(new_bbox_data_root) / "images"
            new_bbox_image_root.mkdir(parents=True, exist_ok=True)

            for x in tqdm.tqdm(image_new_annos):
                src_file = bbox_image_dict[x["file_name"]]
                dst_file = new_bbox_image_root / x['file_name']
                shutil.copy(src_file, dst_file)

            new_anno_root = pathlib.Path(new_bbox_data_root) / "annotations" 
            new_anno_root.mkdir(parents=True, exist_ok=True)
            new_anno_file = new_anno_root / "labels.json"
            with open(new_anno_file, 'w') as f:
                json.dump(new_coco_anno, f)
        print(count)


if __name__ == "__main__":
    fire.Fire(main)