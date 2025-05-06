import os
import pathlib
import tqdm
import json
import fire
import shutil


def main(slice_root, output_root, actuall_run=False):
    # output_root = "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand0422"
    # slice_root = "/mnt/data-home/mobility-multimodal/checkpoint/bbox/dataslices0422"
    slice_folder_list = pathlib.Path(slice_root).glob("*/*")
    slice_folder_list = [x for x in slice_folder_list if x.is_dir()]

    count = {
        "orginal": 0,
        "qa": 0,
        "deduplicated": 0,
        "final": 0,
    }

    for slice_folder in tqdm.tqdm(slice_folder_list):
        image_list = list((slice_folder / "images").glob("*"))
        anno_path = slice_folder / "annotations" / "labels.json"

        with open(anno_path, 'r') as f:
            coco_anno = json.load(f)

        image_annos = coco_anno["images"]
        anno_annos = coco_anno["annotations"]
        categories = coco_anno["categories"]

        orginal_number = len(image_list)
        image_qa_id_set = {x['image_id'] for x in anno_annos}
        image_final_id_set = image_qa_id_set 
        image_final_id_map = {x: i for i, x in enumerate(image_final_id_set)}
        count["orginal"] += orginal_number
        count["qa"] += len(image_qa_id_set)
        # count["deduplicated"] += len(image_keep_id_set)
        count["final"] += len(image_final_id_set)

        image_new_annos = [x for x in image_annos if x['id'] in image_final_id_set]
        anno_new_annos = [x for x in anno_annos if x['image_id'] in image_final_id_set]
        for x in image_new_annos:
            x['id'] = image_final_id_map[x['id']]
        for x in anno_new_annos:
            x['image_id'] = image_final_id_map[x['image_id']]


        print(f'{len(image_new_annos)=}')
        # count += len(image_new_annos)
        print(count)

        if actuall_run:
            new_folder = output_root / slice_folder.relative_to(slice_root)
            new_folder.mkdir(parents=True, exist_ok=True)
            (new_folder / "done").touch()
            (new_folder / "images").mkdir(parents=True, exist_ok=True)
            (new_folder / "annotations").mkdir(parents=True, exist_ok=True)
            os.chmod(new_folder, 0o777)
            os.chmod((new_folder/"images"), 0o777)
            os.chmod((new_folder/"annotations"), 0o777)

            for image in tqdm.tqdm(image_new_annos):
                src_image_path = slice_folder / "images" / image['file_name']
                dst_image_path = new_folder / "images" / image['file_name']
                shutil.copy(src_image_path, dst_image_path)

            coco_anno_dict = {
                "info": {'year': '', 'version': '', 'description': '', 'contributor': '', 'url': '', 'date_created': ''},
                "licenses": [],
                "categories": categories,
            }
            coco_anno_dict["images"] = image_new_annos
            coco_anno_dict["annotations"] = anno_new_annos

            new_anno_file = new_folder / 'annotations' / 'labels.json'
            with open(new_anno_file, 'w') as f:
                json.dump(coco_anno_dict, f)

    print(count)

if __name__ == "__main__":
    fire.Fire(main)