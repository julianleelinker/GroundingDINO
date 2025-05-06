# %%
import pathlib
import tqdm
import json
from collections import defaultdict
import shutil
import fire


def main(vlm_root, dedu_json, actuall_run=False):
    # dedu_json = "/mnt/data-home/chungan/curation/hand0505_image_list_keep_0.95.json"
    # vlm_root = "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand0505"
    # copyt hand0505 to hand0505-back using shutil
    print(f"copying {vlm_root} to {vlm_root}-back")
    shutil.copytree(vlm_root, vlm_root + "-back")
    with open(dedu_json, 'r') as f:
        dedu_list = json.load(f)
    dedu_set = {x for x in dedu_list}

    slice_folder_list = list(pathlib.Path(vlm_root).glob("*"))

    count = {
        "total": 0,
        "dedu": 0,
        "bad_anno": 0,
        "remain": 0,
    }

    for slice_folder in tqdm.tqdm(slice_folder_list):
        image_list = list((slice_folder / "images").glob("*"))
        anno_path = slice_folder / "annotations" / "vlm_annotation.json"
        with open(anno_path, 'r') as f:
            old_annos = json.load(f)
        name_to_anno = {x["image"]: x for x in old_annos}
        new_annos = []
        for image in tqdm.tqdm(image_list):
            count["total"] += 1

            if not str(image) in dedu_set:
                count["dedu"] += 1
                if actuall_run:
                    image.unlink()
                continue

            conversations = name_to_anno[image.name]["conversations"]
            bad_anno_condition = \
                (len(conversations) == 0) \
                or (len(conversations) > 0 and len(conversations[0]["answer"]["groundtruth"].split(" ")) < 50)
            if bad_anno_condition:
                count["bad_anno"] += 1
                if actuall_run:
                    image.unlink()
                continue    

            count["remain"] += 1
            new_annos.append(name_to_anno[image.name])

        if actuall_run:
            with open(anno_path, 'w') as f:
                json.dump(new_annos, f)
        print(count)

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    fire.Fire(main)