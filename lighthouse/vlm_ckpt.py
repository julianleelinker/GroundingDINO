import pathlib
import tqdm
import json
from collections import defaultdict
from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS
import shutil


vlm_json = "/mnt/data-home/mobility-multimodal/vlm-annotations/ckpt1_ckpt2_image_list.json"
acd_json = "/mnt/data-home/mobility-multimodal/gdino-coco/acd_ckpt1_image_list.json"

vlm_de_json = "/mnt/data-home/chungan/curation/checkpoint_1_and_2_image_list_keep_0.95.json"
acd_de_json = "/mnt/data-home/chungan/acd_ckpt1_image_list_keep_0.95.json"


# with open(acd_de_json, 'r') as f:
#     acd_de_list = json.load(f)

# acd_de_dict = defaultdict(list)
# for x in acd_de_list:
#     acd_de_dict[pathlib.Path(x).name].append(x)
# for key, value in acd_de_dict.items():
#     if len(value) > 1:
#         print(key, value)

with open(vlm_de_json, 'r') as f:
    vlm_de_list = json.load(f)

vlm_de_dict = defaultdict(list)
for x in vlm_de_list:
    vlm_de_dict[pathlib.Path(x).name].append(x)
for key, value in vlm_de_dict.items():
    if len(value) > 1:
        print(key, value)

# handle duplicated image
dupl_name_to_path = {
 "lk-241231-p3-s1s2s3s4": "/mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part3_3/images/1725874761.74193668.jpg", 
 "lk-241231-p4-s1s2s3s4": "/mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part4_3/images/1725874761.74193668.jpg",
}

output_root = "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand"


# load anno, iterated over all images and annos
    # find orginal image path, check if its in de_list, if not do nothing
    # if anno not exist, do nothing
    # copy image to new folder, 
    # if new_anno exist, append anno to new_anno
    # else create new_anno
    # save new anno to new folder
    # update number
slice_root = "/mnt/data-home/mobility-multimodal/checkpoint/vlm/ckpt2"
slice_folder_list = pathlib.Path(slice_root).glob("*")
slice_folder_list = [x for x in slice_folder_list if x.is_dir()]
slice_folder_list = [x for x in slice_folder_list if x.name in ["tr-250115"]]
import ipdb; ipdb.set_trace()

accumulate_count = 0
new_anno_list_dict = {}
dry_run = False

for slice_folder in tqdm.tqdm(slice_folder_list):
    image_list = list((slice_folder / "images").glob("*"))
    anno_path = slice_folder / "annotations" / "vlm_annotation.json"
    with open(anno_path, 'r') as f:
        anno_list = json.load(f)
    anno_dict = {x["image"]: x for x in anno_list}
    for image in tqdm.tqdm(image_list):
        if not image.name in vlm_de_dict:
            continue
        bad_condition = len(anno_dict[image.name]["conversations"]) == 0
        if bad_condition:
            continue    
        if len(vlm_de_dict[image.name]) == 1:
            org_image_path = vlm_de_dict[image.name][0]
        elif len(vlm_de_dict[image.name]) == 2:
            print(f"{image.name=} has dupliacted")
            org_image_path = dupl_name_to_path[slice_folder.name]
        accumulate_count += 1
        dst_image_path = pathlib.Path(org_image_path.replace("/mnt/data-home/mobility-multimodal/vlm-annotations", output_root))
        dst_image_path.parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            shutil.copy(org_image_path, dst_image_path)

        new_anno_file = dst_image_path.parent.parent / "annotations" / "vlm_annotation.json"

        if new_anno_file in new_anno_list_dict:
            new_anno_list_dict[new_anno_file].append(anno_dict[image.name])
        else: 
            new_anno_list_dict[new_anno_file] = [anno_dict[image.name]]

    print(f"{accumulate_count=}")

if not dry_run:
    for anno_path, anno_list in new_anno_list_dict.items():
        anno_path.parent.mkdir(parents=True, exist_ok=True)
        with open(anno_path, 'w') as f:
            json.dump(anno_list, f)
        print(f"{anno_path=}, {len(anno_list)=}")