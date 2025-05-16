import pathlib
import tqdm
import json
from collections import defaultdict
import shutil


vlm_json = "/mnt/data-home/mobility-multimodal/vlm-annotations/ckpt1_ckpt2_image_list.json"
vlm_de_json = "/mnt/data-home/chungan/curation/checkpoint_1_and_2_image_list_keep_0.95.json"

acd_json = "/mnt/data-home/mobility-multimodal/gdino-coco/acd_ckpt1_image_list.json"
acd_de_json = "/mnt/data-home/chungan/acd_ckpt1_image_list_keep_0.95.json"

with open(acd_json, 'r') as f:
    acd_list = json.load(f)


with open(acd_de_json, 'r') as f:
    acd_de_list = json.load(f)

acd_de_dict = defaultdict(list)
for x in acd_de_list:
    acd_de_dict[pathlib.Path(x).name].append(x)
for key, value in acd_de_dict.items():
    if len(value) > 1:
        print(key, value)

# with open(vlm_de_json, 'r') as f:
#     vlm_de_list = json.load(f)

# vlm_de_dict = defaultdict(list)
# for x in vlm_de_list:
#     vlm_de_dict[pathlib.Path(x).name].append(x)
# for key, value in vlm_de_dict.items():
#     if len(value) > 1:
        # print(key, value)



# load anno, iterated over all images and annos
    # find orginal image path, check if its in de_list, if not do nothing
    # if anno not exist, do nothing
    # copy image to new folder, 
    # if new_anno exist, append anno to new_anno
    # else create new_anno
    # save new anno to new folder
    # update number


output_root = "/mnt/lighthouseACD/QAed-data//bbox/hand"
slice_root = "/mnt/lighthouseACD/QAed-data//bbox/dataslices"
slice_folder_list = pathlib.Path(slice_root).glob("*")
slice_folder_list = [x for x in slice_folder_list if x.is_dir()]
# slice_folder_list = slice_folder_list[:1]
print(str(slice_folder_list[0]))
exit(0)

accumulate_count = 0
new_anno_list_dict = {}
dry_run = False


categories = [{'id': 0, 'name': 'algae'}, {'id': 1, 'name': 'animal'}, {'id': 2, 'name': 'barricade'}, {'id': 3, 'name': 'boat'}, {'id': 4, 'name': 'bus'}, {'id': 5, 'name': 'car'} , {'id': 6, 'name': 'cone'}, {'id': 7, 'name': 'dog'}, {'id': 8, 'name': 'door'}, {'id': 9, 'name': 'drain'}, {'id': 10, 'name': 'driveway'}, {'id': 11, 'name': 'emergency exit'}, {'id': 12, 'name': 'entrance'}, {'id': 13, 'name': 'excavator'}, {'id': 14, 'name': 'faregate'}, {'id': 15, 'name': 'fence'}, {'id': 16, 'name': 'fire'}, {'id': 17, 'name': 'fish'}, {'id': 18, 'name': 'guardrail'}, {'id': 19, 'name': 'helmet'}, {'id': 20, 'name': 'human'}, {'id': 21, 'name': 'jersey barrier'}, {'id': 22, 'name': 'junk'}, {'id': 23, 'name': 'lane'}, {'id': 24, 'name': 'leaves'}, {'id': 25, 'name': 'litter'}, {'id': 26, 'name': 'manhole'}, {'id': 27, 'name': 'motorcycle'}, {'id': 28, 'name': 'parking lot'}, {'id': 29, 'name': 'passage'}, {'id': 30, 'name': 'palanquin'}, {'id': 31, 'name': 'pipeline'}, {'id': 32, 'name': 'road marking'}, {'id': 33, 'name': 'ruler'}, {'id': 34, 'name': 'sidewalk'}, {'id': 35, 'name': 'smoke'}, {'id': 36, 'name': 'solar panel'}, {'id': 37, 'name': 'storage tank'}, {'id': 38, 'name': 'streetlight'}, {'id': 39, 'name': 'traffic light'}, {'id': 40, 'name': 'traffic sign'}, {'id': 41, 'name': 'tree'}, {'id': 42, 'name': 'truck'}, {'id': 43, 'name': 'vest'}, {'id': 44, 'name': 'weapon'}, {'id': 45, 'name': 'seat'}]


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

    orginal_number = len(image_list)
    image_qa_id_set = {x['image_id'] for x in anno_annos}
    image_keep_id_set = {x['id'] for x in image_annos if x['file_name'] in acd_de_dict}
    image_final_id_set = image_qa_id_set & image_keep_id_set
    image_final_id_map = {x: i for i, x in enumerate(image_final_id_set)}
    count["orginal"] += orginal_number
    count["qa"] += len(image_qa_id_set)
    count["deduplicated"] += len(image_keep_id_set)
    count["final"] += len(image_final_id_set)

    # image_new_annos = []
    # anno_new_annos = []
    # for x in image_annos:
    #     if x['id'] not in image_final_id_set:
    #         continue
    #     x['id'] = image_final_id_map[x['id']]
    #     image_new_annos.append(x)
    # for x in anno_annos:
    #     if x['image_id'] not in image_final_id_set:
    #         continue
    #     x['image_id'] = image_final_id_map[x['image_id']]
    #     anno_new_annos.append(x)

    image_new_annos = [x for x in image_annos if x['id'] in image_final_id_set]
    anno_new_annos = [x for x in anno_annos if x['image_id'] in image_final_id_set]
    for x in image_new_annos:
        x['id'] = image_final_id_map[x['id']]
    for x in anno_new_annos:
        x['image_id'] = image_final_id_map[x['image_id']]

    for image in tqdm.tqdm(image_new_annos):
        org_image_path = acd_de_dict[image['file_name']][0]
        dst_image_path = pathlib.Path(org_image_path.replace('/mnt/lighthouseACD/ACD-gdino-COCO', output_root))
        dst_image_path.parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            shutil.copy(org_image_path, dst_image_path)
    print(f'{len(image_new_annos)=}, {dst_image_path.parent.parent}')
    # count += len(image_new_annos)
    new_anno_file = dst_image_path.parent.parent / 'annotations' / 'labels.json'
    new_anno_file.parent.mkdir(parents=True, exist_ok=True)
    print(count)

    if not dry_run:
        coco_anno_dict = {
            "info": {'year': '', 'version': '', 'description': '', 'contributor': '', 'url': '', 'date_created': ''},
            "licenses": [],
            "categories": categories,
        }
        coco_anno_dict["images"] = image_new_annos
        coco_anno_dict["annotations"] = anno_new_annos

    with open(new_anno_file, 'w') as f:
        json.dump(coco_anno_dict, f)

print(count)
import ipdb; ipdb.set_trace()


'''
    for image in tqdm.tqdm(image_annos):
        if not image['id'] in image_final_id_set:
            continue

        bad_condition = image['id'] not in image_id_set
        if bad_condition:
            continue    
        if len(acd_de_dict[image.name]) == 1:
            org_image_path = acd_de_dict[image.name][0]
        else:
            print("found duplicate image name")
            exit(0)
        # elif len(vlm_de_dict[image.name]) == 2:
        #     print(f"{image.name=} has dupliacted")
        #     org_image_path = dupl_name_to_path[slice_folder.name]
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
'''