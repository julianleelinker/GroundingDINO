import pathlib
import json
import tqdm
import shutil
from collections import defaultdict


acd_de_json = "/mnt/data-home/chungan/acd_ckpt1_image_list_keep_0.95.json"
with open(acd_de_json, 'r') as f:
    acd_de_list = json.load(f)

acd_de_dict = defaultdict(list)
for x in acd_de_list:
    acd_de_dict[pathlib.Path(x).name].append(x)

prefix = "/mnt/lighthouseACD/ACD-gdino-COCO/Public_Works_20241230_image_list_keep_0.95"

output_root = "/mnt/data-home/mobility-multimodal/checkpoint/bbox/hand"
# slice_root = "/mnt/data-home/mobility-multimodal/checkpoint/bbox/dataslices"
slice_folder_list = [f"{prefix}/split{i}_0.30_0.35" for i in range(30,91)]
slice_folder_list = [pathlib.Path(x) for x in slice_folder_list] 
slice_folder_list = slice_folder_list[:1]

accumulate_count = 0
new_anno_list_dict = {}
dry_run = False

coco_anno_dict = {
    "info": {'year': '', 'version': '', 'description': '', 'contributor': '', 'url': '', 'date_created': ''},
    "licenses": [],
    "categories": [{'id': 0, 'name': 'algae'}, {'id': 1, 'name': 'animal'}, {'id': 2, 'name': 'barricade'}, {'id': 3, 'name': 'boat'}, {'id': 4, 'name': 'bus'}, {'id': 5, 'name': 'car'} , {'id': 6, 'name': 'cone'}, {'id': 7, 'name': 'dog'}, {'id': 8, 'name': 'door'}, {'id': 9, 'name': 'drain'}, {'id': 10, 'name': 'driveway'}, {'id': 11, 'name': 'emergency exit'}, {'id': 12, 'name': 'entrance'}, {'id': 13, 'name': 'excavator'}, {'id': 14, 'name': 'faregate'}, {'id': 15, 'name': 'fence'}, {'id': 16, 'name': 'fire'}, {'id': 17, 'name': 'fish'}, {'id': 18, 'name': 'guardrail'}, {'id': 19, 'name': 'helmet'}, {'id': 20, 'name': 'human'}, {'id': 21, 'name': 'jersey barrier'}, {'id': 22, 'name': 'junk'}, {'id': 23, 'name': 'lane'}, {'id': 24, 'name': 'leaves'}, {'id': 25, 'name': 'litter'}, {'id': 26, 'name': 'manhole'}, {'id': 27, 'name': 'motorcycle'}, {'id': 28, 'name': 'parking lot'}, {'id': 29, 'name': 'passage'}, {'id': 30, 'name': 'palanquin'}, {'id': 31, 'name': 'pipeline'}, {'id': 32, 'name': 'road marking'}, {'id': 33, 'name': 'ruler'}, {'id': 34, 'name': 'sidewalk'}, {'id': 35, 'name': 'smoke'}, {'id': 36, 'name': 'solar panel'}, {'id': 37, 'name': 'storage tank'}, {'id': 38, 'name': 'streetlight'}, {'id': 39, 'name': 'traffic light'}, {'id': 40, 'name': 'traffic sign'}, {'id': 41, 'name': 'tree'}, {'id': 42, 'name': 'truck'}, {'id': 43, 'name': 'vest'}, {'id': 44, 'name': 'weapon'}, {'id': 45, 'name': 'seat'}],
}



count = {
    "orginal": 0,
    # "qa": 0,
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
    # image_qa_id_set = {x['image_id'] for x in anno_annos}
    image_keep_id_set = {x['id'] for x in image_annos if x['file_name'] in acd_de_dict}
    image_final_id_set = image_keep_id_set
    image_final_id_map = {x: i for i, x in enumerate(image_final_id_set)}
    count["orginal"] += orginal_number
    # count["qa"] += len(image_qa_id_set)
    count["deduplicated"] += len(image_keep_id_set)
    count["final"] += len(image_final_id_set)

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

coco_anno_dict["images"] = image_new_annos
coco_anno_dict["annotations"] = anno_new_annos

if not dry_run:
    with open(new_anno_file, 'w') as f:
        json.dump(coco_anno_dict, f)

print(count)
import ipdb; ipdb.set_trace()
exit(0)