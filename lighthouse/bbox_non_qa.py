import pathlib
import copy

from common import get_depart, DINO_COCO_SPLITS_0418, DINO_COCO_SPLITS_0508, DINO_COCO_SPLITS_0703



def get_split_name(folder):
    return ('/').join(str(folder).split('/')[-2:])

uploaded_root_to_pattern = {
    "/mnt/data-home/mobility-multimodal/revised_bbox/deduplicated": "*/*",
    "/mnt/data-home/mobility-multimodal/revised_bbox/datasets": "*/*",
    "/mnt/lighthouseACD/ACD-gdino-COCO/Transportation_20250115_image_list_keep_0.95_rededuplicate": "*",
}
qaed_merged_root_to_pattern = { 
    "/mnt/lighthouseACD/QAed-data/bbox/hand0422/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0428/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0521/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0526-iou38/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/upload0527-iou38/": "*/*",
    "/mnt/lighthouseACD/QAed-data/bbox/hand0626-iou38/": "*/*/*",
}

prev_uploaded_folders = copy.deepcopy(DINO_COCO_SPLITS_0418)
prev_uploaded_folders.extend(copy.deepcopy(DINO_COCO_SPLITS_0508))
for data_root, pattern in uploaded_root_to_pattern.items():
    folder_list = list(pathlib.Path(data_root).glob(pattern))
    prev_uploaded_folders.extend(folder_list)


qaed_merged_folder_list = []
for data_root, pattern in qaed_merged_root_to_pattern.items():
    folder_list = list(pathlib.Path(data_root).glob(pattern))
    qaed_merged_folder_list.extend(folder_list)
qaed_merged_split_set = {get_split_name(x) for x in qaed_merged_folder_list}

not_qa_folder_list = [x for x in prev_uploaded_folders if get_split_name(x) not in qaed_merged_split_set]

public_works_folder_list = [x for x in not_qa_folder_list if get_depart(x)=="Public_Works"]
public_works_folder_list = sorted(public_works_folder_list)
JULY_SPLIT_LIST = copy.deepcopy(public_works_folder_list[:64])
others_folder_list = [x for x in not_qa_folder_list if get_depart(x) in ["Sports_Development", "Mass_Rapid_Transit", "Ports_Corporation"]]

JULY_SPLIT_LIST.extend(others_folder_list)
JULY_SPLIT_LIST.extend(copy.deepcopy(DINO_COCO_SPLITS_0703))

numbers = {
    "Public_Works": 0,
    "Sports_Development": 0,
    "Mass_Rapid_Transit": 0,
    "Ports_Corporation": 0,
    "Transportation": 0,
}
for folder in JULY_SPLIT_LIST:
    image_list = list((folder/"images").glob("*"))
    numbers[get_depart(folder)] += len(image_list)
print(numbers)



AUG_FOLDER_LIST = copy.deepcopy(public_works_folder_list[64:])