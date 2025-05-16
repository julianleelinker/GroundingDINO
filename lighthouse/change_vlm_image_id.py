import pathlib
from common import VLM_CKPT1_FOLDERS 
from common import change_vlm_image_id

folder_list = VLM_CKPT1_FOLDERS
folder_list = [pathlib.Path(str(x).replace("/vlm-annotations/", "/checkpoint/vlm/hand/")) for x in folder_list]

remaining_list = [
    # # ckpt2
    # "/mnt/lighthouseACD/QAed-data//vlm/hand/Water_Resources/20250213/Water_Resources_20250213_curated_t1",
    # # ckpt1 remaining
    # "/mnt/lighthouseACD/QAed-data//vlm/hand/Water_Resources/20250106/Water_Resources_20250106_curated_t8_VLM_100000_patch",
    # "/mnt/lighthouseACD/QAed-data//vlm/hand/Transportation/20250120/Transportation_20250120_llava-onevision-0.5b-full",
    # "/mnt/lighthouseACD/QAed-data//vlm/hand/Sports_Development/20241223/Sports_Development_20241223_curated_t6_VLM_100000_patch",
    # patch ckpt2
    # "/mnt/lighthouseACD/QAed-data//vlm/hand/Transportation/20250115/Transportation_20250115_curated_t4/"
]
remaining_list = [pathlib.Path(x) for x in remaining_list]
folder_list = remaining_list
import ipdb; ipdb.set_trace()

start_id = 1 # image_id start from 1
start_id = 93450 # After ckpt1 wihtout 3 manually annotated
start_id = 99569 # After change id remaining_list 
start_id = 101698 # after patch

# After "Water_Resources_20250213_curated_t1", "Water_Resources_20250106_curated_t8_VLM_100000_patch", "Transportation_20250120_llava-onevision-0.5b-full", "Sports_Development_20241223_curated_t6_VLM_100000_patch"
for folder in folder_list:
    if not folder.exists():
        print(f"{folder} not exist")
        continue
    # print(folder)
    print(folder)
    # old_id = start_id
    start_id = change_vlm_image_id(folder, start_id=start_id)
    # print(start_id-old_id)
    print(start_id)

