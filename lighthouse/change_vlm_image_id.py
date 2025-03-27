import pathlib
from common import VLM_CKPT1_FOLDERS 
from common import change_vlm_image_id

folder_list = VLM_CKPT1_FOLDERS
folder_list = [pathlib.Path(str(x).replace("/vlm-annotations/", "/checkpoint/vlm/hand/")) for x in folder_list]


start_id = 93450 # for ckpt2 QAed
for folder in folder_list:
    if not folder.exists():
        print(f"{folder} not exist")
        continue
    # print(folder)
    start_id = change_vlm_image_id(folder, start_id=start_id)
    print(start_id)

