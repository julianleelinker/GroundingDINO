from common import VLM_CKPT2_FOLDERS
from common import change_vlm_image_id

folder_list = VLM_CKPT2_FOLDERS
for folder in folder_list:
    print(folder)
    print(change_vlm_image_id(folder))
