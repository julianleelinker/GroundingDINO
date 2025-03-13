import pathlib
from common import CKPT1_VLM_FOLDERS, DEPART_MAP

for folder in CKPT1_VLM_FOLDERS:
    image_list = list(pathlib.Path(folder/'images').glob('*'))
    print(folder)
    print(len(image_list))