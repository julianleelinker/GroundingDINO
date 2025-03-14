import tqdm
from common import VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, DEPARTS_EN, DINO_COCO_FOLDERS
from common import get_depart


for folder_list, name in [
        (VLM_CKPT1_FOLDERS, 'VLM CKPT1'),
        (VLM_CKPT2_FOLDERS, 'VLM CKPT2'),
    ]:
    print(f'counting {name} data ...')
    depart_count = {depart: 0 for depart in DEPARTS_EN}
    for folder in tqdm.tqdm(folder_list):
        image_list = list((folder/'images').glob('*'))
        depart_count[get_depart(folder)] += len(image_list)

    total_count = 0
    for key, value in depart_count.items():
        total_count += value
        print(f'{key:<32}: {value:>16,}')
    print(f'{"Total":<32}: {total_count:>16,}\n')


print('counting DINO COCO data...')
depart_count = {depart: 0 for depart in DEPARTS_EN}
for folder in tqdm.tqdm(DINO_COCO_FOLDERS):
    splits = list(folder.glob('split*'))
    for split in splits:
        image_list = list((split/'images').glob('*'))
        depart_count[get_depart(folder)] += len(image_list)

total_count = 0
for key, value in depart_count.items():
    total_count += value
    print(f'{key:<32}: {value:>16,}')
print(f'{"Total":<32}: {total_count:>16,}')