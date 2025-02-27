import pathlib
import json
import shutil
import tqdm


# root_path = 'mnt/d/mo/v/Kaohsiung-full-dataset/Kaoshsiung_76152_retrieval_curated_t220_part1'
# root_path_prefix = '/mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/Kaoshsiung_76152_retrieval_curated_t220_part'
root_path_list = [
    '/mnt/data-home/mobility-multimodal/vlm-annotations/Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17',
]
# root_path_list = [f'{root_path_prefix}{i}' for i in range(2, 7)]

ANNO_PER_SPLIT = 4

for root_path in root_path_list:
    print(root_path)
    anno_root = pathlib.Path(root_path) / 'annotations'
    anno_file_list = sorted((anno_root.rglob('*.json')))
    for i, anno_file in enumerate(anno_file_list):
        split_idx = i//ANNO_PER_SPLIT
        sub_root_path = pathlib.Path(f'{root_path}_split{split_idx}')
        (sub_root_path / 'annotations').mkdir(exist_ok=True, parents=True)
        (sub_root_path / 'images').mkdir(exist_ok=True, parents=True)
        with open(anno_file, 'r') as f:
            anno_data = json.load(f)
        for anno in tqdm.tqdm(anno_data):
            image_name = anno['image']
            src_img = pathlib.Path(root_path) / 'images' / image_name
            dst_img = sub_root_path / 'images' / image_name
            shutil.copy(src_img, dst_img)
        shutil.copy(anno_file, sub_root_path / 'annotations' / anno_file.name)
        print(f'{sub_root_path=}')
        # for anno in anno_data: