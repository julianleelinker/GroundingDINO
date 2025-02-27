import json
import tqdm
import pathlib
import shutil


# json_path = '/mnt/data-home/raytan/MetaCLIP/03_curated_Sports_Development_20241223_t7.json'
json_path = '/mnt/data-home/mobility-multimodal/data-curation/Sport_Development/20241223/Sport_Development_20241223_curated_t7.json'
existing_anno_path = pathlib.Path('/mnt/data-home/julian/lighthouse/anno0103/03_curated_Sports_Development_20241223_t11')
# output_anno_root = pathlib.Path('/mnt/data-home/julian/lighthouse/anno0103/Sport_Development_20241223_curated_t7')
output_anno_root = pathlib.Path('/mnt/data-home/mobility-multimodal/vlm-annotations/Sport_Development/20241223/Sport_Development_20241223_curated_t7')
images_per_anno = 1000 # save a json for each target image 
start_id = 1

output_image_root  = output_anno_root / 'images'
output_anno_root  = output_anno_root / 'annotations'
output_image_root.mkdir(exist_ok=True, parents=True)
output_anno_root.mkdir(exist_ok=True, parents=True)


with open(json_path, 'r') as f:
    json_data = json.load(f)
image_path_list = [pathlib.Path(image['image_path']).name for image in json_data]
anno_path_list = sorted((existing_anno_path / 'annotations').rglob('*.json'))
print(f'extracting {len(image_path_list)=}')

anno_dict = {}
for anno_path in anno_path_list:
    with open(anno_path, 'r') as f:
        anno_data = json.load(f)
    tmp_anno_dict = {anno['image']: anno for anno in anno_data}
    anno_dict.update(tmp_anno_dict)

not_found_list = []
filtered_anno = []
for id, image_path in enumerate(image_path_list):
    try:
        anno = anno_dict[image_path]
        filtered_anno.append(anno)
    except KeyError:
        print(f'not found image: {image_path=}')
        not_found_list.append(image_path)
        pass
print(f'{len(filtered_anno)=}')
print(f'{len(not_found_list)=}')

for id, anno in enumerate(filtered_anno):
    anno['id'] = id + start_id

print('copying images...')
for anno in tqdm.tqdm(filtered_anno):
    src_file = existing_anno_path / 'images' / anno['image']
    dst_file = output_image_root / src_file.name
    shutil.copy2(src_file, dst_file)

for i in range(images_per_anno, len(filtered_anno), images_per_anno):
    n_anno = int(i/images_per_anno)
    anno_path = output_anno_root / f'vlm_annotations_{n_anno}.json'
    with open(anno_path, "w") as json_file:
        json.dump(filtered_anno[(n_anno-1)*images_per_anno:i], json_file, indent=4, ensure_ascii=False)
    print(f'data save to {anno_path}')

n_anno = len(filtered_anno)//images_per_anno + 1
anno_path = output_anno_root / f'vlm_annotations_{n_anno}.json'
with open(anno_path, "w") as json_file:
    json.dump(filtered_anno[(n_anno-1)*images_per_anno:len(filtered_anno)], json_file, indent=4, ensure_ascii=False)
print(f'data save to {anno_path}')

import ipdb; ipdb.set_trace()