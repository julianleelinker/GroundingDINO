import pathlib
import json

path = pathlib.Path('/mnt/data-home/mobility-multimodal/vlm-annotations/Water_Resources/20250106/Water_Resources_20250106_curated_t5')
anno_root = path / 'annotations'
json_list = list(anno_root.rglob('vlm_annotations*.json'))
print(len(json_list))

all_data = []
for json_path in json_list:
    with json_path.open() as f:
        data = json.load(f)
    all_data.extend(data)
import ipdb; ipdb.set_trace()

new_anno_path = anno_root / 'new_anno_0306.json'
with (new_anno_path).open('w') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)



with (new_anno_path).open('r') as f:
    data = json.load(f)
print(len(data))