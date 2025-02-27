import json
import pathlib

# anno_root = '/mnt/data-home/mobility-multimodal/vlm-annotations/Sport_Development/20241223/Sport_Development_20241223_curated_t7/annotations/'
anno_root = '/mnt/data-home/mobility-multimodal/\
vlm-annotations/Sports_Development/20250109/\
Sports_Development_20250109_llava-onevision-0.5b-full'
anno_json_list = list(pathlib.Path(anno_root).rglob('*.json'))

max_count, min_count = float('-inf'), float('inf')
max_id, min_id = -1, -1
for anno_json in anno_json_list:
    with open(anno_json) as f:
        annos = json.load(f)
        for anno in annos:
            image_id = anno['id']
            answer = anno['conversations'][0]['answer']['groundtruth']
            word_count = len(answer.split())
            if word_count > max_count:
                max_count = word_count
                max_id = image_id
            if word_count < min_count:
                min_count = word_count
                min_id = image_id
print(f'{max_count=}, {min_count=}')
print(f'{max_id=}, {min_id=}')
import ipdb; ipdb.set_trace()