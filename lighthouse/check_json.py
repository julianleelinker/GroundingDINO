import json
from common import VLM_ANNOTATION_ROOT

json_path_list =[
    'Sports_Development/20250213/Sports_Development_20250213_image_list_keep_0.95',
    'Sports_Development/20241223/Sports_Development_20241223_curated_t4-revised', 
    'Mass_Rapid_Transit/20250213/Mass_Rapid_Transit_20250213_curated_t7',
    'Public_Works/20250206/Public_Works_20250206_curated_t4',
]
json_path_list = [f'{VLM_ANNOTATION_ROOT}/{folder}/annotations/vlm_annotations.json' for folder in json_path_list]
for json_path in json_path_list:
    print(json_path)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        print(f'{len(json_data)=}\n')