import json


sources_by_depart = {
    'Public_Works': 0,
    'Water_Resources': 0,
    'Transportation': 0,
    'Mass_Rapid_Transit': 0,
    'Sports_Development': 0,
    'Taiwan_Power': 0,
    'Linker_Vision_Data_V2': 0,
    'Kaohsiung_Data_V2': 0,
}


json_list_jan = [
'/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Transportation/20241230/Transportation_20241230_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250109/Transportation_20250109_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95.json',
'/mnt/data-home/mobility-multimodal/data-curation/Kaohsiung_Data_V2/Kaohsiung_Data_V2_image_list_keep_0.95.json',
]

# count = 0
for json_path in json_list_jan:
    with open(json_path, 'r') as f:
        image_list = json.load(f)
    print(f'{json_path=}')
    print(len(image_list))
    for key, val in sources_by_depart.items():
        if key in json_path:
            sources_by_depart[key] += len(image_list)
            # count += len(image_list)
            break
print('count:')
for key, val in sources_by_depart.items():
    print(f'{key}, {val}')
import ipdb; ipdb.set_trace()