import json


# file_path = '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json'
file_path = '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95_rededuplicate.json'

with open(file_path, 'r') as f:
    data = json.load(f)

file_list = [
    '/mnt/Jan/Water_Resources/20250106/凱米颱風-水位站/月世界抽水平台/20240726/020023.jpg',
    '/mnt/Jan/Water_Resources/20250106/山陀兒颱風-水位站/月世界抽水平台/20241002/121451.jpg',
    '/mnt/Jan/Water_Resources/20250106/山陀兒颱風-水位站/月世界抽水平台/20241001/213742.jpg',
    '/mnt/Jan/Water_Resources/20250106/山陀兒颱風-水位站/月世界抽水平台/20240929/044713.jpg',
    '/mnt/Jan/Water_Resources/20250106/山陀兒颱風-水位站/月世界抽水平台/20241001/193457.jpg',
]

for file in file_list:
    if file in data:
        print(file)

import ipdb; ipdb.set_trace()
