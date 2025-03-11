import json
import pathlib
import tqdm


prefix = '/mnt/data-home/mobility-multimodal/vlm-annotations'
ckpt1_vlm_folders = [
    "Sports_Development/20241223/Sports_Development_20241223_curated_t7-revised",
    "Sports_Development/20241223/Sports_Development_20241223_curated_t5_VLA_patch-revised",
    "Sports_Development/20250109/Sports_Development_20250109_llava-onevision-0.5b-full-revised",
    "Water_Resources/20250106/Water_Resources_20250106_curated_t5-revised",
    "Transportation/20250109/Transportation_20250109_curated_t7-revised",
    "Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95-revised",
    "Transportation/20241230/Transportation_20241230_curated_t7-revised",
    "Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95-revised",
    "Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95-revised",
    "Public_Works/20241230/Public_Works_20241230_curated_t5_part-revised",
    # patch data
    "Sports_Development/20241223/Sports_Development_20241223_curated_t6_VLM_100000_patch",
    "Transportation/20250115/Transportation_20250115_curated_t1",
    "Transportation/20250120/Transportation_20250120_llava-onevision-0.5b-full",
    "Water_Resources/20250106/Water_Resources_20250106_curated_t8_VLM_100000_patch",
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split0",
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split1",
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split2",
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split3",
]
# add linker data
ij_list = []
for i in range(1, 6):
    for j in range(1, 5):
        ij_list.append((i, j))
i = 6
for j in range(1, 4):
    ij_list.append((i, j))
for i, j in ij_list:
    ckpt1_vlm_folders.append(f'Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part{i}_{j}-revised')
ckpt1_vlm_folders = [f'{prefix}/{folder}' for folder in ckpt1_vlm_folders]


full_image_list = []
for folder in ckpt1_vlm_folders:
    image_root = pathlib.Path(folder) / 'images'
    print(image_root)
    assert image_root.exists(), f'{image_root} not exists'
    image_list = list(image_root.glob("*"))
    for image_path in tqdm.tqdm(image_list):
        full_image_list.append(str(image_path))

print(len(full_image_list))
with open('checkpoint1_image_list.json', 'w') as f:
    json.dump(full_image_list, f, ensure_ascii=False)

import ipdb; ipdb.set_trace()

