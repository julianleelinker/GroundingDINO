import pathlib
import json


DEPART_MAP = {
    'China_Steel'           : '運發局',
    'Mass_Rapid_Transit'    : '水利局', 
    'Ports_Corporation'     : '交通局',      
    'Public_Works'          : '捷運局',         
    'Sports_Development'    : '台電',         
    'Taiwan_Power'          : '工務局',         
    'Transportation'        : '中鋼',          
    'Water_Resources'       : '港務局',     
    'Kaohsiung-full-dataset': 'Linker',
    'Linker_Vision_Data_V3' : 'LinkerV3',
}
DEPARTS_CH = list(DEPART_MAP.values())
DEPARTS_EN = list(DEPART_MAP.keys())

VLM_ANNOTATION_ROOT = '/mnt/data-home/mobility-multimodal/vlm-annotations'
DATA_CURATION_ROOT = '/mnt/data-home/mobility-multimodal/data-curation'

CKPT1_LINKER_FOLDERS = [f'Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part{i}_{j}' for i in range(1, 6) for j in range(1, 5)]
CKPT1_LINKER_FOLDERS.extend([f'Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part6_{j}' for j in range(1, 4)])
CKPT1_LINKER_FOLDERS = [pathlib.Path(f'{VLM_ANNOTATION_ROOT}/{folder}') for folder in CKPT1_LINKER_FOLDERS]

VLM_CKPT1_FOLDERS = [
    'Sports_Development/20241223/Sports_Development_20241223_curated_t7',
    'Sports_Development/20241223/Sports_Development_20241223_curated_t5_VLA_patch',
    'Sports_Development/20250109/Sports_Development_20250109_llava-onevision-0.5b-full',
    'Water_Resources/20250106/Water_Resources_20250106_curated_t5',
    'Transportation/20250109/Transportation_20250109_curated_t7',
    'Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95',
    'Transportation/20241230/Transportation_20241230_curated_t7',
    'Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95',
    'Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95',
    'Public_Works/20241230/Public_Works_20241230_curated_t5_part',
    #  patch data
    'Sports_Development/20241223/Sports_Development_20241223_curated_t6_VLM_100000_patch',
    'Transportation/20250115/Transportation_20250115_curated_t1',
    'Transportation/20250120/Transportation_20250120_llava-onevision-0.5b-full',
    'Water_Resources/20250106/Water_Resources_20250106_curated_t8_VLM_100000_patch',
    'Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split0',
    'Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split1',
    'Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split2',
    'Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split3',
]
VLM_CKPT1_FOLDERS = [pathlib.Path(f'{VLM_ANNOTATION_ROOT}/{folder}') for folder in VLM_CKPT1_FOLDERS]
VLM_CKPT1_FOLDERS.extend(CKPT1_LINKER_FOLDERS)

VLM_CKPT2_FOLDERS = [
    'Sports_Development/20241223/Sports_Development_20241223_curated_t4-revised', 
    'Sports_Development/20250213/Sports_Development_20250213_image_list_keep_0.95',
    'Mass_Rapid_Transit/20250213/Mass_Rapid_Transit_20250213_curated_t7',
    'Public_Works/20250206/Public_Works_20250206_curated_t4',
    ### re running gpt
    'China_Steel/20250226/China_Steel_20250226_image_list_keep_0.95',
    'Water_Resources/20250213/Water_Resources_20250213_curated_t1',
    'Transportation/20250115/Transportation_20250115_curated_t4', 
    ###
    'Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t14',
    'Linker_Vision_Data_V3/Linker_Vision_Data_V3_curated_t6_new',
]
VLM_CKPT2_FOLDERS = [pathlib.Path(f'{VLM_ANNOTATION_ROOT}/{folder}') for folder in VLM_CKPT2_FOLDERS]

DINO_COCO_ROOT = '/mnt/lighthouseACD/ACD-gdino-COCO'
DINO_COCO_FOLDERS = [
    'Mass_Rapid_Transit_20250109_image_list_keep_0.95_0.30_0.35',
    'Ports_Corporation_20250124_image_list_keep_0.95',
    'Public_Works_20241230_image_list_keep_0.95',
    'Public_Works_20250106_image_list_keep_0.95',
    'Sports_Development_20241223_image_list_keep_0.95',
    'Sports_Development_20250109_image_list_keep_0.95',
    'Taiwan_Power_20250106_image_list_keep_0.95_0.30_0.35',
    'Transportation_20241230_image_list_keep_0.95',
    'Transportation_20250109_image_list_keep_0.95',
    'Transportation_20250115_image_list_keep_0.95_rededuplicate',
    'Water_Resources_20250106_image_list_keep_0.95',
]
DINO_COCO_FOLDERS = [pathlib.Path(f'{DINO_COCO_ROOT}/{folder}') for folder in DINO_COCO_FOLDERS]

def get_depart(path: str | pathlib.Path, ch: bool=False) -> str | None:
    for depart, depart_ch in DEPART_MAP.items():
        if depart in str(path):
            if ch:
                return depart_ch
            return depart
    return None

def change_vlm_image_id(path: str | pathlib.Path, start_id: int=1) -> int:
    anno_path = pathlib.Path(path) / 'annotations' / 'vlm_annotations.json'
    new_id = start_id
    with anno_path.open('r') as f:
        anno_data = json.load(f)
    for data in anno_data:
        data['id'] = new_id
        new_id += 1
    with anno_path.open('w') as f:
        json.dump(anno_data, f, indent=4, ensure_ascii=False)
    return new_id