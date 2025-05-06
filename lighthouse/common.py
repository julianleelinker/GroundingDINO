import os
import shutil
import pathlib
import json
import tqdm
import pandas as pd


DATAVERSE_PASSWORD = os.environ.get("DATAVERSE_PASSWORD")
DATAVERSE_CKPT1_PROJECT_ID = 230 
DATAVERSE_CKPT2_PROJECT_ID = 464 
DATAVERSE_LVM300K_PROJECT_ID = 225
DATAVERSE_CURATION_HOST = "https://visionai.linkervision.ai/dataverse/curation"
DATAVERSE_EMAIL = "julianlee@linkervision.com"
DATAVERSE_SERVICE_ID_QA = "2bd928e5-a98f-4aae-a093-8545c57c103f"
DATAVERSE_SERVICE_ID_HAND = "697aa90b-00d0-4455-8863-bd2ad70a93e7"

DEPART_MAP = {
    "China_Steel"           : ("China_Steel"       , "中鋼", ),
    "Mass_Rapid_Transit"    : ("Mass_Rapid_Transit", "捷運局",),
    "Ports_Corporation"     : ("Ports_Corporation" , "港務局",),
    "Public_Works"          : ("Public_Works"      , "工務局",),
    "Sports_Development"    : ("Sports_Development", "運發局",),
    "Taiwan_Power"          : ("Taiwan_Power"      , "台電",  ),
    "Transportation"        : ("Transportation"    , "交通局",),
    "Water_Resources"       : ("Water_Resources"   , "水利局",),

    "Kaohsiung-full-dataset": ("Linker"            , "Linker",),
    "Linker_Vision_Data_V3" : ("Linker"            , "Linker",),
}

#  英文名                     中文名       縮寫
#  "China_Steel"             "中鋼",      cs      
#  "Mass_Rapid_Transit"      "捷運局",    mrt
#  "Ports_Corporation"       "港務局",     pc
#  "Public_Works"            "工務局",     pw
#  "Sports_Development"      "運發局",     sd
#  "Taiwan_Power"            "台電",       tp      
#  "Transportation"          "交通局",     tr
#  "Water_Resources"         "水利局",     wr
#  "Kaohsiung-full-dataset"  "Linker",    lk
#  "Linker_Vision_Data_V3"   "LinkerV3",  lk3

seen = [set(), set()]
DEPARTS = [[], []]
for x in DEPART_MAP.values():
    for i in range(2):
        if x[i] in seen[i]:
            continue
        seen[i].add(x[i])
        DEPARTS[i].append(x[i])
DEPARTS_EN, DEPARTS_CH = DEPARTS[0], DEPARTS[1]


VLM_ANNOTATION_ROOT = "/mnt/data-home/mobility-multimodal/vlm-annotations"
DATA_CURATION_ROOT = "/mnt/data-home/mobility-multimodal/data-curation"

CKPT1_LINKER_FOLDERS = [f"Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part{i}_{j}" for i in range(1, 6) for j in range(1, 5)]
CKPT1_LINKER_FOLDERS.extend([f"Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part6_{j}" for j in range(1, 4)])
CKPT1_LINKER_FOLDERS = [pathlib.Path(f"{VLM_ANNOTATION_ROOT}/{folder}") for folder in CKPT1_LINKER_FOLDERS]

VLM_CKPT1_FOLDERS = [
    "Sports_Development/20241223/Sports_Development_20241223_curated_t7",
    "Sports_Development/20241223/Sports_Development_20241223_curated_t5_VLA_patch",
    "Sports_Development/20250109/Sports_Development_20250109_llava-onevision-0.5b-full",
    "Water_Resources/20250106/Water_Resources_20250106_curated_t5",
    "Transportation/20250109/Transportation_20250109_curated_t7",
    "Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95",
    "Transportation/20241230/Transportation_20241230_curated_t7",
    "Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95",
    "Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95",
    "Public_Works/20241230/Public_Works_20241230_curated_t5_part",
    #  patch data
    "Sports_Development/20241223/Sports_Development_20241223_curated_t6_VLM_100000_patch", # remove union
    "Transportation/20250115/Transportation_20250115_curated_t1", # remove union
    "Transportation/20250120/Transportation_20250120_llava-onevision-0.5b-full",
    "Water_Resources/20250106/Water_Resources_20250106_curated_t8_VLM_100000_patch", # remove union
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split0",
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split1",
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split2",
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t17_split3",
    "Water_Resources/20250213/Water_Resources_20250213_curated_t1",
    "Transportation/20250115/Transportation_20250115_curated_t4", 
]
VLM_CKPT1_FOLDERS = [pathlib.Path(f"{VLM_ANNOTATION_ROOT}/{folder}") for folder in VLM_CKPT1_FOLDERS]
VLM_CKPT1_FOLDERS.extend(CKPT1_LINKER_FOLDERS)

VLM_CKPT2_FOLDERS = [
    "Sports_Development/20241223/Sports_Development_20241223_curated_t4-revised", 
    "Sports_Development/20250213/Sports_Development_20250213_image_list_keep_0.95",
    "Mass_Rapid_Transit/20250213/Mass_Rapid_Transit_20250213_curated_t7",
    "Public_Works/20250206/Public_Works_20250206_curated_t4",
    ### re running gpt
    "China_Steel/20250226/China_Steel_20250226_image_list_keep_0.95",
    ###
    "Ports_Corporation/20250124/Ports_Corporation_20250124_curated_t14",
    "Linker_Vision_Data_V3/Linker_Vision_Data_V3_curated_t6_new",
]
VLM_CKPT2_FOLDERS = [pathlib.Path(f"{VLM_ANNOTATION_ROOT}/{folder}") for folder in VLM_CKPT2_FOLDERS]

VLM_ADDDED_0407_FOLDERS = [
    "Sports_Development/20250226/Sports_Development_20250226_curated_t20",
    "Water_Resources/20250213/Water_Resources_20250213_curated_t2",
    "Water_Resources/20250324/Water_Resources_20250324_image_list_keep_0.95",
    "Mass_Rapid_Transit/20250213/Mass_Rapid_Transit_20250213_curated_t50",
    "Ports_Corporation/20250226/Ports_Corporation_20250226_curated_t11",
    "Transportation/20250304/Transportation_20250304_curated_t4",
    "Linker_Vision_Data_V3/20250408/Linker_Vision_Data_V3_curated_t26",
]
VLM_ADDDED_0407_FOLDERS = [pathlib.Path(f"{VLM_ANNOTATION_ROOT}/{folder}") for folder in VLM_ADDDED_0407_FOLDERS]

DINO_COCO_TARGET_ROOT = "/mnt/lighthouseACD/ACD-gdino-COCO"
DINO_COCO_DEPRECATED_FOLDERS = [
    "Mass_Rapid_Transit_20250109_image_list_keep_0.95_0.30_0.35",
    "Ports_Corporation_20250124_image_list_keep_0.95",
    "Public_Works_20241230_image_list_keep_0.95",
    "Public_Works_20250106_image_list_keep_0.95",
    "Sports_Development_20241223_image_list_keep_0.95",
    "Sports_Development_20250109_image_list_keep_0.95",
    "Taiwan_Power_20250106_image_list_keep_0.95_0.30_0.35",
    "Transportation_20241230_image_list_keep_0.95",
    "Transportation_20250109_image_list_keep_0.95",
    "Transportation_20250115_image_list_keep_0.95_rededuplicate",
    "Water_Resources_20250106_image_list_keep_0.95",
    # not used
    # "linker_4M_image_list_keep_0.95" 
]
DINO_COCO_DEPRECATED_FOLDERS = [pathlib.Path(f"{DINO_COCO_TARGET_ROOT}/{folder}") for folder in DINO_COCO_DEPRECATED_FOLDERS]

DINO_COCO_SOURCE_ROOT = "/mnt/lighthouseACD/augmented-curated-data-new"

AUGMENTED_CURATED_EXCLUDED_JSONS = {
    "Public_Works/20241230/split_image_list_keep/part_0_image_list_keep_0.95_llava-onevision-0.5b-part.json",
    "Public_Works/20241230/split_image_list_keep/part_0_image_list_keep_0.95.json",
    "Public_Works/20241230/split_image_list_keep/part_0_image_list_keep_0.95_llava-onevision-0.5b-full.json",
    "Public_Works/20241230/split_image_list_keep/part_1_image_list_keep_0.95_llava-onevision-0.5b-full.json",
    "Public_Works/20241230/split_image_list_keep/part_1_image_list_keep_0.95_llava-onevision-0.5b-part.json",
    "Public_Works/20241230/split_image_list_keep/part_1_image_list_keep_0.95.json",
    "Public_Works/20241230/Public_Works_20241230_final_image_list_keep_0.95.json",
    "Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95.json",
    "Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95_rededuplicate.json",
    "Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95.json",
    "Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95_rededuplicate.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_8.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_9.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_6.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_7.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_2.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_1.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_4.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_0.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_3.json",
    "Public_Works/20250206/raw_data/Public_Works_20250206_image_list_keep_0.95_part_5.json",
    "Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95_rededuplicate.json",
    "Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95.json",
    "Ports_Corporation/20250124/Ports_Corporation_20250124_image_list_keep_0.95.json",
    "Transportation/20241230/Transportation_20241230_image_list_keep_0.95.json",
    "Transportation/20241230/Transportation_20241230_image_list_keep_0.95_rededuplicate.json",
    "Transportation/20250115/Transportation_20250115_image_list_keep_0.95_rededuplicate.json",
    "Transportation/20250115/Transportation_20250115_image_list_keep_0.95.json",
    "Transportation/20250109/Transportation_20250109_image_list_keep_0.95.json",
    "Transportation/20250109/Transportation_20250109_image_list_keep_0.95_rededuplicate.json",
    "Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95_rededuplicate.json",
    "Mass_Rapid_Transit/20250109/Mass_Rapid_Transit_20250109_image_list_keep_0.95.json",
    "Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95_rededuplicate.json",
    "Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95.json",
    "Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95_rededuplicate.json",
    "Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95.json",
    "Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95.json",
    "Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95_rededuplicate.json",
    "Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95.json",
    "Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95_full.json",
    "Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95_deprecated.json",
    "Kaohsiung_Data_V2/Kaohsiung_Data_V2_image_list_keep_0.95.json",
}
AUGMENTED_CURATED_EXCLUDED_JSONS = {pathlib.Path(f"{DATA_CURATION_ROOT}/{x}") for x in AUGMENTED_CURATED_EXCLUDED_JSONS}
AUGMENTED_CURATED_JSONS_0418 = {
    # running 250411
    "Water_Resources/20250324/Water_Resources_20250324_image_list_keep_0.95.json",
    "Water_Resources/20250213/Water_Resources_20250213_image_list_keep_0.95.json",
    "Ports_Corporation/20250226/Ports_Corporation_20250226_image_list_keep_0.95.json",
    "Transportation/20250304/Transportation_20250304_image_list_keep_0.95.json",
    "Transportation/20250120/Transportation_20250120_image_list_keep_0.95.json",
    "Mass_Rapid_Transit/20250213/Mass_Rapid_Transit_20250213_image_list_keep_0.95.json",
    "Sports_Development/20250319/Sports_Development_20250319_image_list_keep_0.95.json",
    "Sports_Development/20250226/Sports_Development_20250226_image_list_keep_0.95.json",
    "Sports_Development/20250213/Sports_Development_20250213_image_list_keep_0.95.json",
    "China_Steel/20250226/China_Steel_20250226_image_list_keep_0.95.json",
}
AUGMENTED_CURATED_JSONS_0418 = {pathlib.Path(f"{DATA_CURATION_ROOT}/{x}") for x in AUGMENTED_CURATED_JSONS_0418}
DINO_COCO_SPLITS_0418 = []
DINO_COCO_TARGET_ROOT_NEW = "/mnt/lighthouseACD/ACD-gdino-COCO-new"
DINO_COCO_FOLDERS_0418 =[
   pathlib.Path(DINO_COCO_TARGET_ROOT_NEW) / pathlib.Path(x).stem for x in AUGMENTED_CURATED_JSONS_0418
]
for folder in DINO_COCO_FOLDERS_0418:
    if folder.is_dir():
        split_list = list(folder.glob("split*"))
    DINO_COCO_SPLITS_0418.extend(split_list)

AUGMENTED_CURATED_EXCLUDED_JSONS = AUGMENTED_CURATED_EXCLUDED_JSONS \
    | AUGMENTED_CURATED_JSONS_0418


def get_depart(path: str | pathlib.Path, chout: bool=False, chin: bool=False) -> str | None:
    for depart, (depart_en, depart_ch) in DEPART_MAP.items():
        if (depart in str(path)) or (depart_ch in str(path)):
            if chout:
                return depart_ch
            return depart_en
    return None


def get_depart_date(path: str | pathlib.Path, ch: bool=False) -> tuple[str | None, str | None]:
    depart = get_depart(path)
    date = str(path).split(f"{depart}_")[1]
    if not ch:
        return depart, date
    return get_depart(path, chout=True), date


def change_vlm_image_id(path: str | pathlib.Path, start_id: int=1) -> int:
    anno_path = pathlib.Path(path) / "annotations" / "vlm_annotation.json"
    new_id = start_id
    with anno_path.open("r") as f:
        anno_data = json.load(f)
    for data in anno_data:
        data["id"] = new_id
        new_id += 1
    with anno_path.open("w") as f:
        json.dump(anno_data, f, indent=4, ensure_ascii=False)
    return new_id


def copy_images_in_image_list(image_list: list, dst_root: str | pathlib.Path, path_to_name = "name_to_path.txt", always_save_mapping: bool = False):
    assert not dst_root.exists(), "dst_root already exists"
    dst_root.mkdir(exist_ok=True, parents=True)
    os.chmod(dst_root, 0o777)
    name_to_path_map = []

    save_path_map = False
    for image_path in tqdm.tqdm(image_list):
        src_path = pathlib.Path(image_path)
        dst_path = pathlib.Path(dst_root) / src_path.name

        count = 0
        rename_path = dst_path
        while rename_path.exists():
            save_path_map = True
            print(f"{rename_path} already exists")
            count += 1
            rename_path = rename_path.parent / (f"{dst_path.stem}_{count}{rename_path.suffix}")

        name_to_path_map.append(f"'{rename_path.name}','{src_path}'")
        shutil.copy2(src_path, rename_path)

    if always_save_mapping or save_path_map:
        with open(dst_root / path_to_name, "w") as f:
            f.write("\"copied_name\",\"original_path\"\n")
            f.write("\n".join(name_to_path_map))


def copy_images_in_json(json_path: str | pathlib.Path, dst_root: str | pathlib.Path, is_image_list: bool=True, path_to_name = "name_to_path.txt", split_size: int=0):
    with open(json_path, "r") as f:
        image_list = json.load(f)
    if not is_image_list:
        image_list = [data["image_path"] for data in image_list]
    if split_size==0:
        copy_images_in_image_list(image_list, dst_root, path_to_name)
    else:
        splited_image_list = [image_list[i:i + split_size] for i in range(0, len(image_list), split_size)]
        for i, chunk in tqdm.tqdm(enumerate(splited_image_list), total=len(splited_image_list)):
            copy_images_in_image_list(chunk, dst_root / f"split{i}", path_to_name, always_save_mapping=True)


STATS_COLUMN_DTYPES = {
    "folder": "string",
    "depart": "string",
    "split": "string",
    "number": "Int64",               # nullable integer
    "annotated": "Int64",
    "qa_number": "Int64",
    "uploaded": "boolean",
    "dv": "boolean",
    "ckpt": "string",
    "notes": "string",
    "path": "string",
}


def load_stats_fwf(file_path, numeric_cols):
    with open(file_path.replace("txt", "json")) as f:
        col_widths = json.load(f)

    col_widths = list(col_widths.values())
    colspecs = []
    start = 0
    for w in col_widths:
        colspecs.append((start, start + w))
        start += w + 2  # 2-space gap between columns from `save_fwf`

    df_loaded = pd.read_fwf(
        file_path,
        colspecs=colspecs,
        dtype=STATS_COLUMN_DTYPES,
    )

    for col in numeric_cols:
        df_loaded[col] = df_loaded[col].str.replace(",", "", regex=False)

    df_loaded = df_loaded.astype(
        STATS_COLUMN_DTYPES
    )
    return df_loaded


def save_stats_fwf(df, numeric_cols, save_name):
    df_formatted = df.copy()
    for col in numeric_cols:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:,}")

    # Convert all to strings for alignment
    df_str = df_formatted.astype(str)

    # Calculate max width considering both column names and formatted values
    col_widths = {
        col: int(max(df_str[col].map(len).max(), len(col)))
        for col in df.columns
    }
    
    # save column width for future loading
    with open(save_name.replace("txt", "json"), "w") as f:
        json.dump(col_widths, f)

    # Align values: right for numbers, left for others
    def align_column(col_name, series):
        if col_name in numeric_cols:
            return series.str.rjust(col_widths[col_name])
        else:
            return series.str.ljust(col_widths[col_name])

    df_aligned = pd.DataFrame({
        col: align_column(col, df_str[col])
        for col in df.columns
    })

    # Create aligned header
    header = "  ".join([
        col.rjust(col_widths[col]) if col in numeric_cols else col.ljust(col_widths[col])
        for col in df.columns
    ])

    # Write to text file
    with open(save_name, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for _, row in df_aligned.iterrows():
            f.write("  ".join(row) + "\n")


if __name__ == "__main__":
    import ipdb; ipdb.set_trace()
    json_path = "/mnt/data-home/mobility-multimodal/data-curation/China_Steel/20250226/China_Steel_20250226_image_list_keep_0.95.json"
    dst_root = "/mnt/data-home/julian"
    dst_path = pathlib.Path(dst_root) / pathlib.Path(json_path).stem 
    copy_images_in_json(json_path, dst_path)
