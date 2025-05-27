import pathlib
import tqdm
import pandas as pd
import os 
import fire
from dataverse_sdk import *
from dataverse_sdk.connections import get_connection
from export_project_dataslices import export_dataslice_to_local
from common import DATAVERSE_PASSWORD, DATAVERSE_BBOX_QA_PROJECT_ID, DATAVERSE_CURATION_HOST, DATAVERSE_EMAIL, DATAVERSE_SERVICE_ID_QA, get_depart


SLICE_NAME_TO_DATASET_0428 = {
    "upload0415-pc-20250226-split7":   "upload0415_港務局_20250226_image_list_keep_0.95/split7_0.30_0.35" ,
    "upload0415-pc-20250226-split1":   "upload0415_港務局_20250226_image_list_keep_0.95/split1_0.30_0.35" ,
    "upload0415-wr-20250213-split5":   "upload0415_水利局_20250213_image_list_keep_0.95/split5_0.30_0.35" ,
    "upload0415-mrt-20250213-split5":  "upload0415_捷運局_20250213_image_list_keep_0.95/split5_0.30_0.35" ,
    "upload0415-mrt-20250213-split2":  "upload0415_捷運局_20250213_image_list_keep_0.95/split2_0.30_0.35" ,
    "upload0415-mrt-20250213-split3":  "upload0415_捷運局_20250213_image_list_keep_0.95/split3_0.30_0.35" ,
    "upload0415-wr-20250213-split0":   "upload0415_水利局_20250213_image_list_keep_0.95/split0_0.30_0.35" ,
    "upload0415-mrt-20250213-split4":  "upload0415_捷運局_20250213_image_list_keep_0.95/split4_0.30_0.35" ,
    "upload0415-cs-20250226-split0":   "upload0415_中鋼_20250226_image_list_keep_0.95/split0_0.30_0.35"   ,
    "upload0415-mrt-20250213-split0":  "upload0415_捷運局_20250213_image_list_keep_0.95/split0_0.30_0.35" ,
    "upload0415-mrt-20250213-split1":  "upload0415_捷運局_20250213_image_list_keep_0.95/split1_0.30_0.35" ,

    "upload0408-sd-20241223-split10":  "upload0408_運發局_20241223_image_list_keep_0.95/split10_0.30_0.35",
    "upload0408-mrt-20250109-split0":  "upload0408_捷運局_20250109_image_list_keep_0.95_0.30_0.35/split0" ,
    "upload0408-wr-20250106-split1":   "upload0408_水利局_20250106_image_list_keep_0.95/split1_0.30_0.35" ,
    "upload0408-wr-20250106-split5":   "upload0408_水利局_20250106_image_list_keep_0.95/split5_0.30_0.35" ,
    "upload0408-wr-20250106-split7":   "upload0408_水利局_20250106_image_list_keep_0.95/split7_0.30_0.35" ,
    "upload0408-wr-20250106-split11":  "upload0408_水利局_20250106_image_list_keep_0.95/split11_0.30_0.35",
    "upload0408-wr-20250106-split4":   "upload0408_水利局_20250106_image_list_keep_0.95/split4_0.30_0.35" ,
    "upload0408-wr-20250106-split3":   "upload0408_水利局_20250106_image_list_keep_0.95/split3_0.30_0.35" ,
    "upload0408-wr-20250106-split15":  "upload0408_水利局_20250106_image_list_keep_0.95/split15_0.30_0.35",
    "upload0408-wr-20250106-split8":   "upload0408_水利局_20250106_image_list_keep_0.95/split8_0.30_0.35" ,
    "upload0408-wr-20250106-split6":   "upload0408_水利局_20250106_image_list_keep_0.95/split6_0.30_0.35" ,
    "upload0408-wr-20250106-split14":  "upload0408_水利局_20250106_image_list_keep_0.95/split14_0.30_0.35",
    "upload0408-wr-20250106-split2":   "upload0408_水利局_20250106_image_list_keep_0.95/split2_0.30_0.35" ,
    "upload0408-wr-20250106-split9":   "upload0408_水利局_20250106_image_list_keep_0.95/split9_0.30_0.35" ,
    "upload0408-wr-20250106-split12":  "upload0408_水利局_20250106_image_list_keep_0.95/split12_0.30_0.35",
    "upload0408-wr-20250106-split13":  "upload0408_水利局_20250106_image_list_keep_0.95/split13_0.30_0.35",
    "upload0408-tp-20250106-split0":   "upload0408_台電_20250106_image_list_keep_0.95_0.30_0.35/split0"   ,
    "upload0408-pw-20241230-split1":   "upload0408_工務局_20241230_image_list_keep_0.95/split1_0.30_0.35" ,
    "upload0408-pw-20241230-split20":  "upload0408_工務局_20241230_image_list_keep_0.95/split20_0.30_0.35",
    "upload0408-pw-20241230-split5":   "upload0408_工務局_20241230_image_list_keep_0.95/split5_0.30_0.35" ,
    "upload0408-pw-20241230-split15":  "upload0408_工務局_20241230_image_list_keep_0.95/split15_0.30_0.35",
    "upload0408-pw-20241230-split31":  "upload0408_工務局_20241230_image_list_keep_0.95/split31_0.30_0.35",
    "upload0408-pw-20241230-split28":  "upload0408_工務局_20241230_image_list_keep_0.95/split28_0.30_0.35",
    "upload0408-pw-20241230-split7":   "upload0408_工務局_20241230_image_list_keep_0.95/split7_0.30_0.35" ,
    "upload0408-pw-20241230-split17":  "upload0408_工務局_20241230_image_list_keep_0.95/split17_0.30_0.35",
    "upload0408-pw-20241230-split11":  "upload0408_工務局_20241230_image_list_keep_0.95/split11_0.30_0.35",
    "upload0408-pw-20241230-split10":  "upload0408_工務局_20241230_image_list_keep_0.95/split10_0.30_0.35",
    "upload0408-pw-20241230-split6":   "upload0408_工務局_20241230_image_list_keep_0.95/split6_0.30_0.35" ,
    "upload0408-pw-20241230-split21":  "upload0408_工務局_20241230_image_list_keep_0.95/split21_0.30_0.35",
    "upload0408-pw-20241230-split27":  "upload0408_工務局_20241230_image_list_keep_0.95/split27_0.30_0.35",
    "upload0408-pw-20241230-split29":  "upload0408_工務局_20241230_image_list_keep_0.95/split29_0.30_0.35",
    "upload0408-pw-20241230-split25":  "upload0408_工務局_20241230_image_list_keep_0.95/split25_0.30_0.35",
    "upload0408-pw-20241230-split16":  "upload0408_工務局_20241230_image_list_keep_0.95/split16_0.30_0.35",
    "upload0408-pw-20241230-split26":  "upload0408_工務局_20241230_image_list_keep_0.95/split26_0.30_0.35",
    "upload0408-pw-20241230-split3":   "upload0408_工務局_20241230_image_list_keep_0.95/split3_0.30_0.35" ,
    "upload0408-pw-20241230-split9":   "upload0408_工務局_20241230_image_list_keep_0.95/split9_0.30_0.35" ,
    "upload0408-pw-20241230-split19":  "upload0408_工務局_20241230_image_list_keep_0.95/split19_0.30_0.35",
    "upload0408-pw-20241230-split8":   "upload0408_工務局_20241230_image_list_keep_0.95/split8_0.30_0.35" ,
    "upload0408-pw-20241230-split23":  "upload0408_工務局_20241230_image_list_keep_0.95/split23_0.30_0.35",
    "upload0408-pw-20241230-split13":  "upload0408_工務局_20241230_image_list_keep_0.95/split13_0.30_0.35",
    "upload0408-pw-20241230-split4":   "upload0408_工務局_20241230_image_list_keep_0.95/split4_0.30_0.35" ,
    "upload0408-pw-20241230-split12":  "upload0408_工務局_20241230_image_list_keep_0.95/split12_0.30_0.35",
    "upload0408-pw-20241230-split2":   "upload0408_工務局_20241230_image_list_keep_0.95/split2_0.30_0.35" ,
    "upload0408-sd-20241223-split2":   "upload0408_運發局_20241223_image_list_keep_0.95/split2_0.30_0.35" ,
    "upload0408-sd-20241223-split8":   "upload0408_運發局_20241223_image_list_keep_0.95/split8_0.30_0.35" ,
    "upload0408-sd-20241223-split7":   "upload0408_運發局_20241223_image_list_keep_0.95/split7_0.30_0.35" ,
    "upload0408-sd-20241223-split5":   "upload0408_運發局_20241223_image_list_keep_0.95/split5_0.30_0.35" ,
    "upload0408-sd-20241223-split4":   "upload0408_運發局_20241223_image_list_keep_0.95/split4_0.30_0.35" ,
    "upload0408-sd-20241223-split11":  "upload0408_運發局_20241223_image_list_keep_0.95/split11_0.30_0.35",
    "upload0408-sd-20241223-split12":  "upload0408_運發局_20241223_image_list_keep_0.95/split12_0.30_0.35",
    "upload0408-sd-20241223-split6":   "upload0408_運發局_20241223_image_list_keep_0.95/split6_0.30_0.35" ,
    "upload0408-sd-20241223-split3":   "upload0408_運發局_20241223_image_list_keep_0.95/split3_0.30_0.35" ,
    "upload0408-sd-20241223-split9":   "upload0408_運發局_20241223_image_list_keep_0.95/split9_0.30_0.35" ,
    "upload0408-sd-20241223-split1":   "upload0408_運發局_20241223_image_list_keep_0.95/split1_0.30_0.35" ,
    "upload0408-pw-split1":            "upload0408_工務局_20250106_image_list_keep_0.95/split1_0.30_0.35" ,
    "upload0408-sd-split1":            "upload0408_運發局_20250109_image_list_keep_0.95/split1_0.30_0.35" ,
    "upload0408-sd-split13":           "upload0408_運發局_20241223_image_list_keep_0.95/split13_0.30_0.35",
}


SLICE_NAME_TO_DATASET_0521 = {
"upload0408-tr-split13": "Transportation_20250109_image_list_keep_0.95/split13_0.30_0.35",
"upload0408-tr-split19": "Transportation_20250109_image_list_keep_0.95/split19_0.30_0.35",
"upload0408-tr-split12": "Transportation_20250109_image_list_keep_0.95/split12_0.30_0.35",
"transportation-20250115-split30": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split30_0.30_0.35",
"transportation-20250115-split34": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split34_0.30_0.35",
"transportation-20250115-split32": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split32_0.30_0.35",
"transportation-20250115-split33": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split33_0.30_0.35",
"transportation-20250115-split40": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split40_0.30_0.35",
"transportation-20250115-split24": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split24_0.30_0.35",
"transportation-20250115-split29": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split29_0.30_0.35",
"transportation-20250115-split27": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split27_0.30_0.35",
"transportation-20250115-split20": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split20_0.30_0.35",
"transportation-20250115-split25": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split25_0.30_0.35",
"transportation-20250115-split21": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split21_0.30_0.35",
"transportation-20250115-split22": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split22_0.30_0.35",
"transportation-20250115-split23": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split23_0.30_0.35",
"transportation-20250115-split16": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split16_0.30_0.35",
"transportation-20250115-split26": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split26_0.30_0.35",
"transportation-20250115-split18": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split18_0.30_0.35",
"transportation-20250115-split17": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split17_0.30_0.35",
"transportation-20250115-split19": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split19_0.30_0.35",
"transportation-20250115-split15": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split15_0.30_0.35",
"transportation-20250115-split5": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split5_0.30_0.35",
"transportation-20250115-split14": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split14_0.30_0.35",
"transportation-20250115-split7": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split7_0.30_0.35",
"transportation-20250115-split3": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split3_0.30_0.35",
"transportation-20250115-split6": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split6_0.30_0.35",
"transportation-20250115-split13": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split13_0.30_0.35",
"transportation-20250115-split8": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split8_0.30_0.35",
"transportation-20250115-split12": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split12_0.30_0.35",
"transportation-20250115-split10": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split10_0.30_0.35",
"transportation-20250115-split9": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split9_0.30_0.35",
"transportation-20250115-split4": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split4_0.30_0.35",
"transportation-20250115-split0": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split0_0.30_0.35",
"transportation-20250115-split2": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split2_0.30_0.35",
"transportation-20250115-split11": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split11_0.30_0.35",
"transportation-20250115-split1": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split1_0.30_0.35",
}


def get_new_data_slices_set(new_excel_path, old_excel_path=None):
    # new_excel_path = "/home/julian/LVM相關資訊-0421.xlsx"

    if old_excel_path is None:
        old_slices = set()
    else:
        old_df = pd.read_excel(old_excel_path, sheet_name="new dataset", engine="openpyxl")
        old_df = old_df[old_df["QA完成"].isin(["V", "v"])]
        old_slices = set(old_df["data slice 名稱"])

    # Read the Excel file into a DataFrame
    df = pd.read_excel(new_excel_path, sheet_name="new dataset", engine="openpyxl")

    filtered_df = df[df["QA完成"].isin(["V", "v"])]
    filtered_df = filtered_df[filtered_df["data slice 名稱"].isin(old_slices) == False]
    # new_slices_to_name = set(filtered_df["data slice 名稱"])
    new_slices_to_name = dict(zip(filtered_df["data slice 名稱"], filtered_df["dataset"]))
    return new_slices_to_name

def get_slice_name(anno_task_name):
    i = anno_task_name.split("split")[-1]
    if "upload0408-交通局" in anno_task_name:
        return anno_task_name.replace("upload0408-交通局", "upload0408-tr").lower(),f"Transportation_20250109_image_list_keep_0.95/split{i}_0.30_0.35"
    else:
        depart_en = get_depart(anno_task_name)
        depart_ch = get_depart(anno_task_name, chout=True)
        return anno_task_name.replace(depart_ch, depart_en).lower(), f"Transportation_20250115_image_list_keep_0.95_rededuplicate/split{i}_0.30_0.35"

    #Transportation_20250109_image_list_keep_0.95
    #Transportation_20250115_image_list_keep_0.95_rededuplicate


def find_excel_new_0428(excel_path):
    df = pd.read_excel(excel_path, sheet_name="new dataset", engine="openpyxl")
    import ipdb; ipdb.set_trace()

    filtered_df = df[df["QA完成"].isin(["V", "v"])]
    filtered_df = filtered_df[filtered_df["data slice 名稱"].isin(old_slices) == False]
    # new_slices_to_name = set(filtered_df["data slice 名稱"])
    new_slices_to_name = dict(zip(filtered_df["data slice 名稱"], filtered_df["dataset"]))
    import ipdb; ipdb.set_trace()
    return new_slices_to_name

anno_task_list_0423 = [
    # 4/22補 QA完成
    "交通局-20250115-split8",
    "交通局-20250115-split40",
    "交通局-20250115-split14",
    "upload0408-交通局-split12",
    "upload0408-交通局-split19",
    # 4/22補 QA中
    "交通局-20250115-split3",
    "交通局-20250115-split7",
    "交通局-20250115-split5",
    "交通局-20250115-split32",
    "upload0408-交通局-split13",
    "交通局-20250115-split33",
    "交通局-20250115-split30",
    "交通局-20250115-split34",
    "交通局-20250115-split24",
    # 4/22 已標註尚未QA
    "交通局-20250115-split29",
    "交通局-20250115-split27",
    "交通局-20250115-split20",
    "交通局-20250115-split25",
    "交通局-20250115-split21",
    "交通局-20250115-split22",
    "交通局-20250115-split23",
    "交通局-20250115-split16",
    "交通局-20250115-split26",
    "交通局-20250115-split18",
    "交通局-20250115-split17",
    "交通局-20250115-split19",
    "交通局-20250115-split15",
]
new_slices_to_name = {get_slice_name(x)[0]: get_slice_name(x)[1] for x in anno_task_list_0423}

SLICE_NAME_TO_DATASET_0526 = {
"upload0415-sd-20250226-split1": "upload0415_運發局_20250226_image_list_keep_0.95/split1_0.30_0.35",
"upload0415-sd-20250226-split2": "upload0415_運發局_20250226_image_list_keep_0.95/split2_0.30_0.35",
"upload0415-mrt-20250213-split6": "upload0415_捷運局_20250213_image_list_keep_0.95/split6_0.30_0.35",
"upload0415-pc-20250226-split3": "upload0415_港務局_20250226_image_list_keep_0.95/split3_0.30_0.35",
"upload0415-pc-20250226-split2": "upload0415_港務局_20250226_image_list_keep_0.95/split2_0.30_0.35",
"upload0415-tr-20250304-split5": "upload0415_交通局_20250304_image_list_keep_0.95/split5_0.30_0.35",
"upload0415-tr-20250304-split2": "upload0415_交通局_20250304_image_list_keep_0.95/split2_0.30_0.35",
"upload0415-wr-20250213-split2": "upload0415_水利局_20250213_image_list_keep_0.95/split2_0.30_0.35",
"upload0415-wr-20250213-split3": "upload0415_水利局_20250213_image_list_keep_0.95/split3_0.30_0.35",
"upload0415-wr-20250213-split6": "upload0415_水利局_20250213_image_list_keep_0.95/split6_0.30_0.35",
"upload0415-wr-20250324-split0": "upload0415_水利局_20250324_image_list_keep_0.95/split0_0.30_0.35",
"upload0415-wr-20250213-split1": "upload0415_水利局_20250213_image_list_keep_0.95/split1_0.30_0.35",
"upload0415-wr-20250213-split4": "upload0415_水利局_20250213_image_list_keep_0.95/split4_0.30_0.35",
"upload0415-sd-20250213-split0": "upload0415_運發局_20250213_image_list_keep_0.95/split0_0.30_0.35",
"upload0408-pw-20241230-split168": "upload0408_工務局_20241230_image_list_keep_0.95/split168_0.30_0.35",
"upload0408-pw-20241230-split76": "upload0408_工務局_20241230_image_list_keep_0.95/split76_0.30_0.35",
"upload0408-pw-20241230-split164": "upload0408_工務局_20241230_image_list_keep_0.95/split164_0.30_0.35",
"upload0408-pw-20241230-split38": "upload0408_工務局_20241230_image_list_keep_0.95/split38_0.30_0.35",
"upload0408-pw-20241230-split65": "upload0408_工務局_20241230_image_list_keep_0.95/split65_0.30_0.35",
"upload0408-pw-20241230-split85": "upload0408_工務局_20241230_image_list_keep_0.95/split85_0.30_0.35",
"upload0408-pw-20241230-split118": "upload0408_工務局_20241230_image_list_keep_0.95/split118_0.30_0.35",
"upload0408-pw-20241230-split49": "upload0408_工務局_20241230_image_list_keep_0.95/split49_0.30_0.35",
"upload0408-pw-20241230-split53": "upload0408_工務局_20241230_image_list_keep_0.95/split53_0.30_0.35",
"upload0408-pw-20241230-split59": "upload0408_工務局_20241230_image_list_keep_0.95/split59_0.30_0.35",
"upload0408-pw-20241230-split141": "upload0408_工務局_20241230_image_list_keep_0.95/split141_0.30_0.35",
"upload0408-pw-20241230-split0": "upload0408_工務局_20241230_image_list_keep_0.95/split0_0.30_0.35",
"upload0408-pw-20241230-split60": "upload0408_工務局_20241230_image_list_keep_0.95/split60_0.30_0.35",
"upload0408-pw-20241230-split70": "upload0408_工務局_20241230_image_list_keep_0.95/split70_0.30_0.35",
"upload0408-pw-20241230-split66": "upload0408_工務局_20241230_image_list_keep_0.95/split66_0.30_0.35",
"upload0408-pw-20241230-split173": "upload0408_工務局_20241230_image_list_keep_0.95/split173_0.30_0.35",
"upload0408-pw-20241230-split146": "upload0408_工務局_20241230_image_list_keep_0.95/split146_0.30_0.35",
"upload0408-pw-20241230-split176": "upload0408_工務局_20241230_image_list_keep_0.95/split176_0.30_0.35",
"upload0408-tr-20241230-split7": "upload0408_交通局_20241230_image_list_keep_0.95/split7_0.30_0.35",
"upload0408-tr-20241230-split14": "upload0408_交通局_20241230_image_list_keep_0.95/split14_0.30_0.35",
"upload0408-tr-20241230-split1": "upload0408_交通局_20241230_image_list_keep_0.95/split1_0.30_0.35",
"upload0408-tr-20241230-split5": "upload0408_交通局_20241230_image_list_keep_0.95/split5_0.30_0.35",
"upload0408-tr-20241230-split10": "upload0408_交通局_20241230_image_list_keep_0.95/split10_0.30_0.35",
"upload0408-tr-20241230-split11": "upload0408_交通局_20241230_image_list_keep_0.95/split11_0.30_0.35",
"upload0408-tr-20241230-split15": "upload0408_交通局_20241230_image_list_keep_0.95/split15_0.30_0.35",
"upload0408-tr-20241230-split18": "upload0408_交通局_20241230_image_list_keep_0.95/split18_0.30_0.35",
"upload0408-tr-20241230-split20": "upload0408_交通局_20241230_image_list_keep_0.95/split20_0.30_0.35",
"upload0408-tr-20241230-split6": "upload0408_交通局_20241230_image_list_keep_0.95/split6_0.30_0.35",
"upload0408-tr-20241230-split17": "upload0408_交通局_20241230_image_list_keep_0.95/split17_0.30_0.35",
"upload0408-tr-20241230-split13": "upload0408_交通局_20241230_image_list_keep_0.95/split13_0.30_0.35",
"upload0408-tr-20241230-split9": "upload0408_交通局_20241230_image_list_keep_0.95/split9_0.30_0.35",
"upload0408-tr-20241230-split22": "upload0408_交通局_20241230_image_list_keep_0.95/split22_0.30_0.35",
"upload0408-tr-20241230-split8": "upload0408_交通局_20241230_image_list_keep_0.95/split8_0.30_0.35",
"upload0408-tr-20241230-split12": "upload0408_交通局_20241230_image_list_keep_0.95/split12_0.30_0.35",
"upload0408-tr-20241230-split4": "upload0408_交通局_20241230_image_list_keep_0.95/split4_0.30_0.35",
"upload0408-tr-20241230-split3": "upload0408_交通局_20241230_image_list_keep_0.95/split3_0.30_0.35",
"upload0408-tr-20241230-split16": "upload0408_交通局_20241230_image_list_keep_0.95/split16_0.30_0.35",
"upload0408-tr-20241230-split21": "upload0408_交通局_20241230_image_list_keep_0.95/split21_0.30_0.35",
"upload0408-tr-20241230-split2": "upload0408_交通局_20241230_image_list_keep_0.95/split2_0.30_0.35",
"upload0408-tr-20241230-split19": "upload0408_交通局_20241230_image_list_keep_0.95/split19_0.30_0.35",
"upload0408-tr-20241230-split23": "upload0408_交通局_20241230_image_list_keep_0.95/split23_0.30_0.35",
"upload0408-pw-20241230-split14": "upload0408_工務局_20241230_image_list_keep_0.95/split14_0.30_0.35",
"upload0408-pw-20241230-split30": "upload0408_工務局_20241230_image_list_keep_0.95/split30_0.30_0.35",
"upload0408-pw-20241230-split18": "upload0408_工務局_20241230_image_list_keep_0.95/split18_0.30_0.35",
"upload0408-pw-20241230-split24": "upload0408_工務局_20241230_image_list_keep_0.95/split24_0.30_0.35",
"upload0408-pw-20241230-split34": "upload0408_工務局_20241230_image_list_keep_0.95/split34_0.30_0.35",
"upload0408-tr-20250109-split1": "upload0408_交通局_20250109_image_list_keep_0.95/split1_0.30_0.35",
"upload0408-tr-20250109-split5": "upload0408_交通局_20250109_image_list_keep_0.95/split5_0.30_0.35",
"upload0408-tr-20250109-split14": "upload0408_交通局_20250109_image_list_keep_0.95/split14_0.30_0.35",
"upload0408-tr-20250109-split3": "upload0408_交通局_20250109_image_list_keep_0.95/split3_0.30_0.35",
"upload0408-tr-20250109-split10": "upload0408_交通局_20250109_image_list_keep_0.95/split10_0.30_0.35",
"upload0408-tr-20250109-split17": "upload0408_交通局_20250109_image_list_keep_0.95/split17_0.30_0.35",
"upload0408-tr-20250109-split7": "upload0408_交通局_20250109_image_list_keep_0.95/split7_0.30_0.35",
"upload0408-tr-20250109-split6": "upload0408_交通局_20250109_image_list_keep_0.95/split6_0.30_0.35",
"upload0408-tr-20250109-split11": "upload0408_交通局_20250109_image_list_keep_0.95/split11_0.30_0.35",
"upload0408-tr-20250109-split18": "upload0408_交通局_20250109_image_list_keep_0.95/split18_0.30_0.35",
"upload0408-tr-20250109-split16": "upload0408_交通局_20250109_image_list_keep_0.95/split16_0.30_0.35",
"upload0408-tr-20250109-split15": "upload0408_交通局_20250109_image_list_keep_0.95/split15_0.30_0.35",
"upload0408-tr-20250109-split8": "upload0408_交通局_20250109_image_list_keep_0.95/split8_0.30_0.35",
"upload0408-tr-20250109-split4": "upload0408_交通局_20250109_image_list_keep_0.95/split4_0.30_0.35",
"upload0408-tr-20250109-split9": "upload0408_交通局_20250109_image_list_keep_0.95/split9_0.30_0.35",
"transportation-20250115-split38": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split38_0.30_0.35",
"transportation-20250115-split39": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split39_0.30_0.35",
"transportation-20250115-split36": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split36_0.30_0.35",
"transportation-20250115-split35": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split35_0.30_0.35",
"transportation-20250115-split37": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split37_0.30_0.35",
"transportation-20250115-split31": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split31_0.30_0.35",
"transportation-20250115-split28": "Transportation_20250115_image_list_keep_0.95_rededuplicate/split28_0.30_0.35",
}


def main(save_root, new_excel=None, old_excel=None):
    client = DataverseClient(
                host=DataverseHost.PRODUCTION.value ,
                email="julianlee@linkervision.com", 
                password=DATAVERSE_PASSWORD,
                service_id="2bd928e5-a98f-4aae-a093-8545c57c103f",
                # alias="prod",
                alias="default",
            )
    assert client is get_connection("default")
    # projects = client.list_projects(current_user = False,)
    lvm300k_slices = client.list_dataslices(project_id=DATAVERSE_BBOX_QA_PROJECT_ID, client_alias=client.alias)

    # new_slices_to_name = get_new_data_slices_set(new_excel_path=new_excel, old_excel_path=old_excel)
    excel_path = '/home/julian/LVM相關資訊(new dataset).csv'
    # new_slices_to_name = find_excel_new_0428(excel_path)

    # new_slices = [x for x in lvm300k_slices if x["name"] in new_slices_to_name]
    slice_name_to_dataset = SLICE_NAME_TO_DATASET_0526
    new_slices = [x for x in lvm300k_slices if x["name"] in slice_name_to_dataset]
    # import ipdb; ipdb.set_trace()
    for dataslice in tqdm.tqdm(new_slices, total=len(new_slices)):
        dataslice_id = dataslice['id']
        dataset_name = slice_name_to_dataset[dataslice["name"]]

        depart_en = get_depart(dataset_name)
        depart_ch = get_depart(dataset_name, chout=True)
        dataset_name = dataset_name.replace(depart_ch, depart_en)
        if "upload" in dataset_name:
            folder_name = ('_').join(dataset_name.split('_')[1:])
        else:
            folder_name = dataset_name
        # import ipdb; ipdb.set_trace()
        # dataset_name = new_slices_to_name[dataslice["name"]]
        # depart = get_depart(dataset_name)
        # folder_name = f"{depart}_{('_').join(dataset_name.split('_')[2:])}"
        # target_folder = f"{save_root}/{folder_name}"
        target_folder = f"{save_root}/{folder_name}"
        print(f"{dataslice['name']}\n {target_folder}")
        pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)
        os.chmod(target_folder, 0o777) 
        export_dataslice_to_local(
            host=DATAVERSE_CURATION_HOST,
            email=DATAVERSE_EMAIL,
            password=DATAVERSE_PASSWORD,
            service_id=DATAVERSE_SERVICE_ID_QA,
            dataslice_id=dataslice_id,
            target_folder=target_folder,
            sequential=False,
            annotation_name="groundtruth",
            export_format="coco",
        )


if __name__ == "__main__":
    fire.Fire(main)
