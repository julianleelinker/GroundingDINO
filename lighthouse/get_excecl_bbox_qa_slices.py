import pathlib
import tqdm
import pandas as pd
import os 
import fire
from dataverse_sdk import *
from dataverse_sdk.connections import get_connection
from export_project_dataslices import export_dataslice_to_local
from common import DATAVERSE_PASSWORD, DATAVERSE_LVM300K_PROJECT_ID, DATAVERSE_CURATION_HOST, DATAVERSE_EMAIL, DATAVERSE_SERVICE_ID_QA, get_depart


def get_new_data_slices_set(new_excel_path, old_excel_path=None):
    new_excel_path = "/home/julian/LVM相關資訊-0421.xlsx"

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


def main(new_excel, save_root, old_excel=None):
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
    lvm300k_slices = client.list_dataslices(project_id=DATAVERSE_LVM300K_PROJECT_ID, client_alias=client.alias)

    # new_slices_to_name = get_new_data_slices_set(new_excel_path=new_excel, old_excel_path=old_excel)
    # new_slices = [x for x in lvm300k_slices if x["name"] in new_slices_to_name]
    new_slices = [x for x in lvm300k_slices if x["name"] in new_slices_to_name]
    # import ipdb; ipdb.set_trace()
    for dataslice in tqdm.tqdm(new_slices, total=len(new_slices)):
        dataslice_id = dataslice['id']
        dataset_name = new_slices_to_name[dataslice["name"]]
        # depart = get_depart(dataset_name)
        # folder_name = f"{depart}_{('_').join(dataset_name.split('_')[2:])}"
        # target_folder = f"{save_root}/{folder_name}"
        target_folder = f"{save_root}/{dataset_name}"
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