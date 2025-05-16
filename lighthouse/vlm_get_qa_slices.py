import pathlib
import tqdm
import pandas as pd
import os 
import fire
from dataverse_sdk import *
from dataverse_sdk.connections import get_connection
from export_project_dataslices import export_dataslice_to_local
from common import DATAVERSE_PASSWORD, DATAVERSE_CKPT2_PROJECT_ID, DATAVERSE_CURATION_HOST, DATAVERSE_EMAIL, DATAVERSE_SERVICE_ID_QA, get_depart


VLM0505_SLICE_NAME = {
"tr-250416" : "upload0416_交通局_20250304_curated_t4",                                               
"wr-250416" : "upload0416_水利局_20250213_curated_t2",                                               
"wr-250324" : "upload0416_水利局_20250324_image_list_keep_0.95",                                     
"pw-250206" : "工務局_20250206-Public_Works_20250206_curated_t4",                                    
"mrt-250213": "捷運局_20250213-Mass_Rapid_Transit_20250213_curated_t7",
"sd-250213" : "運發局_20250213-Sports_Development_20250213_image_list_keep_0.95",                    
"sd-241223" : "運發局_20241223-Sports_Development_20241223_curated_t4-revised",                      
"pc-250124" : "港務局_20250124-Ports_Corporation_20250124_curated_t14",                              
"cs-250226" : "中鋼_20250226-China_Steel_20250226_image_list_keep_0.95",
"lk3"       : "LinkerV3_Linker_Vision_Data_V3_curated_t6_new-Linker_Vision_Data_V3_curated_t6_new",
}


# save_root = "/mnt/lighthouseACD/QAed-data/vlm/hand0505"
def main(save_root):
    client = DataverseClient(
                host=DataverseHost.PRODUCTION.value ,
                email="julianlee@linkervision.com", 
                password=DATAVERSE_PASSWORD,
                service_id="2bd928e5-a98f-4aae-a093-8545c57c103f",
                # alias="prod",
                alias="default",
            )
    assert client is get_connection("default")

    # exporting vlm dataslices
    # vlm_ckpt1_slices = client.list_dataslices(project_id=DATAVERSE_CKPT1_PROJECT_ID, client_alias=client.alias)
    vlm_slices = client.list_dataslices(project_id=DATAVERSE_CKPT2_PROJECT_ID, client_alias=client.alias)
    print(f"{len(vlm_slices)=}")
    vlm_slice_name = VLM0505_SLICE_NAME
    vlm_slices = [x for x in vlm_slices if x['name'] in vlm_slice_name]
    print(f"{len(vlm_slices)=}")
    host = "https://visionai.linkervision.ai/dataverse/curation"
    email = "julianlee@linkervision.com"
    password = DATAVERSE_PASSWORD
    service_id = "2bd928e5-a98f-4aae-a093-8545c57c103f"
    export_format = "vlm"
    anno = "groundtruth"
    for dataslice in vlm_slices:
        dataset_name = vlm_slice_name[dataslice['name']]
        dataset_name = dataset_name.replace("upload0416_", "")
        depart_en = get_depart(dataset_name)
        depart_ch = get_depart(dataset_name, chout=True)
        dataset_name = dataset_name.replace(depart_ch, depart_en)
        target_folder = f"{save_root}/{dataset_name}"
        dataslice_id = dataslice['id']
        # import ipdb; ipdb.set_trace()
        pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)
        os.chmod(target_folder, 0o777) 
        export_dataslice_to_local(
            host=host,
            email=email,
            password=password,
            service_id=service_id,
            dataslice_id=dataslice_id,
            target_folder=target_folder,
            sequential=False,
            annotation_name=anno,
            export_format=export_format,
        )
    exit(0)


if __name__ == "__main__":
    fire.Fire(main)
