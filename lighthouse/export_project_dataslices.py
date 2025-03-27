from dataverse_sdk import *
from dataverse_sdk.connections import get_connection
from export_dataslice_large import export_dataslice_to_local
from common import DATAVERSE_PASSWORD, DATAVERSE_CKPT1_PROJECT_ID, DATAVERSE_CKPT2_PROJECT_ID, DATAVERSE_LVM300K_PROJECT_ID


if __name__ == "__main__":
    client = DataverseClient(
                host=DataverseHost.PRODUCTION.value ,
                email="julianlee@linkervision.com", 
                password=DATAVERSE_PASSWORD,
                service_id="2bd928e5-a98f-4aae-a093-8545c57c103f",
                # alias="prod",
                alias="default",
            )
    assert client is get_connection("default")
    projects = client.list_projects(current_user = False,)
                                    # exclude_sensor_type=SensorType.LIDAR,
                                    # image_type= OntologyImageType._2D_BOUNDING_BOX)

    # exporting vlm dataslices
    vlm_ckpt1_slices = client.list_dataslices(project_id=DATAVERSE_CKPT1_PROJECT_ID, client_alias=client.alias)
    vlm_ckpt2_slices = client.list_dataslices(project_id=DATAVERSE_CKPT2_PROJECT_ID, client_alias=client.alias)
    host = "https://visionai.linkervision.ai/dataverse/curation"
    email = "julianlee@linkervision.com"
    password = DATAVERSE_PASSWORD
    service_id = "2bd928e5-a98f-4aae-a093-8545c57c103f"
    export_format = "vlm"
    anno = "groundtruth"
    target_root_to_ckpt = {
        "/mnt/data-home/mobility-multimodal/checkpoint/vlm/ckpt1": vlm_ckpt1_slices,
        "/mnt/data-home/mobility-multimodal/checkpoint/vlm/ckpt2": vlm_ckpt2_slices,
    }
    for target_root, vlm_slices in target_root_to_ckpt.items():
        for dataslice in vlm_slices:
            dataslice_id = dataslice['id']
            target_folder = f"{target_root}/{dataslice["name"]}"
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


    # exporting lvm300k dataslices
    lvm300k_slices = client.list_dataslices(project_id=DATAVERSE_LVM300K_PROJECT_ID, client_alias=client.alias)
    host = "https://visionai.linkervision.ai/dataverse/curation"
    email = "julianlee@linkervision.com"
    password = DATAVERSE_PASSWORD
    service_id = "2bd928e5-a98f-4aae-a093-8545c57c103f"
    export_format = "coco"
    anno = "groundtruth"
    target_root = "/mnt/data-home/mobility-multimodal/checkpoint/bbox"
    for dataslice in lvm300k_slices:
        dataslice_id = dataslice['id']
        target_folder = f"{target_root}/{dataslice["name"]}"
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