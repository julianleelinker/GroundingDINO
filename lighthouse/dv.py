from dataverse_sdk import *
from dataverse_sdk.connections import get_connection
from common import DATAVERSE_PASSWORD, DATAVERSE_CKPT1_PROJECT_ID, DATAVERSE_CKPT2_PROJECT_ID


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
ckpt1_data_slice = client.list_dataslices(project_id=DATAVERSE_CKPT1_PROJECT_ID, client_alias=client.alias)
ckpt2_data_slice = client.list_dataslices(project_id=DATAVERSE_CKPT2_PROJECT_ID, client_alias=client.alias)
import ipdb; ipdb.set_trace()