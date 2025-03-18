from dataverse_sdk import *
from dataverse_sdk.connections import get_connection
client = DataverseClient(
    host=DataverseHost.PRODUCTION.value, email="julianlee@linkervision.com ", password="Baoyun5820", service_id="2bd928e5-a98f-4aae-a093-8545c57c103f", alias="default", force = True
)
assert client is get_connection("default")
 
# user = client.get_user()
 
client.list_projects()
 
projects = client.list_projects(current_user = False,)
                                # exclude_sensor_type=SensorType.LIDAR,
                                # image_type= OntologyImageType._2D_BOUNDING_BOX)
 
import ipdb; ipdb.set_trace()