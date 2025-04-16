import pathlib
import subprocess
import fire
from common import DEPART_MAP, VLM_CKPT1_FOLDERS, VLM_CKPT2_FOLDERS, VLM_ANNOTATION_ROOT, DATAVERSE_PASSWORD, DATAVERSE_CKPT1_PROJECT_ID, DATAVERSE_CKPT2_PROJECT_ID, VLM_ADDDED_0407_FOLDERS
from common import get_depart_date


PROJECT_ID_CKPT1 = 230 
PROJECT_ID_CKPT2 = 464 
WORKING_SERVICE_ID =  "2bd928e5-a98f-4aae-a093-8545c57c103f"
CKPT_HAND_PROJECT_ID = 122
CKPT_HAND_SERVICE_ID = "697aa90b-00d0-4455-8863-bd2ad70a93e7"


def main(prefix):
    folder_list = VLM_CKPT2_FOLDERS
    service_id = WORKING_SERVICE_ID
    project_id = PROJECT_ID_CKPT2
    folder_list = VLM_ADDDED_0407_FOLDERS

    # check folder exit
    for folder in folder_list:
        assert pathlib.Path(folder).exists(), f"{folder} not exist"

    count = 0
    data_idx = 0
    print(len(folder_list))
    for folder in folder_list[data_idx:]:
        depart, subfolder = get_depart_date(folder, ch=True)
        # dataset_name = f"{DEPART_MAP[depart]}_{date}-{subfolder}"
        dataset_name = f"{prefix}_{depart}_{subfolder}"

        print(f"uploading data idx {data_idx}")
        print(folder)
        print(f'{dataset_name=}\n')
        # import ipdb; ipdb.set_trace()

        # command = [
        #     "conda", "run", "-n", "dataverse-sdk", "python", "tools/import_vqa.py",
        #     "-host", "https://visionai.linkervision.ai/dataverse/curation",
        #     "-e", "julianlee@linkervision.com",
        #     "-p", DATAVERSE_PASSWORD,
        #     "-s", service_id,
        #     "-project", str(project_id),
        #     "--folder", folder,
        #     "-name", dataset_name
        # ]
        command = [
            f"conda run -n dataverse-sdk python tools/import_vqa.py -host https://visionai.linkervision.ai/dataverse/curation -e julianlee@linkervision.com -p {DATAVERSE_PASSWORD} -s {service_id} -project {project_id} --folder {folder} -name {dataset_name}"
        ]
        print(command)
        print('\n')
        # subprocess.run(command, check=True)
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable="/bin/bash")
            
        (folder / 'uploaded').touch()
        count += 1
        data_idx += 1
    print(f'total {count} data upload')


if __name__ == "__main__":
    fire.Fire(main)