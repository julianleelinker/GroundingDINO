import pathlib
import subprocess
import fire
from common import DEPART_MAP, VLM_ANNOTATION_ROOT, DATAVERSE_PASSWORD, DATAVERSE_CKPT1_PROJECT_ID, DATAVERSE_CKPT2_PROJECT_ID, VLM_ADDDED_0407_FOLDERS, DATAVERSE_SERVICE_ID_QA, DATAVERSE_SERVICE_ID_HAND
from common import get_depart_date


def main(folder, prefix):
    service_id = DATAVERSE_SERVICE_ID_HAND
    project_id = DATAVERSE_CKPT2_PROJECT_ID
    # folder_list = VLM_ADDDED_0407_FOLDERS
    folder_list = list(pathlib.Path(folder).glob("*"))

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
        import ipdb; ipdb.set_trace()

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