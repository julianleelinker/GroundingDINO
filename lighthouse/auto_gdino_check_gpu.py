import subprocess
import time
import pathlib
import os
import tqdm
from common import DINO_COCO_SOURCE_ROOT, DINO_COCO_TARGET_ROOT_NEW, AUGMENTED_CURATED_RUNNING_JSONS, copy_images_in_json


def rsync_copy(remote_user, remote_host, remote_path, local_path):
    """
    Copies a file (or directory) from a remote server to a local path using rsync.
    """
    # Example rsync options:
    #   -a : archive mode
    #   -v : verbose
    #   -z : compress file data during the transfer
    # You can customize or remove these flags as needed.
    
    command = [
        "rsync",
        "-avz",  # or remove flags you don't need
        f"{remote_user}@{remote_host}:{remote_path}",
        local_path
    ]
    
    try:
        # Execute the rsync command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # If successful, you can read any output as:
        print("STDOUT:", result.stdout)
        print("File transfer completed successfully!")
        
    except subprocess.CalledProcessError as e:
        # Handle errors
        print("STDERR:", e.stderr)
        print(f"Rsync failed with return code {e.returncode}")


def get_gpu_memory(skip_gpu_list=[]):
    try:
        command = "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")  # Split output into lines
            result = {}
            for idx, line in enumerate(lines):
                total_memory, used_memory = map(int, line.split(", "))  # Convert to integers
                print(f"GPU {idx}: {used_memory}MB / {total_memory}MB")
                result[idx] = total_memory - used_memory
            result = {k: v for k, v in result.items() if k not in skip_gpu_list}
            return result
        else:
            print("Error running nvidia-smi:", result.stderr)
            return None
    except Exception as e:
        print("Error:", e)
        return None


def main():
    json_list = AUGMENTED_CURATED_RUNNING_JSONS
    json_list = [x for x in json_list if "China_Steel" in str(x)]

    dst_root = pathlib.Path(DINO_COCO_SOURCE_ROOT)

    conda_env = "lighthouse"
    # DATASET = 'Public_Works_20241230_image_list_keep_0.95'
    # dataset_name = 'Mass_Rapid_Transit_20250213_image_list_keep_0.95'

    source_root = DINO_COCO_SOURCE_ROOT
    mem_thres = 4000
    wait_time = 300
    skip_gpu_list = [0,]

    base_command = (
        "CUDA_VISIBLE_DEVICES={gpu_id} nohup conda run -n {conda_env} python lighthouse/inference_gpt_on_images_in_folder.py "
        "-c groundingdino/config/GroundingDINO_SwinT_OGC.py "
        "-p weights/groundingdino_swint_ogc.pth "
        "-o {target_root}/{dataset} "
        "--box_threshold 0.4 "
        "--text_threshold 0.3 "
        "--high_threshold 0.35 "
        "-t 'all' "
        "--enlarge_scale 1.4 "
        "--ios_threshold 0.3 "
        "-i {source_root}/{dataset}/split{i} "
        "> {dataset}-{i}.log 2>&1 &"
    )

    while True:
        dataset_list = []

        for json_path in json_list:
            dst = dst_root / json_path.stem
            dataset_list.extend(list(dst.glob('split*')))
        dataset_list = [x for x in dataset_list if not (x/"runned").exists()]
        dataset_list = sorted(dataset_list, key=lambda x: int(str(x).split('split')[-1]))

        result = get_gpu_memory(skip_gpu_list=skip_gpu_list)
        # running_number = len(result) - len(skip_gpu_list)
        # import ipdb; ipdb.set_trace()
        for (gpu_id, memory), data_path in zip(result.items(), dataset_list[:len(result)]):
            if gpu_id in skip_gpu_list:
                continue
            if memory > mem_thres:
                # dataset_list = pathlib.Path(f'{DATA_ROOT}/{dataset_name}').glob('split*')
                # dataset_list = sorted(dataset_list, key=lambda x: int(str(x).split('split')[-1]))
                # for data_path in dataset_list:
                #     if not (data_path / 'runned').exists():
                #         break
                split_index = int(str(data_path).split('split')[-1])
                command = base_command.format(i=split_index, conda_env=conda_env, source_root=source_root, dataset=data_path.parent.name, gpu_id=gpu_id, target_root=DINO_COCO_TARGET_ROOT_NEW)
                print(f"GPU {gpu_id} has enough memory, running inference on {data_path}")
                print(f"Executing: {command}")
                subprocess.run(command, shell=True, executable="/bin/bash")
                (data_path / 'runned').touch()
            else:
                print(f"GPU {gpu_id} doesn't have enough memory, check later")
        print(f'Waiting for {wait_time} seconds...\n')
        time.sleep(wait_time)


if __name__ == "__main__":
    main()