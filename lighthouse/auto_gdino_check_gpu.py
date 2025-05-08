import subprocess
import time
import pathlib
import os
import tqdm
from common import DINO_COCO_SOURCE_ROOT, DINO_COCO_TARGET_ROOT_NEW, AUGMENTED_CURATED_JSONS_0508, copy_images_in_json


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
    json_list = AUGMENTED_CURATED_JSONS_0508

    dst_root = pathlib.Path(DINO_COCO_SOURCE_ROOT)

    conda_env = "lighthouse"

    source_root = DINO_COCO_SOURCE_ROOT
    mem_thres = 4000
    wait_time = 300
    # skip_gpu_list = [0,]
    # skip_gpu_list = [1,2,3]
    skip_gpu_list = []

    base_command = (
        "CUDA_VISIBLE_DEVICES={gpu_id} nohup conda run -n {conda_env} python lighthouse/inference_gpt_on_images_in_folder.py "
        "-c groundingdino/config/GroundingDINO_SwinT_OGC.py "
        "-p weights/groundingdino_swint_ogc.pth "
        "-o {target_root}/{dataset}/split{i} "
        "-i {source_root}/{dataset}/split{i} "
        "> logs/{dataset}-{i}.log 2>&1 &"
    )

    while True:
        dataset_list = []

        for json_path in json_list:
            dst = dst_root / json_path.stem
            dataset_list.extend(list(dst.glob('split*')))
        dataset_list = [x for x in dataset_list if not (x/"runned.txt").exists()]
        dataset_list = sorted(dataset_list, key=lambda x: int(str(x).split('split')[-1]))

        result = get_gpu_memory(skip_gpu_list=skip_gpu_list)
        for (gpu_id, memory), data_path in zip(result.items(), dataset_list[:len(result)]):
            if gpu_id in skip_gpu_list:
                continue
            if memory > mem_thres:
                split_index = int(str(data_path).split('split')[-1])
                command = base_command.format(i=split_index, conda_env=conda_env, source_root=source_root, dataset=data_path.parent.name, gpu_id=gpu_id, target_root=DINO_COCO_TARGET_ROOT_NEW)
                print(f"GPU {gpu_id} has enough memory, running inference on {data_path}")
                print(f"Executing: {command}")
                subprocess.run(command, shell=True, executable="/bin/bash")
                (data_path / 'runned.txt').touch()
            else:
                print(f"GPU {gpu_id} doesn't have enough memory, check later")
        print(f'Waiting for {wait_time} seconds...\n')
        time.sleep(wait_time)


if __name__ == "__main__":
    main()