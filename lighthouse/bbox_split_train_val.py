import pathlib 
import random
import json
import fire


def main(scratch=False):
    data_root = "/mnt/lighthouseACD/image_text"
    # data_root = "/mnt/lighthouseACD/image_text/hand0428/Water_Resources_20250213_image_list_keep_0.95/split0_0.30_0.35_s2.5_mt0.26"
    train_json = f"{data_root}/train.json"
    val_json = f"{data_root}/val.json"

    all_tar_list = list(pathlib.Path(data_root).rglob("*.tar"))
    all_tar_list = [str(tar) for tar in all_tar_list]
    print(len(all_tar_list))

    with open(train_json, "r") as f:
        prev_train_list = json.load(f)
    if scratch:
        prev_train_list = []

    with open(val_json, "r") as f:
        prev_val_list = json.load(f)
    if scratch:
        prev_val_list = []

    prev_train_set = set(prev_train_list)
    prev_val_set = set(prev_val_list)
    print(f"{len(prev_train_list)=}")
    print(f"{len(prev_val_list)=}")
    import ipdb; ipdb.set_trace()

    unused_tar_list = list(set(all_tar_list) - prev_train_set - prev_val_set)
    print(len(unused_tar_list))

    # choose 10% of the tar files for validation
    random.shuffle(unused_tar_list)
    new_val_num = int(len(all_tar_list) * 0.15) - len(prev_val_list)
    print(f"{new_val_num=}")
    new_train_set = set(unused_tar_list[new_val_num:])
    new_val_set = set(unused_tar_list[:new_val_num])
    print(f"{len(new_train_set)=}")
    print(f"{len(new_val_set)=}")
    train_set = prev_train_set.union(new_train_set)
    val_set = prev_val_set.union(new_val_set)
    print(f"{len(train_set)=}")
    print(f"{len(val_set)=}")

    train_set = {str(pathlib.Path(x).relative_to(data_root)) for x in train_set}
    val_set = {str(pathlib.Path(x).relative_to(data_root)) for x in val_set}
    import ipdb; ipdb.set_trace()

    with open(train_json, "w") as f:
        json.dump(list(train_set), f)

    with open(val_json, "w") as f:
        json.dump(list(val_set), f)


if __name__ == "__main__":
    fire.Fire(main)