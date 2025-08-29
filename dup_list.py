import pathlib
from collections import defaultdict
import json
import copy

remove_list = [
    "電線斷裂_1552 仁福#89 架空地線斷線_183793.jpg", #3
    "電桿異常_1424 水寮#20-21_斷桿、斷線_S__18325524_0.jpg", #5
    "FC_竹坑#14-T1882GD49-FC跳脫_S__4751388.jpg", #7
    "電線斷裂_2105預校#16右2-2左2_高壓斷線_預校16右2-2左2 斷線_0.jpg", #9
    "FC_1552 仁福#88 FC跳脫_183797.jpg", #13
    "電線斷裂_南星#37-41(T1583BA01~)-#2斷線_S__4685853.jpg", #15
    "電線斷裂_003_0911_月光#77~78_架空G斷_003_0911_月光#77~78_架空G斷-2.jpg", #19
    "電線斷裂_1737 仁靶#10右1-2 高壓線斷落_77859.jpg", #21
    "電線斷裂_007_0943_鹿埔#38-1~鹿埔#39高壓斷線_007_0943_鹿埔#38-1~鹿埔#39高壓斷線-2.jpg", #23
    "1140604災情照片(電桿倒斷-傾斜-熔絲鏈開關脫落)_熔絲錄開關_20250610_1140604災情照片(電桿倒斷-傾斜-熔絲鏈開關脫落)_熔絲錄開關_仁愛#49右1.jpg", #25
    "電線斷裂_004_0939_鹿埔28~鹿埔30-1斷線_004_0939_鹿埔28~鹿埔30-1斷線-4.jpg", #27
    "FC_013_1755_口湖#29熔絲斷_013_1755_口湖#29熔絲斷-3.jpg", #29
    "電桿異常_036_1621_枋六#43右2~右4外物碰觸+FC跳脫+輕鋼歪斜_036_1621_枋六#43右2~右4外物碰觸+FC跳脫+輕鋼歪斜-2.jpg", #31
    "1140604災情照片(電桿倒斷-傾斜-熔絲鏈開關脫落)_熔絲錄開關_20250610_1140604災情照片(電桿倒斷-傾斜-熔絲鏈開關脫落)_熔絲錄開關_九江#97.jpg", #33
]
data_root = "/mnt/lighthouseACD/QAed-data/bbox/hand07xx-iou38/project-id-697"
image_list = list(pathlib.Path(f"{data_root}").glob("Taiwan_Power*/split*/images/*"))
print(len(image_list))
image_dict = {x.name: x for x in image_list}

split_path_group = defaultdict(set)
actuall_run = False
for x in remove_list:
    if x in image_dict:
        image_path = image_dict[x]
        if actuall_run:
            image_path.unlink()
        split_path_group[image_path.parent].add(pathlib.Path(image_path).name)
for k, v in split_path_group.items():
    print(f"{k}: {len(v)}")

# import ipdb; ipdb.set_trace()
for k, v in split_path_group.items():
    anno_root = k.parent / "annotations" / "labels.json"
    with open(anno_root, "r") as f:
        anno = json.load(f)
    new_anno = {"info": anno["info"], "licenses": anno["licenses"], "categories": anno["categories"], "images": [], "annotations": []}
    for image in anno["images"]:
        remove_id = set()
        if pathlib.Path(image["file_name"]).name in v:
            remove_id.add(image["id"])
            continue
        new_anno["images"].append(image)
    for annotation in anno["annotations"]:
        if annotation["image_id"] in remove_id:
            continue
        new_anno["annotations"].append(annotation)
    print(f"{len(anno['images'])=}")
    print(f"{len(new_anno['images'])=}")
    if actuall_run:
        with open(anno_root, "w") as f:
            json.dump(new_anno, f, indent=4)
    # import ipdb; ipdb.set_trace()