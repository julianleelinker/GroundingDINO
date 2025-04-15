# %%
import pathlib
import json
import tqdm

# %%
bad_image_name_list = [
    "高雄市事故影片_110年_9月_0915-0929自由同盟_0923_自由一路 同盟一路口_20210923090923_s60.jpg", #transportation
    "高雄市事故影片_110年_9月_0915-0929自由同盟_0916_自由一路 同盟一路口_20210916130013_s530.jpg",
    "高雄市事故影片_110年_7月_0710站西建國_站西路與建國三路口東南角_20210710155231_s410.jpg",
    "高雄市事故影片_112年_6月_0615三多凱旋_C000080 凱旋二路 三多二路_20230615080914_s550.jpg",
    "C999106_20241226_20241226062205_s100.jpg",
    "20250109_C999107_20241226_20241226191619_s0.jpg",
    "C999099_20241216_20241216193615_s40.jpg",
    "C999099_20241221_20241221083402_s230.jpg",
    "921演唱會進散場_NO.115_1F(E區02)_2024_9_21 下午 (UTC+08_00) 04_00_56_s16800.jpg", #sports
    "921演唱會進散場_NO.121_1F(E區14)_2024_9_21 下午 (UTC+08_00) 04_00_56_s4090.jpg",
    "921演唱會進散場_NO.117_1F(E區06)_2024_9_21 下午 (UTC+08_00) 04_00_56_s10830.jpg",
    "921演唱會進散場_NO.128_1F(G區10)_2024_9_21 下午 (UTC+08_00) 04_00_56_s18660.jpg",
    "921演唱會進散場_NO.132_M區(東門)_2024_9_21 下午 (UTC+08_00) 04_00_56_s20440.jpg",
    "921演唱會進散場_NO.126_1F(G區06)_2024_9_21 下午 (UTC+08_00) 04_00_56_s16060.jpg",
    "921演唱會進散場_NO.114_1F(E區01)_2024_9_21 下午 (UTC+08_00) 04_00_55_s18740.jpg",
    "921演唱會進散場_NO.113_1F(D區02)_2024_9_21 下午 (UTC+08_00) 04_00_55_s8500.jpg",
    "921演唱會進散場_NO.66_VIP 8( 外)_2024_9_21 下午 (UTC+08_00) 04_00_55_s16649.jpg",
    "20250109_3.事件影片及模擬事件影片(T0M)_輕軌輪椅上下車_T02 2024-06-24_camip7(這_sq01283_s310.jpg", #mrt
    "20250109_7.捷運車廂影像(T22)_T3_1131226_2024-12-26_06-00-00_9600-Cam1_s2330.jpg",
    "20250109_7.捷運車廂影像(T22)_T11_1131226_2024-12-26_05-30-00_7200-Cam3_s1600.jpg",
    "11308_1356fe18-7fc3-4cf5-8b75-827ad01819eb_20240818_200350.jpg", #water
    "154432.jpg",
    "20250106-凱米颱風-水位站-月世界抽水平台-20240724-175945.jpg",
    "20250106-凱米颱風-水位站-淵源橋-20240724-141410.jpg",
    "20250106-凱米颱風-水位站-鳳山圳滯洪池站-20240724-194625.jpg",
    "20250106-凱米颱風-水位站-玉庫二號橋-20240724-155147.jpg",
    "OA_2025_1_3_013736837_s350.jpg", #ports
    "OA_2025_1_2_080819703_s330.jpg",
    "OA_2025_1_3_060740292_s170.jpg",
    "CCTV影像1.1~1.15_北孔快車道N-870_北孔快車道N-870-20250108-160000_s6200.jpg",
    "OA_2024_12_28_042830780_s350.jpg",
    "CCTV影像1.1~1.15_中興交叉道C130_中興交叉道C130-球型-20250109-160000_s1340.jpg",
    "市7 電桿傾斜.jpg", #power
    "DJI_20240903102826_0126_D.JPG", #??
    "20250106_台電示範廠區｜電桿熔絲鏈開關_空拍機_DJI_20241119102354_0004_D_s50.jpg",
    "20250106_台電示範廠區｜電桿熔絲鏈開關_空拍機_DJI_20241119103622_0007_D_s35.jpg",
    "20250106_台電示範廠區｜電桿熔絲鏈開關_空拍機_DJI_20241119103622_0007_D_s43.jpg",
    "1726884140.64852166.jpg", #linker
    "1725100349.20400500.jpg",
    "1724050990.04516244.jpg",
    "1726219367.46140361.jpg",
    "1726219838.68560338.jpg",
    "1725080466.42883945.jpg",
    "1724755386.25589776.jpg",
]
# image_root = "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand"
image_root = "/mnt/data-home/mobility-multimodal/checkpoint/vlm/hand-revised-0415"
image_list = list(pathlib.Path(image_root).rglob("*.jpg"))
image_list.extend(list(pathlib.Path(image_root).rglob("*.JPG")))
image_name_set = {pathlib.Path(x).name for x in image_list}
image_name_to_path = {
    pathlib.Path(x).name: x for x in image_list
}

print(f"{len(image_list)}")
print(f"{len(image_name_set)=}")

# %%
count = 0
for x in bad_image_name_list:
    image_path = image_name_to_path[x]
    anno_path = image_path.parent.parent / "annotations" / "vlm_annotation.json"
    with open(anno_path, 'r') as f:
        anno_data = json.load(f)
    for i, anno in tqdm.tqdm(enumerate(anno_data), total=len(anno_data)):
        if anno["image"] == x:
            anno_data.pop(i) 
            print(f"popping {x}")
            count += 1
            break
    # save the new annotation file
    with open(anno_path, 'w') as f:
        json.dump(anno_data, f)
    # remove the image
    image_path.unlink()
print(count)
# %%
