import json
import pathlib
import tqdm


# load new anno.json , save to revised anno
#todo_anno_list = [
#    # only this two use this path to do vlm annotations
#    '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20250106/Public_Works_20250106_image_list_keep_0.95_rededuplicate.json',
#    '/mnt/data-home/mobility-multimodal/data-curation/Taiwan_Power/20250106/Taiwan_Power_20250106_image_list_keep_0.95_rededuplicate.json',
#    # 
#    # '/mnt/data-home/mobility-multimodal/data-curation/Public_Works/20241230/Public_Works_20241230_image_list_keep_0.95_rededuplicate.json',
#    # '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20241223/Sports_Development_20241223_image_list_keep_0.95_rededuplicate.json',
#    # '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95_rededuplicate.json',
#    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20241230/Transportation_20241230_image_list_keep_0.95_rededuplicate.json',
#    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250109/Transportation_20250109_image_list_keep_0.95_rededuplicate.json',
#    # '/mnt/data-home/mobility-multimodal/data-curation/Transportation/20250115/Transportation_20250115_image_list_keep_0.95_rededuplicate.json',
#]
#
## TODO deal with this tomorrow after gpt done load vlm_annoataions.json, save to revised anno
#vlm_anno_list = [
#    # '/mnt/data-home/mobility-multimodal/data-curation/Water_Resources/20250106/Water_Resources_20250106_image_list_keep_0.95_rededuplicate.json',
#
#    '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95_rededuplicate.json',
#]
#
#todo_anno_list = vlm_anno_list




# for image_path_file in todo_anno_list:
#     with open(image_path_file, 'r') as f:
#         image_path = json.load(f)
#     image_set = {pathlib.Path(image).stem for image in image_path}
#     dataverse_path = image_path_file.replace('data-curation', 'vlm-annotations').replace('image_list_keep_0.95_rededuplicate.json', 'image_list_keep_0.95')
#     dataverse_path = pathlib.Path(dataverse_path)
#     anno_path = dataverse_path / 'annotations'
#     anno_jsons = list(anno_path.rglob('new_anno_0306.json'))
#     annos = []
#     for anno_json in anno_jsons:
#         with open(anno_json, 'r') as f:
#             data = json.load(f)
#         annos.extend(data)

#     revised_annos = []
#     image_root = dataverse_path / 'images'
#     for anno in annos:
#         image_stem = anno['image'].split('.')[0]
#         if image_stem in image_set:
#             revised_annos.append(anno)
#             continue
#         print(anno['image'])
#         if (image_root / anno['image']).exists():
#             print(f'removing {anno["image"]}')
#             (image_root / anno['image']).unlink()

#     revised_anno_path = dataverse_path / 'annotations' / f'revised_anno.json'
#     with open(revised_anno_path, 'w') as f:
#         json.dump(revised_annos, f, ensure_ascii=False, indent=4)


 
def remove_duplicated(dataverse_path, rededuplicated_json, old_anno_str='*new_anno_0306.json', new_anno_name='revised_anno.json', dry_run=True, verbose=True):
    anno_path = dataverse_path / 'annotations'
    anno_jsons = list(anno_path.rglob(old_anno_str))
    anno_data = []
    for anno_json in anno_jsons:
        with open(anno_json, 'r') as f:
            data = json.load(f)
        anno_data.extend(data)

    revised_annos = []
    image_root = dataverse_path / 'images'

    with open(rededuplicated_json, 'r') as f:
        image_path = json.load(f)
    image_set = {pathlib.Path(image).stem for image in image_path}
    print(len(image_set))

    removed_count = 0
    for anno in tqdm.tqdm(anno_data):
        image_stem = ('.').join(anno['image'].split('.')[:-1])
        if image_stem in image_set:
            revised_annos.append(anno)
            continue

        removed_count += 1
        if verbose:
            print(f'found duplicated {anno["image"]}')
        if (image_root / anno['image']).exists() and not dry_run:
            (image_root / anno['image']).unlink()

    if dry_run:
        print(f'dry run, removed {removed_count} images from {dataverse_path}')
    else:
        revised_anno_path = dataverse_path / 'annotations' / new_anno_name
        with open(revised_anno_path, 'w') as f:
            json.dump(revised_annos, f, ensure_ascii=False, indent=4)
        print(f'removed {removed_count} images from {dataverse_path}')


if __name__ == '__main__':
    # dataverse_path = '/mnt/data-home/mobility-multimodal/vlm-annotations/Sports_Development/20250109/Sports_Development_20250109_llava-onevision-0.5b-full/'
    # rededuplicated_json = '/mnt/data-home/mobility-multimodal/data-curation/Sports_Development/20250109/Sports_Development_20250109_image_list_keep_0.95_rededuplicate.json'
    rededuplicated_json = '/mnt/data-home/mobility-multimodal/data-curation/Linker_Vision_Data_V2/linker_4M_image_list_keep_0.95_full.json'
    dataverse_path_prefix = '/mnt/data-home/mobility-multimodal/vlm-annotations/Kaohsiung-full-dataset/20241231/Kaoshsiung_76152_retrieval_curated_t220_part'

    dataverse_path_list = []
    for i in range(1, 6):
        for j in range(1, 5):
            dataverse_path_list.append(f'{dataverse_path_prefix}{i}_{j}/')
    i = 6
    for j in range(1, 4):
        dataverse_path_list.append(f'{dataverse_path_prefix}{i}_{j}/')

    for dataverse_path in dataverse_path_list:
        remove_duplicated(pathlib.Path(dataverse_path), rededuplicated_json, verbose=False, dry_run=False)