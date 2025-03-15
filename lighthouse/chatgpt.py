import os
import base64
import pathlib
import shutil
import json
import random
import tqdm
import argparse
import os
from collections import defaultdict
from openai import AzureOpenAI
from openai import BadRequestError, InternalServerError
from common import DEPARTS_EN


PROMPT_LIST = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo’s key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]


def make_parser():
    parser = argparse.ArgumentParser("ask chatgpt to describe image")
    parser.add_argument(
        "-j",
        "--json_path",
        required=True,
        type=str,
        help="the json file path of image list",
    )
    return parser.parse_args()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ask_chatgpt_describe_image(azure_openai_api_key, image_path, prompt="Please briefly describe the image.\n"):
    config = {
                "azure_endpoint": "https://azure-openai-vision-platform.openai.azure.com/",
                "api_key": azure_openai_api_key,
                "api_version":"2024-02-15-preview"
            }
    client = AzureOpenAI(**config)

    base64_image = encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                  "type": "text",
                  "text": prompt,
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  },
                },
            ],
        }
    ]
    try:
        completion = client.chat.completions.create(
                        model= "4o",
                        messages=messages,
                        temperature=0.5,
                        top_p=1,
                        max_tokens=500,
                        # stop='\n',
                    )
    except BadRequestError as e:
        print(e)
        return None
    except InternalServerError as e:
        print(e)
        return None
    return(completion.choices[0].message.content)


def ask_chatgpt_describe_image_find_suitable_answer(azure_openai_api_key, image_path, input_prompt_list, ansewer_length=50, try_limit=5):
    prompt_list = input_prompt_list.copy()
    random.shuffle(prompt_list)
    # print(f'{prompt_list=}')
    try_count, answer = 0, None
    while answer is None and try_count < try_limit:
        prompt = prompt_list.pop()
        try_count += 1
        response = ask_chatgpt_describe_image(azure_openai_api_key, image_path, prompt)
        if response is None:
            continue
        word_count = len(response.split())
        if word_count > ansewer_length:
            answer = response
    return answer, prompt


def generate_vlm_pretraining_annotation(id, image_name, prompt, response):
    return {
        "id": id,
        "image": image_name,
        "conversations": [
            {
                "question_id": 1,
                "question": prompt,
                "answer": {
                    "groundtruth": response,
                }
            }
        ]
    }
    

def create_path_name_mapping(image_path_list, mapping_file_path):
    image_name_path_dict = defaultdict(list)
    for image_path in image_path_list:
        image_name_path_dict[image_path.name].append(image_path)
    all_path_to_name = {}
    if len(image_name_path_dict) == len(image_path_list):
        all_path_to_name = {path_list[0]: name for name, path_list in image_name_path_dict.items()}
        return all_path_to_name

    renamed_path_name = {}
    for image_name, image_path_list in image_name_path_dict.items():
        if len(image_path_list) == 1:
            all_path_to_name[image_path_list[0]] = image_name
            continue
        for i in range(len(image_path_list)):
            new_image_name = f'{image_path_list[i].stem}-{i}{image_path_list[i].suffix}'
            renamed_path_name[image_path_list[i]] = new_image_name
    with open(mapping_file_path, 'w') as f:
        for path, name in renamed_path_name.items():
            f.write(f'"{name}","{path}"\n')
    all_path_to_name.update(renamed_path_name)
    return all_path_to_name


if __name__=='__main__':
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
    args = make_parser()
    output_root = args.json_path.replace('data-curation', 'vlm-annotations').replace('.json', '')
    output_root = pathlib.Path(output_root)
    print(f'{output_root=}')

    with open(args.json_path, 'r') as f:
        json_data = json.load(f)
    if 'image_path' in json_data[0]:
        image_path_list = [pathlib.Path(image['image_path']) for image in json_data]
    else:
        image_path_list = [pathlib.Path(image) for image in json_data]

    path_to_name = create_path_name_mapping(image_path_list, output_root / 'rename_name_path.txt')

    output_image_root = output_root / 'images'
    output_anno_root = output_root / 'annotations'
    output_image_root.mkdir(exist_ok=True, parents=True)
    os.chmod(output_image_root, 0o777)
    output_anno_root.mkdir(exist_ok=True, parents=True)
    os.chmod(output_anno_root, 0o777)

    anno_path = output_anno_root / 'vlm_annotations.json'
    prev_name_set, prev_annos_data = set(), []
    if anno_path.exists():
        with open(anno_path, 'r') as f:
            prev_annos_data = json.load(f)
    for anno in prev_annos_data:
        prev_name_set.add(anno['image'])

    annotation_list = prev_annos_data
    image_id = len(prev_annos_data) + 1
    max_count, min_count = float('-inf'), float('inf')
    max_id, min_id = -1, -1
    for image_path in tqdm.tqdm(image_path_list):
        image_path = image_path.resolve()
        new_image_name = path_to_name[image_path]

        if new_image_name in prev_name_set:
            print(f'{new_image_name} already exists')
            continue

        response, prompt = ask_chatgpt_describe_image_find_suitable_answer(AZURE_OPENAI_API_KEY, image_path, PROMPT_LIST, ansewer_length=50, try_limit=5)
        if not response:
            continue

        annotation_list.append(generate_vlm_pretraining_annotation(image_id, new_image_name, prompt, response))
        image_id += 1
        with open(anno_path, "w") as json_file:
            json.dump(annotation_list, json_file, indent=4, ensure_ascii=False)
        dst_file = output_image_root / new_image_name
        shutil.copy2(image_path, dst_file)

        # count response length
        word_count = len(response.split())
        if word_count > max_count:
            max_count = word_count
            max_id = image_id
        if word_count < min_count:
            min_count = word_count
            min_id = image_id

    (output_root / 'done').touch()
    print(f'{max_count=}, {min_count=}')
    print(f'{max_id=}, {min_id=}')
    print(f'{image_id=}, data save to {output_root}')
    print(f'number of annotations: {image_id-1}')