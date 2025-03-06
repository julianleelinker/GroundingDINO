import os
import base64
import pathlib
import shutil
import json
import random
import tqdm
import argparse
import os
from openai import AzureOpenAI
from openai import BadRequestError, InternalServerError


DEPART_LIST = [
    'China_Steel',
    'Mass_Rapid_Transit',
    'Ports_Corporation',
    'Public_Works',
    'Sports_Development',
    'Taiwan_Power',
    'Transportation',
    'Water_Resources',
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
    # print(str(image_path))
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

    output_image_root = output_root / 'images'
    output_anno_root = output_root / 'annotations'
    output_image_root.mkdir(exist_ok=True, parents=True)
    os.chmod(output_image_root, 0o777)
    output_anno_root.mkdir(exist_ok=True, parents=True)
    os.chmod(output_anno_root, 0o777)
    images_per_anno = 1000 # save a json for each target image 

    prev_annos_list = list(output_anno_root.rglob('*.json'))
    prev_image_set = set()
    for previous_annos in prev_annos_list:
        with open(previous_annos, 'r') as f:
            previous_annos_data = json.load(f)
        for anno in previous_annos_data:
            prev_image_set.add(anno['image'])
    # assert len(prev_image_set)==len(prev_annos_list)*1000, f'{len(prev_image_set)=}, {len(prev_annos_list)=}'    

    # temp
    prev_image_set = {
        'jpg-2024_10_03_10_11_50.jpg',
        '013054.jpg',
        '222855.jpg',
        '121956.jpg',
        ' jpg-2024_09_29_18_12_51.jpg',
        '640x480_2024_07_25_11-15.jpg',
        '640x480_2024_10_02_05-00.jpg',
        '131716.jpg',
    }
    # temp

    prompt_list = [
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
    annotation_list = []
    image_id = 1
    max_count, min_count = float('-inf'), float('inf')
    max_id, min_id = -1, -1
    n_anno = len(prev_annos_list) + 1
    anno_path = output_anno_root / f'vlm_annotations_{n_anno}.json'

    for image_path in tqdm.tqdm(image_path_list):
        src_file = image_path.resolve()
        for depart in DEPART_LIST:
            if depart in str(src_file):
                break
        folder_list = str(src_file).split(f'/{depart}/')
        dst_name = ('-').join(folder_list[1:]).replace('/', '-')
        dst_name = ('.').join(dst_name.split('.')[:-1]) + '.' + dst_name.split('.')[-1].lower()

        # temp, need to change file_name to dst_name
        file_name = src_file.stem + src_file.suffix.lower()
        if file_name in prev_image_set:
            print(f'{file_name} already exists')
            continue

        response, prompt = ask_chatgpt_describe_image_find_suitable_answer(AZURE_OPENAI_API_KEY, image_path, prompt_list, ansewer_length=50, try_limit=5)
        # print(f'{prompt=}')
        # print(f'{response=}')
        if response:
            # copy file in image_path_list
            dst_file = output_image_root / dst_name
            shutil.copy2(src_file, dst_file)
            annotation_list.append(generate_vlm_pretraining_annotation(image_id, dst_name, prompt, response))

            # count response length
            word_count = len(response.split())
            if word_count > max_count:
                max_count = word_count
                max_id = image_id
            if word_count < min_count:
                min_count = word_count
                min_id = image_id

            with open(anno_path, "w") as json_file:
                json.dump(annotation_list, json_file, indent=4, ensure_ascii=False)
            # print(f'data save to {anno_path}')
            image_id += 1
            if (image_id-1)//images_per_anno>0 and (image_id-1)%images_per_anno==0:
                annotation_list = []
                n_anno += 1
                anno_path = output_anno_root / f'vlm_annotations_{n_anno}.json'

    print(f'{max_count=}, {min_count=}')
    print(f'{max_id=}, {min_id=}')
    print(f'{image_id=}, data save to {output_root}')
    print(f'number of annotations: {image_id-1}')