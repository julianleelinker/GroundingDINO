# Notes
1. "id" in vlm_annotations.json has one-to-one correspondance to "image" (hence image file), accumulate with all data images
2. each vlm_annotations.json has 1000 image's annotations.
3. the question is randomly seleted from below question list and ask gpt o1
# Question list
```
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
```
# Annotation directory structure
```
data_root
├── annotations
│   ├── vlm_annotations_1.json
│   ├── vlm_annotations_2.json
│   ├── ...
│   └── vlm_annotations_{m}.json
└── images
    ├── image_name_1.json
    ├── image_name_2.json
    ├── ...
    └── image_name_{n}.json
```

# vlm_annotations.json data format
```
[
    {
        "id": 1,
        "image": "image_name_1.jpg",
        "conversations": [
            {
                "question_id": 1,
                "question": "Offer a succinct explanation of the picture presented.",
                "answer": {
                    "groundtruth": "The picture shows a body of water covered with green algae. The algae form dense, irregular patches on the water's surface, creating a vibrant green pattern interspersed with areas of open water."
                }
            }
        ]
    },
    {
        "id": 2,
        "image": "image_name_2.jpg",
        "conversations": [
            {
                "question_id": 1,
                "question": "Write a terse but informative summary of the picture.",
                "answer": {
                    "groundtruth": "The picture shows a body of water with a significant presence of green algae covering its surface. The water appears blue and relatively calm, contrasting with the bright green patches of algae."
                }
            }
        ]
    }
]
```
