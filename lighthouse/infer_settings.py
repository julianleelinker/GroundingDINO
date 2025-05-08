import os 

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')

DINO_INFER_CLASSES = [
    'algae',
    'animal',
    'barricade',
    'boat',
    'bus',
    'car',
    'cone',
    'dog',
    'door',
    'drain',
    'driveway',
    'emergency exit',
    'entrance',
    'excavator',
    'faregate',
    'fence',
    'fire',
    'fish',
    'guardrail',
    'helmet',
    'human',
    'jersey barrier',
    'junk',
    'lane',
    'leaves',
    'litter',
    'manhole',
    'motorcycle',
    'parking lot',
    'passage',
    'palanquin',
    'pipeline',
    'road marking',
    'ruler',
    'sidewalk',
    'smoke',
    'solar panel',
    'storage tank',
    'streetlight',
    'traffic light',
    'traffic sign',
    'tree',
    'truck',
    'vest',
    'weapon',
    'seat',
]

BOX_THRESHOLD = 0.36
TEXT_THRESHOLD = 0.28
medium_threshold = 0.4
high_threshold = 0.5
HARD_CLASSES_TO_THRESHOLD = {
    'algae': medium_threshold,
    'faregate': medium_threshold,
    'fire': medium_threshold,
    'guardrail': medium_threshold,
    'solar panel': medium_threshold,
    'weapon': medium_threshold,

    "animal": high_threshold,
    "junk": high_threshold,
    "litter": high_threshold,
    "barricade": high_threshold,
    "parking lot": high_threshold,
    "human": high_threshold,
    "palanquin": high_threshold,
    "drain": high_threshold,
    "manhole": high_threshold,
    "lane": high_threshold,
    "storage tank": high_threshold,
    "excavator": high_threshold,
}

