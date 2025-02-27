cat_list = [
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
    'seat',
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
]
cat_list = sorted(cat_list)
add_barcket = [f'[[{cat}]]' for cat in cat_list]
for cat in cat_list:
    print(f"    '{cat}',")
print(cat_list)
print((' ').join(add_barcket))

tmp = 'algae]] [[drain]] [[faregate]] [[fire]] [[guardrail]] [[palanquin]] [[solar panel]] [[storage tank]] [[weapon'

result = sorted(tmp.split(']] [['))
print(result)
for res in result:
    print(f'[[{res}]]')
for res in result:
    print(f"'{res}',")