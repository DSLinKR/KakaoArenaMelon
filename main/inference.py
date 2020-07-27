from cold_start_FastText import cold_start
import json
import pandas as pd

train = pd.read_json('train.json')
test = pd.read_json('test.json')

final_returnval = cold_start(train, test)
with open('results.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(final_returnval, ensure_ascii=False))