import os
import json
import numpy as np
from collections import defaultdict


normal = defaultdict(list)
hnsw = defaultdict(list)
for i in range(1, 6):
    with open(os.path.join('result', f'{i}.result')) as f:
        for line in f:
            tmp = json.loads(line.replace('\'', '"'))
            for k, v in tmp.items():
                normal[k].append(v) 

    with open(os.path.join('result', f'{i}_hnsw.result')) as f:
        for line in f:
            tmp = json.loads(line.replace('\'', '"'))
            for k, v in tmp.items():
                hnsw[k].append(v) 


print('normal')
for k in normal:
    print(k, np.mean(normal[k])) 

print('hnsw')
for k in hnsw:
    print(k, np.mean(hnsw[k])) 

