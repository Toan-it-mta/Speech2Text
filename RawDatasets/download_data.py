# import gdown

# gdown.download_folder("https://drive.google.com/drive/u/1/folders/1jYXNVsIkqMKJ5Tf2Cx0u52Zm_XbyjnEO",remaining_ok=True)
from pymongo import MongoClient
import librosa
from tqdm import tqdm
import json

with open("./ASR_time.json",'r',encoding='utf-8') as f1:
    samples_time = json.load(f1)

with open("./ASR.json",'r',encoding='utf-8') as f2:
    samples = json.load(f2)
# random_samples = list(collection.find({},{"path": 1,"_id": 0}))
    #Update sample is visitted
assert len(samples_time) == len(samples)
for idx in tqdm(range(0,len(samples))):
    assert samples[idx]['path']==samples_time[idx]['path']
    samples[idx]['time'] = samples_time[idx]['time']

with open("./ASR_time_done.json",'w',encoding='utf-8') as f3:
    json.dump(samples,f3,ensure_ascii=False)
