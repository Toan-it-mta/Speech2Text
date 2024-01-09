import os
import pickle
import json

def read_file_pkl(pkl_file_path):
    with open(pkl_file_path,'rb') as f:
        data = pickle.load(f)
    return data

for root, _, files in os.walk("./pkls_done"):
    for file in files:
        file_name = file.split('.')[0]
        path_file = os.path.join(root,file)
        data = read_file_pkl(path_file)
        for record in data:
            record['visitted'] = False
            record['gold_sentence'] = ''
            record['labeled'] = False
            record['humman'] = None
            del record['array']
        with open(os.path.join('./jsons',file_name+".json"),'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False)
        