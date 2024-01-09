from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import string 
import unicodedata
import re
import pickle
import os

def read_file_pkl(pkl_file_path):
    with open(pkl_file_path,'rb') as f:
        data = pickle.load(f)
    return data

def preprocess_text(text: str):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\xa0', '')
    text = re.sub(r"\s{2,}", ' ',text);
    return text

def dataframe_from_path(folders_path):
    paths = []
    arrays = []
    gold_sentences = []
    srs = []
    for root, _, files in os.walk(folders_path):
        for file in files:
            print("=== ",file," ===")
            file_path_pkl = os.path.join(root,file)
            data = read_file_pkl(file_path_pkl)
            for record in tqdm(data):
                if record['compare'] == True:
                    paths.append(record['path'])
                    arrays.append(record['array'])
                    srs.append(16000)
                    gold_sentences.append(preprocess_text(record['sentence']))
    dict = {"path": paths, "arr": arrays, "sr": srs, "gold_sentences": gold_sentences}
    df = pd.DataFrame(dict)
    return df

if __name__ == "__main__":
    dframe = dataframe_from_path('Datasets/pkls_done')
    train_df, test_df = train_test_split(dframe, test_size=0.4,random_state=19)
    print('========= Train processing =========')
    train_dataset = Dataset.from_dict(train_df)
    print('========= Test processing =========')
    test_dataset = Dataset.from_dict(test_df)
    dataset_dict = DatasetDict({'train':train_dataset, 'valid': test_dataset})
    dataset_dict.save_to_disk("Train/part2/vlsp_vivos_fpt_aia_27_12_2023")