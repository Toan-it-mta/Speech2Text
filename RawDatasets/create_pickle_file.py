import os
from tqdm import tqdm
import librosa
import pickle

def mappingAudioScript(audio_path, script_path):
    obj = {
            'path': audio_path,
            'array': [],
            'sentence': ""
        }
    try:
        y, _ = librosa.load(audio_path,sr=16000,mono=True)
        obj['array'] = y
        with open(script_path,encoding='utf-8') as f: 
            sentence = f.readline().strip()
        obj['sentence'] = sentence
        return obj
    except:
        return obj
    
def read_data(Audio_Folder, Script_Folder):   
    lst = [] 
    for root, _, files in os.walk(Audio_Folder):
        for name in tqdm(files):
            try:
                audio_path = os.path.join(root,name)
                args = root.split(os.sep)
                args[0] = Script_Folder
                Script_Folder_Path = os.sep.join(args)
                script_path = os.path.join(Script_Folder_Path,name.split('.')[0]+'.txt')
                obj = mappingAudioScript(audio_path,script_path)
                lst.append(obj)
            except Exception as e:
                pass
    with open('./Data47.pkl', 'wb') as f:
        pickle.dump(lst, f)

if __name__ == '__main__':
    Audio_Folder = 'output_segment_audio'
    Script_Folder = 'output_segment_srt'
    read_data(Audio_Folder,Script_Folder)





















# import os
# from tqdm import tqdm
# import librosa
# import pickle
# import shutil
# import csv

# def mappingAudioScript(audio_path, script_path):
#     obj = {
#             'path': audio_path,
#             'array': [],
#             'sentence': ""
#         }
#     try:
#         y, _ = librosa.load(audio_path,sr=16000,mono=True)
#         obj['array'] = y
#         with open(script_path,encoding='utf-8') as f: 
#             sentence = f.readline().strip()
#         obj['sentence'] = sentence
#         return obj
#     except:
#         return obj

# def getAudioScript(script_path):
#     try:
#         with open(script_path,encoding='utf-8') as f: 
#             sentence = f.readline().strip()
#             return sentence
#     except:
#         return ""
    
# def read_data(Audio_Folder, Script_Folder):
#     with open("./AIA_ASR/metadata.csv",'w',encoding='utf-8') as f_csv:
#         writer = csv.writer(f_csv)
#         writer.writerow(['file_path','script'])
#         for root, _, files in os.walk(Audio_Folder):
#             for name in tqdm(files):
#                 try:
#                     old_audio_path = os.path.join(root,name) # Audio path
#                     new_audio_path = os.path.join("./AIA_ASR/data",name)
#                     args = root.split(os.sep)
#                     args[0] = Script_Folder
#                     Script_Folder_Path = os.sep.join(args)
#                     script_path = os.path.join(Script_Folder_Path,name.split('.')[0]+'.txt')
#                     shutil.copyfile(old_audio_path,new_audio_path)
#                     script = getAudioScript(script_path)
#                     row = [new_audio_path,script]
#                     writer.writerow(row)    
#                 except Exception as e:
#                     pass


# if __name__ == '__main__':
#     Audio_Folder = 'output_segment_audio'
#     Script_Folder = 'output_segment_srt'
#     read_data(Audio_Folder,Script_Folder)
