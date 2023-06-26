import pickle

f = open(f'D:\onedrive\OneDrive - whu.edu.cn\Desktop\myModel\data\python\python_dedupe_definitions_v2.pkl', 'rb')
data = pickle.load(f)
for d in data:
    print(d)
    break