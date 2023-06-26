import os
from tqdm import tqdm
import jsonlines

#获取目标文件夹的路径
filedir ='../data/python/train/'
#获取当前文件夹中的文件名称列表
filenames = os.listdir(filedir)
filenames = tqdm(filenames)
#print(filenames)
#打开当前目录下的train.jsonl文件，如果没有则创建
with jsonlines.open(filedir+'train.jsonl', mode='a') as writer:
    for filename in filenames:
        filepath = filedir+'/'+filename
        #遍历单个文件，读取行数
        with open(filepath, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                writer.write(item)

#关闭文件
print("Merge Complete!")