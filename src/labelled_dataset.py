import random
import pickle
import json
from tqdm import tqdm

choose = [x for x in range(0, 14)]
random.shuffle(choose)


def dump_to_json(output_data, outputfile):
    with open(outputfile, 'w') as fp:
        json.dump(output_data, fp, indent=4)


def read_data(dataset_split='train', all_count=100000, neg = 1):
    '''
    Read data from json file
    '''
    count = 0
    labble_dataset = []

    if dataset_split == 'train':
        data = json.load(open(f'../dataset/train/{dataset_split}.json'))
        # data = json.load(open(f'../dataset/python/{dataset_split}_data_{i}.json'))
        for elem in data:
            json_elem = {}
            # print(elem)
            json_elem['code'] = elem['code_tokens']
            json_elem['doc'] = elem['docstring_tokens']
            json_elem['ast'] = elem['ast_edges']
            json_elem['count'] = count
            labble_dataset.append(json_elem)
            count += 1
            if count == all_count - 1:
                break
    else:
        data = json.load(open(f'../dataset/test/{dataset_split}.json'))
        for elem in data:
            json_elem = {}
            json_elem['code'] = elem['code_tokens']
            json_elem['doc'] = elem['docstring_tokens']
            json_elem['ast'] = elem['ast_edges']
            json_elem['count'] = count
            labble_dataset.append(json_elem)
            count += 1
            if count == all_count - 1:
                break
    labelled_dataset=[]
    print(f"Reading {dataset_split} Set")
    for i in tqdm(range(len(labble_dataset))):
        code = labble_dataset[i]['code']
        ast = labble_dataset[i]['ast']
        doc = labble_dataset[i]['doc']
        count = labble_dataset[i]['count']
        #print(i)
        count2 = count
        if dataset_split == 'train':
            json_elem = {}
            while count==count2:
                ran = random.choice(labble_dataset)
                doc2, count2 = ran['doc'], ran['count']
            json_elem['code'] = code
            json_elem['doc'] = doc
            json_elem['ast'] = ast
            json_elem['doc2'] = doc2
            labelled_dataset.append(json_elem)
        else:
            distractor_list = {}
            count_list = []
            # while count==count2 or count2 in count_list:
            #     ran =  random.choice(labble_dataset)
            #     code2, ast2, count2, doc2= ran['code'], ran['ast'], ran['count'], ran['doc']
            #     #code2, count2, doc2 = ran['code'], ran['count'], ran['doc']
            #     if doc2 == doc:
            #         count2 = count
            distractor_list['code'] = code
            distractor_list['doc'] = doc
            distractor_list['ast'] = ast
            # distractor_list['code2'] = code2
            # distractor_list['ast2'] = ast2
            count_list.append(count2)
            labelled_dataset.append(distractor_list)
    random.shuffle(labelled_dataset)
    return labelled_dataset


def write_data(dataset, dataset_split='train'):
    dump_to_json(dataset, "../dataset/labelled_dataset_{}.json".format(dataset_split))


dataset = read_data('train', 475799)
print("read trainning dataset complete!")
write_data(dataset, 'train')
print("write trainning dataset complete!")
'''
dataset = read_data('valid', 5000)
print("read valid dataset complete!")
write_data(dataset, 'valid')
print("write valid dataset complete!")
'''
dataset = read_data('test', 10000)
print("read test dataset complete!")
write_data(dataset, 'test')
print("write test dataset complete!")
