import json
from tqdm import tqdm
import jsonlines
import nltk
import os
from preprocess import *
# from generate_graph_from_ast import *
import re
import pickle
import gzip
# from generate_graph_from_ast import *
import DFG


from graph_process import *
path = '../dataset/java/'
copath = '../data/'
dataset = 'parallel'
if dataset == 'codesearchnet':
    train_path = path + '/train'
    test_path = path + '/test/test.jsonl'
    valid_path = path + '/valid/valid.jsonl'
    train_data = '../dataset/python/train_data'
    valid_data = '../dataset/python/valid_data.json'
else:
    train_path = '../dataset/train.json'
    test_path = '../dataset/test.json'
    train_data = '../dataset/train/train.json'
    test_data = '../dataset/test/test.json'

json_data = [('train_path', 'train_data'), ('test_path','test_path'), ('valid_path', 'valid_path')]


def not_comment(seq):
    try:
        if seq[0] == '#':
            return False
        return True
    except:
        return True


# 将原始数据集中的有效数据提取出来，单独建立json文件，包括 code,code_tokens, docstring, docstring_tokens, ast
def csn_read_jsonlfile(filename):
    input_data = []
    import jsonlines
    with open(filename, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            input_data.append(item)
    output_data = []
    for line in tqdm(input_data):
        json_elem = {}
        # print(line)
        code = line['code']
        docstring = line['docstring']
        # print(docstring)
        code_tokens = code_tokenize(str_process(code))
        json_elem['code'] = code
        json_elem['docstring'] = docstring
        json_elem['code_tokens'] = code_tokens
        json_elem['docstring_tokens'] = code_tokenize(str_process(docstring))


        Gr = TCG(java_code)
        json_elem['ast_edges'] = list(Gr.edges)
        json_elem['ast_node'] = list(Gr.nodes)

        # print(json_elem)
        output_data.append(json_elem)
        # break
    # print(len(output_data))
    return output_data


def co_read_jsonfile(filename):
    output_data = []
    with open(filename, 'r') as reader:
        reader = json.load(reader)
        for line in tqdm(reader):
            json_elem = {}
            #print(line)
            code = line['snippet']
            if line['rewritten_intent'] != None:
                docstring = line['rewritten_intent']
            else:
                docstring = line['intent']
            #print(docstring)
            code_tokens = code_tokenize(str_process(code))
            json_elem['code'] = code
            json_elem['docstring'] = docstring
            json_elem['code_tokens'] = code_tokens
            json_elem['docstring_tokens'] = code_tokenize(str_process(docstring))


            Gr = TCG(java_code)
            json_elem['ast_edges'] = list(Gr.edges())
            json_elem['ast_node'] = list(Gr.nodes())

            #print(json_elem)
            output_data.append(json_elem)
            #break
    # print(len(output_data))
    return output_data
    # 创建纯文本的code docstring ast文件，便于后面训练词向量


def read_jsonfile(filename):
    output_data = []
    with open(filename, 'r', encoding='utf-8') as reader:
        lines = json.load(reader)
        for line in tqdm(lines):
            json_elem = {}
            #print(line)
            code = line['source_code']
            docstring = line['nl']
            if docstring == '':
                continue
            # print(docstring)
            code_tokens = code_tokenize(code)
            json_elem['code'] = code
            json_elem['docstring'] = docstring
            json_elem['code_tokens'] = code_tokens
            json_elem['docstring_tokens'] = code_tokenize(docstring)

            Gr = TCG(code)
            json_elem['ast_edges'] = list(Gr.edges())
            json_elem['ast_node'] = list(Gr.nodes())

            # print(json_elem)
            output_data.append(json_elem)
            # break
    print(len(output_data))
    return output_data
    # 创建纯文本的code docstring ast文件，便于后面训练词向量


def read_python_file(code, doc):
    output_data = []
    with open(code, 'r') as c, open(doc, 'r') as d:
        json_elem = {}

        code = c.readline()
        code = code[1:-2]
        print(code)
        docstring = d.readline()
        #print(code)
        #print(docstring)
        code_tokens = code_tokenize(code)
        json_elem['code'] = code
        json_elem['docstring'] = docstring
        json_elem['code_tokens'] = code_tokens
        json_elem['docstring_tokens'] = code_tokenize(docstring)


        Gr = TCG(java_code)
        json_elem['ast_edges'] = list(Gr.edges)
        json_elem['ast_node'] = list(Gr.nodes)

        #print(json_elem)
        output_data.append(json_elem)
            #break
    # print(len(output_data))
    return output_data
    # 创建纯文本的code docstring ast文件，便于后面训练词向量


def dump_to_json(output_data, outputfile):
    with open(outputfile, 'a+') as fp:
        json.dump(output_data, fp, indent=4)


def json_dump_text(output_json):
    count = 0

    with open("../dataset/all.code","a+", encoding='utf-8') as fp:
        for data in output_json:
            fp.write(' '.join(data["code_tokens"]))
            fp.write("\n")
    with open("../dataset/all.docstring","a+", encoding='utf-8') as fp:
        for data in output_json:
            fp.write(' '.join(data["docstring_tokens"]))
            fp.write("\n")
    with open("../dataset/all.ast","a+", encoding='utf-8') as fp:
        for data in output_json:
            for edge in data["ast_edges"]:
                fp.write(str(edge[0])+'<split>'+str(edge[1]))
                fp.write("\n")


if __name__ == '__main__':
    if dataset == 'codesearchnet':
        # with open(test_path, 'r') as load_f:
        #   load_dict = json.load(load_f)
        #   print(load_dict[0])
        # print(code_tokenize(load_dict[0]["snippet"]))
        # read_jsonlfile('../data/java/train/java_train_0.jsonl','../dataset/java/test.jsonl')
        train_file = os.listdir(train_path)
        count = 0
        for train_path in train_file:
            train_set = csn_read_jsonlfile('../data/python/train/' + train_path)
            #json_dump_text(train_set)
            dump_to_json(train_set, train_data + '_{}.json'.format(count))
            count += 1
        print("Completed Processing Train Set")
        train_set = []

        test_set = csn_read_jsonlfile(test_path)
        json_dump_text(test_set)
        dump_to_json(test_set, test_data)
        print("Completed Processing Test Set")

        valid_set = csn_read_jsonlfile(valid_path)
        json_dump_text(valid_set)
        dump_to_json(valid_set, valid_data)
        print("Completed Processing Train Set")
    else:
        test_set = read_jsonfile(test_path)
        json_dump_text(test_set)
        dump_to_json(test_set, test_data)
        print("Completed Processing Test Set")
        #
        train_set = read_jsonfile(train_path)
        json_dump_text(train_set)
        dump_to_json(train_set, train_data)
        print("Completed Processing Train Set")





    #print(word_tokenize("format number of spaces between strings `Python`, `:` and `Very Good` to be `20`"))
    #print(code_tokenize("driver.find_element_by_xpath(\"//p[@id, 'one']/following-sibling::p\")"))

