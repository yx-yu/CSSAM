import torch
import json
import os
import math


def create_dictionary(data, dtype='code', file_save=False):
    word_frequency_dict = {}
    word2id = {}
    id2word = {}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1

    if dtype == 'code':
        for i in data:
            for item in i:
                if item in word_frequency_dict:
                    word_frequency_dict[item] += 1
                else:
                    word_frequency_dict[item] = 1
        freq_dic = sorted(word_frequency_dict.items(), key=lambda x: x[1], reverse=True)
        for word in freq_dic:
            word2id[word[0]] = len(word2id)
        for key, value in word2id.items():
            id2word[value] = key
        if file_save is True:
            save_dic(word2id, id2word, dtype='code')
        return word2id, id2word
    if dtype == 'comment':
        word2id['<bos>'] = 2
        word2id['<eos>'] = 3
        for i in data:
            for item in i:
                if item in word_frequency_dict:
                    word_frequency_dict[item] += 1
                else:
                    word_frequency_dict[item] = 1
        freq_dic = sorted(word_frequency_dict.items(), key=lambda x: x[1], reverse=True)
        for word in freq_dic:
            if word[0] in word2id:
                continue
            else:
                word2id[word[0]] = len(word2id)
        for key, value in word2id.items():
            id2word[value] = key
        if file_save is True:
            save_dic(word2id, id2word, dtype='comment')
        return word2id, id2word


def save_dic(word_dic, id_dic, dtype='code'):
    if dtype == 'code':
        with open('./extra_file/code_word2id_dict.json', 'w') as f1:
            json.dump(word_dic, f1)
            f1.close()

        with open('./extra_file/code_id2word_dict.json', 'w') as f2:
            json.dump(id_dic, f2)
            f2.close()

    if dtype == 'comment':
        with open('./extra_file/comment_word2id_dict.json', 'w') as f1:
            json.dump(word_dic, f1)
            f1.close()

        with open('./extra_file/comment_id2word_dict.json', 'w') as f2:
            json.dump(id_dic, f2)
            f2.close()
    return


def load_dic(dype='code'):
    if dype == 'code':
        if os.path.exists('./extra_file/code_word2id_dict.json') and \
                os.path.exists('./extra_file/code_id2word_dict.json'):
            with open('./extra_file/code_id2word_dict.json', 'r') as f:
                word2id_dict = json.load(f)
                f.close()
            with open('./extra_file/code_id2word_dict.json', 'r') as f:
                id2word_dict = json.load(f)
                f.close()
        # else:
        #     word2id_dict,id2word_dict=create_dictionary(data,dtype=dype)
    if dype == 'comment':
        if os.path.exists('./extra_file/comment_word2id_dict.json') and \
                os.path.exists('./extra_file/comment_id2word_dict.json'):
            with open('./extra_file/comment_id2word_dict.json', 'r') as f:
                word2id_dict = json.load(f)
                f.close()
            with open('./extra_file/comment_id2word_dict.json', 'r') as f:
                id2word_dict = json.load(f)
                f.close()
        # else:
        #     word2id_dict,id2word_dict=create_dictionary(data,dtype=dype)

    return word2id_dict, id2word_dict
