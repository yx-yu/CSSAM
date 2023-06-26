from tree_sitter import Language, Parser
from sklearn import model_selection
import pandas
import pickle
import random
import numpy as np
import nltk
from keras.preprocessing import text, sequence
from fasttext import tokenize
from fasttext import load_model
from rpython import flowspace
from gatlayer import GAT
import torch

codes = ''
with open(f'D://onedrive//OneDrive - whu.edu.cn//Desktop//myModel//dataset//all.docstring','r', encoding='utf-8') as f:
    for line in f:
        codes += line.lower()
#print(codes)
trainDF={}
trainDF['doc'] = codes


# PY_LANGUAGE = Language('../build/my-languages.so', 'python')
# parser = Parser()
# parser.set_language(PY_LANGUAGE)
# code = '''
# d.decode('cp1251').encode('utf8')
# '''
# tree = parser.parse(bytes(code, "utf8"))
#print(tree.root_node.children[0])
def parse_py(node):
    if node == None:
        return
    print(node)
    for child in node.children:
        parse_py(child)

def parse_tree(node, stri):
    if node == None:
        return
    print(stri+str(node.type))
    stri += '    '
    for child in node.children:
        parse_tree(child,stri)

def create_embeddings(fname, embed_type):
    embeddings_index = load_model(fname)

    # create a tokenizer
    token = nltk.word_tokenize(trainDF[embed_type])
    token = nltk.FreqDist(token)
    sort = []
    temp = sorted(token.items(), key=lambda x: x[1], reverse=True)
    for key in temp:
        sort.append(key[0])
    word_index = {}
    for i, word in enumerate(sort):
        word_index[word] = i+1

    # create token-embedding mapping
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for i, word in enumerate(sort):
        embedding_vector = np.array(embeddings_index.get_word_vector(word),dtype='float32')
        if embedding_vector is not None:
            embedding_matrix[i+1] = embedding_vector

    return word_index, embedding_matrix
# parse_py(tree.root_node)
# c = '''
#     n = add(a,b)
# '''
# parse_tree(tree.root_node,"")
# # #print(tree.root_node)
# # #print(tree.root_node.children[0])
# word_index, embedding_matrix = create_embeddings(f'D://onedrive//OneDrive - whu.edu.cn//Desktop//myModel//trained_models//doc.bin', 'doc')
# print(word_index)
# print(embedding_matrix.shape)
# embeddings_index = load_model('../trained_models/code.bin')
# print(embeddings_index.get_word_vector("a"))
# print(embeddings_index.get_word_vector("a"))
# token = tokenize(trainDF['doc'])
# word_dict = {}
# for i in token:
#     if i not in word_dict.keys():
#         word_dict[i] = 1
#     else:
#         word_dict[i] += 1
# count = 0
# for i, j in word_dict.items():
#     if j > 4:
#         count +=1
# print(count)
# sort = []
# temp = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
# for key in temp:
#     sort.append(key[0])
# word_index = {}
# for i, word in enumerate(sort):
#     word_index[word] = i
# print(word_index)
import networkx as nx
def get_Graph_info(g,embeddings_index):
    G = nx.Graph()
    G.add_edges_from(g)
    #print(np.array(nx.adjacency_matrix(G).todense()))
    nodes = G.nodes
    print(list(nodes))
    adj = np.array(nx.adjacency_matrix(G).todense())
    nfeat = []
    for n in nodes:
        vec = embeddings_index.get(n)
        if vec is not None:
            nfeat.append(vec)
        else:
            nfeat.append([0]*128)
    return torch.Tensor(nfeat), adj
g = [['module', 'expression_statement'], ['expression_statement', 'call'], ['call', 'attribute'], ['call', 'argument_list'], ['attribute', 'a'], ['attribute', '.'], ['attribute', 'pop'], ['argument_list', '('], ['argument_list', 'index'], ['argument_list', ')']]
G = nx.Graph()
G.add_edges_from(g)
#print(G.nodes)
#print(np.array(nx.adjacency_matrix(G).todense()))
embeddings_index = {}
for i, line in enumerate(open('../word_vec/node.vec', 'r', encoding='utf-8')):
    if i == 0:
        continue
    values = line.split(" ")
    #print(" ".join(values[:-128]))
    embeddings_index[" ".join(values[:-128])] = np.asarray(values[-128:], dtype='float32')
nfeat,adj = get_Graph_info(g,embeddings_index)
#print(nfeat.shape)
#print(adj)
# gat = GAT(128,200,200,0.2,0.2,8)
# x = torch.randn(30,128)
# print(x,x.shape)
# adj = np.random.randint(0,2,(30,30))
# adj = torch.tensor(adj)
# print(adj,adj.shape)
# z = gat(x,adj)
# print(z,z.shape)
