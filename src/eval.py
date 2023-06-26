from sklearn import model_selection
import pandas
import pickle
import random
import re
import numpy as np
import time
import json
import yaml
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.utils.data
from keras.preprocessing import text, sequence
from model import *
import nltk
from fasttext import load_model
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
#torch.distributed.init_process_group(backend="nccl")
from preprocess import *
import math
import torch.nn.functional as F

def load_DF(fname):
    codes = ''
    with open(fname, 'r',encoding='utf-8') as f:
        for line in f:
            codes += line
    # print(codes)
    return codes

trainDF={}
trainDF['doc'] = load_DF(f'../dataset/all.docstring')
trainDF['code'] = load_DF(f'../dataset/all.code')
#trainDF['ast'] = load_DF(f'../dataset/all.ast')

# Loading word embeddings
def prepare_sequence(seq, seq_len, to_ix):
    idxs_list = []
    for seq_elem in seq:
        idxs = []
        for w in seq_elem.split():
            geti = to_ix.get(w, 0)
            if geti == 0:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(w)
                print(seq_elem)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            idxs.append(geti)
        if len(idxs) > seq_len:
            idxs = idxs[:seq_len]
        while len(idxs) < seq_len:
            idxs.append(0)
        idxs.reverse()
        idxs_list.append(idxs)
    return torch.tensor(idxs_list, dtype=torch.long)

def read_node_vec(fname):
    node_embedding_index = {}
    for i, line in enumerate(open('../word_vec/node.vec', 'r', encoding='utf-8')):
        if i == 0:
            continue
        values = line.split(" ")
        #print(" ".join(values[:-128]))
        node_embedding_index[" ".join(values[:-128])] = np.asarray(values[-128:], dtype='float32')
    return node_embedding_index

def get_Graph_info(gs,embeddings_index, node_len, adj_len):
    nodesl , adjsl = [], []
    #nodesl = torch.Tensor(nodesl)
    for g in gs:
        g = eval(g)
        G = nx.Graph()
        G.add_weighted_edges_from(g)
        nodes = G.nodes
        # print(np.array(nx.adjacency_matrix(G).todense()))
        if len(G.edges) != 0:
            adj = np.array(nx.adjacency_matrix(G).todense())
        l = adj_len-adj.shape[0]
        if(l > 0):
            adj = np.pad(adj,((0,l),(0,l)))
        else:
            adj = adj[:adj_len,:adj_len]
        nfeat = []
        for n in nodes:
            vec = np.random.randn(128)
            #vec = embeddings_index.get(n)
            if vec is not None:
                nfeat.append(vec)
            else:
                nfeat.append([0] * 128)
        while len(nfeat) < node_len:
            nfeat.append([0] * 128)
        nodesl.append(nfeat[:100])
        adjsl.append(adj[:])
    return torch.Tensor(nodesl), torch.Tensor(adjsl)
    
def create_embeddings(fname, embed_type):
    embeddings_index = load_model(fname)

    # create a tokenizer
    token = code_tokenize(str_process(trainDF[embed_type]))
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
    #print(word_index)
    #print(embedding_matrix)
    return word_index, embedding_matrix

with open("../config.yml", 'r') as config_file:
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)

##############################################################
#                     load dataset                           #
##############################################################
def read_json(dataset,dataset_type):
    ret = []
    if dataset_type == 'train':
        for elem in dataset:
            code = ' '.join(elem['code'])
            doc = ' '.join(elem['doc'])
            ast = ''.join(str(elem['ast']))
            doc2 = ' '.join(elem['doc2'])
            ret.append((code, ast, doc,doc2))
            #ret.append((code, doc, doc2))
    elif dataset_type == 'test':
        for elem in dataset:
            code = ' '.join(elem['code'])
            doc = ' '.join(elem['doc'])
            ast = ''.join(str(elem['ast']))
            ret.append((code, ast, doc))
            #ret.append((code, doc, code2))
    return ret

if cfg["dataset"] == 'conala':
    #train_dataset = json.load(open('../dataset/labelled_dataset_train.json', 'rb'))
    test_dataset = json.load(open('../dataset/labelled_dataset_test.json', 'rb'))

    #train_dataset = read_json(train_dataset,'train')
    test_dataset = read_json(test_dataset,'test')
    # codes_train, asts_train, docs_train, docs2_train = zip(*train_dataset)
    codes_test, asts_test, docs_test = zip(*test_dataset)
    # codes_train, docs_train, docs2_train = zip(*train_dataset)
    #codes_test,  docs_test, codes2_test = zip(*test_dataset)

    if cfg["model"] == 'base':
        # train_dataset = list(zip(codes_train, docs_train, docs2_train))
        test_dataset = list(zip(codes_test, docs_test))
    elif cfg["model"] == 'REModel':
        # train_dataset = list(zip(codes_train, docs_train, docs2_train))
        test_dataset = list(zip(codes_test, docs_test))
    elif cfg["model"] == 'graph':
        # train_dataset = list(zip(codes_train, asts_train, docs_train, docs2_train))
        test_dataset = list(zip(codes_test, asts_test, docs_test))

##############################################################
#                     load dataset                           #
##############################################################
random_seed = cfg["random_seed"]
np.random.seed(random_seed)
embedding_dim = cfg["embedding_dim"]
learning_rate = cfg["learning_rate"]
seq_len_doc = 0
seq_len_code = 0
hidden_size = cfg["hidden_size"]
dense_dim = cfg["dense_dim"]
output_dim = cfg["output_dim"]
num_layers_lstm = cfg["num_layers_lstm"]
use_cuda = cfg["use_cuda"]
batch_size = cfg["batch_size"]
model_type = cfg["model"]
# n_iters = 4000
# num_epochs = n_iters / (len(train_dataset) / batch_size)
# num_epochs = int(num_epochs)
num_epochs = cfg["epochs"]
use_softmax_classifier = cfg["use_softmax_classifier"]
use_bin = cfg["use_bin"]
use_bidirectional = cfg["use_bidirectional"]
use_adam = cfg["use_adam"]
use_parallel = cfg["use_parallel"]
save_path = cfg["save_path"]
if use_cuda:
    device_id = cfg["device_id"]
    torch.cuda.set_device(device_id)



print(f"Number of epochs = {num_epochs}")
print(f"Batch size = {batch_size}")
print(f"Dataset = {cfg['dataset']}")
print(f"Model = {model_type.upper()}")

##############################################################
#                     load dataset                           #
##############################################################

# Create word-index mapping
word_to_ix_doc = {}
word_to_ix_code = {}
seq_len_code = seq_len_doc = seq_len_ast = 300
load_var = False


if cfg["dataset"] == 'conala':
    word_to_ix_doc, weights_matrix_doc = create_embeddings(f'../trained_models/doc.bin', 'doc')
    word_to_ix_code, weights_matrix_code = create_embeddings(f'../trained_models/code.bin', 'code')
    #word_to_ix_ast, weights_matrix_ast = create_ast_embeddings(f'../word_vec/ast.vec', 'ast')
    weights_matrix_doc = torch.from_numpy(weights_matrix_doc)
    weights_matrix_code = torch.from_numpy(weights_matrix_code)
    #weights_matrix_ast = torch.from_numpy(weights_matrix_ast)
elif cfg["dataset"] == 'codesearchnet':
    if not load_var:
        word_to_ix_doc, weights_matrix_doc = create_embeddings(f'../{save_path}/doc_model.bin', 'doc')
        word_to_ix_code, weights_matrix_code = create_embeddings(f'../{save_path}/code_model.bin', 'code')
        word_to_ix_ast, weights_matrix_ast = create_embeddings(f'../{save_path}/ast_model.bin', 'ast')
        weights_matrix_doc = torch.from_numpy(weights_matrix_doc)
        weights_matrix_code = torch.from_numpy(weights_matrix_code)
        weights_matrix_ast = torch.from_numpy(weights_matrix_ast)
    else:
        word_to_ix_doc, weights_matrix_doc = pickle.load(open("../variables/doc_var",'rb'))
        word_to_ix_code, weights_matrix_code = pickle.load(open("../variables/code_var",'rb'))
        word_to_ix_ast, weights_matrix_ast = pickle.load(open("../variables/ast_var",'rb'))
#node_embed_matrix = read_node_vec("")
if cfg['model'] == 'base':
    print("------------BASE MODEL-------------")
    sim_model = BaseModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code)
elif cfg['model'] == 'REModel':
    print("------------CRESS MODEL-------------")
    sim_model = REModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code)
elif cfg['model'] == 'graph':
    print("------------GRAPH MODEL-------------")
    sim_model = GraphModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code)

if use_parallel:
    sim_model = nn.DataParallel(sim_model)
    sim_model = sim_model.module
sim_model.load_state_dict(torch.load(f"../{save_path}/sim_model_maxpool_re_final_200"))


if torch.cuda.is_available() and use_cuda:
    sim_model.cuda()
    
ret_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)
ret1_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=500,
                                          shuffle=False)
def eval_base():
    print("-----------------------EVAL BASE----------------------")
    mrr = 0
    count = 0
    r1 = 0
    r5 = 0
    r10 = 0
    ndcg = 0

    rank_list = []
    sim_model.eval()

    with torch.no_grad():
        codebase = []
        docbase = []
        for i, (code_seq, doc_seq) in enumerate(tqdm(ret_loader)):
            doc_in_ori = prepare_sequence(doc_seq, seq_len_doc, word_to_ix_doc)
            ranked_list = []
            for j, (code_seqs, doc_seq) in enumerate(ret1_loader):
                code_in = prepare_sequence(code_seqs, seq_len_code, word_to_ix_code)
                doc_in = doc_in_ori.expand(code_in.shape[0], 300)
                if torch.cuda.is_available() and use_cuda:
                    doc_in = doc_in.cuda()
                if torch.cuda.is_available() and use_cuda:
                    code_in = code_in.cuda()
                sim_score = sim_model(doc_in, code_in)
                sim_score = sim_score.tolist()
                ranked_list += list(zip(sim_score, code_seqs))
            ranked_list = sorted(ranked_list, reverse=True)
            doc_seq = doc_seq[0]
            code_seq = code_seq[0]
            rank = 0
            for k, (score, cand) in enumerate(ranked_list):
                if cand == code_seq and rank == 0:
                    rank = k + 1
                    break
            if not rank:
                count += 1
                continue
            rank_list.append([doc_seq, code_seq, rank])
            mrr += 1.0 / (rank)
            if rank < 50:
                ndcg += 1.0 / math.log2(rank + 1)
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            if rank <= 10:
                r10 += 1
            count += 1
            
    print(count)
    mrr /= count
    r1 /= count
    r5 /= count
    r10 /= count
    ndcg /= count
    print("MRR = ", mrr)
    print("Recall@1 = ", r1)
    print("Recall@5 = ", r5)
    print("Recall@10 = ", r10)
    print("NDCG@50 = ", ndcg)
    df = pandas.DataFrame(rank_list, columns=['Query', 'Gold', 'Rank'])
    df.to_pickle(f"../results/results_base_{cfg['dataset']}.pkl")
    with open(f"../results/results_base_{cfg['dataset']}.txt", "w") as f:
        f.write(f"MRR = {mrr}\n")
        f.write(f"Recall@1 = {r1}\n")
        f.write(f"Recall@5 = {r5}\n")
        f.write(f"Recall@10 = {r10}\n")
        f.write(f"NDCG@50 = {ndcg}\n")


def eval_re():
    print("-----------------------EVAL CRESS----------------------")
    mrr = 0
    count = 0
    r1 = 0
    r5 = 0
    r10 = 0
    ndcg = 0

    rank_list = []
    sim_model.eval()

    with torch.no_grad():
        codebase = []
        docbase = []
        for i, (code_seq, doc_seq) in enumerate(tqdm(ret_loader)):
            doc_in_ori = prepare_sequence(doc_seq, seq_len_doc, word_to_ix_doc)
            ranked_list = []
            for j, (code_seqs, doc_seq) in enumerate(ret1_loader):
                code_in = prepare_sequence(code_seqs, seq_len_code, word_to_ix_code)
                doc_in = doc_in_ori.expand(code_in.shape[0], 300)
                if torch.cuda.is_available() and use_cuda:
                    doc_in = doc_in.cuda()
                if torch.cuda.is_available() and use_cuda:
                    code_in = code_in.cuda()
                sim_score = sim_model(doc_in, code_in)
                sim_score = sim_score.tolist()
                ranked_list += list(zip(sim_score, code_seqs))
            ranked_list = sorted(ranked_list, reverse=True)
            doc_seq = doc_seq[0]
            code_seq = code_seq[0]
            rank = 0
            for k, (score, cand) in enumerate(ranked_list):
                if cand == code_seq and rank == 0:
                    rank = k + 1
                    break
            if not rank:
                count += 1
                continue
            rank_list.append([doc_seq, code_seq, rank])
            mrr += 1.0 / (rank)
            if rank < 50:
                ndcg += 1.0 / math.log2(rank + 1)
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            if rank <= 10:
                r10 += 1
            count += 1
    print(count)
    mrr /= count
    r1 /= count
    r5 /= count
    r10 /= count
    ndcg /= count
    print("MRR = ", mrr)
    print("Recall@1 = ", r1)
    print("Recall@5 = ", r5)
    print("Recall@10 = ", r10)
    print("NDCG@50 = ", ndcg)
    df = pandas.DataFrame(rank_list, columns=['Query', 'Gold', 'Rank'])
    df.to_pickle(f"../results/results_re_maxpool_{cfg['dataset']}.pkl")
    with open(f"../results/results_re_maxpool_{cfg['dataset']}.txt", "w") as f:
        f.write(f"MRR = {mrr}\n")
        f.write(f"Recall@1 = {r1}\n")
        f.write(f"Recall@5 = {r5}\n")
        f.write(f"Recall@10 = {r10}\n")
        f.write(f"NDCG@50 = {ndcg}\n")

def eval_cress():
    print("-----------------------EVAL CRESS----------------------")
    mrr = 0
    count = 0
    r1 = 0
    r5 = 0
    r10 = 0
    ndcg = 0

    rank_list = []
    sim_model.eval()

    with torch.no_grad():
        codebase = []
        docbase = []
        for j, (code_seqs, doc_seqs) in enumerate(ret1_loader):
            code_in = prepare_sequence(code_seqs, seq_len_code, word_to_ix_code)
            doc_in = prepare_sequence(doc_seqs, seq_len_doc, word_to_ix_doc)
            if torch.cuda.is_available() and use_cuda:
                doc_in = doc_in.cuda()
            if torch.cuda.is_available() and use_cuda:
                code_in = code_in.cuda()
            doc, code = sim_model(doc_in, code_in)
            doc = doc.tolist()
            code = code.tolist()
            codebase.append(code[:])
            docbase.append(doc[:])
        code_vecs = []
        doc_vecs = []
        for i, vec in enumerate(codebase):
             code_vecs.append((i, vec))
        for i, vec in enumerate(docbase):
             doc_vecs.append((i, vec))
        for i, doc in doc_vecs:
            for j, code in code_vecs:
                score = F.cosine_similarity(doc, code)
                rank_list.append((j, code, score))
            ranked_list = sorted(rank_list, key=lambda x:x[-1],reverse=True)
            rank = 0
            for k, code, scode in ranked_list:
                if k == i:
                    rank = k + 1
                    break
                rank += 1
            mrr += 1.0 / (rank+1)
            if rank < 50:
                ndcg += 1.0 / math.log2(rank + 1)
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            if rank <= 10:
                r10 += 1
            count += 1
    print(count)
    mrr /= count
    r1 /= count
    r5 /= count
    r10 /= count
    ndcg /= count
    print("MRR = ", mrr)
    print("Recall@1 = ", r1)
    print("Recall@5 = ", r5)
    print("Recall@10 = ", r10)
    print("NDCG@50 = ", ndcg)
    df = pandas.DataFrame(rank_list, columns=['Query', 'Gold', 'Rank'])
    df.to_pickle(f"../results/results_re_maxpool_{cfg['dataset']}.pkl")
    with open(f"../results/results_re_maxpool_{cfg['dataset']}.txt", "w") as f:
        f.write(f"MRR = {mrr}\n")
        f.write(f"Recall@1 = {r1}\n")
        f.write(f"Recall@5 = {r5}\n")
        f.write(f"Recall@10 = {r10}\n")
        f.write(f"NDCG@50 = {ndcg}\n")

def eval_graph():
    print("-----------------------EVAL GRAGH----------------------")
    mrr = 0
    count = 0
    r1 = 0
    r5 = 0
    r10 = 0
    ndcg = 0

    rank_list = []
    sim_model.eval()

    with torch.no_grad():
        codebase = []
        docbase = []
        for i, (code_seq, ast_seq, doc_seq) in enumerate(tqdm(ret_loader)):
            doc_in_ori = prepare_sequence(doc_seq, seq_len_doc, word_to_ix_doc)
            ranked_list = []
            for j, (code_seqs, ast_seq, doc_seq) in enumerate(ret1_loader):
                code_in = prepare_sequence(code_seqs, seq_len_code, word_to_ix_code)
                doc_in = doc_in_ori.expand(code_in.shape[0], 300)
                nodes, adjs = get_Graph_info(ast_seq, word_to_ix_code, 100, 100)
                if torch.cuda.is_available() and use_cuda:
                    doc_in = doc_in.cuda()
                    code_in = code_in.cuda()
                    nodes = nodes.cuda()
                    adjs = adjs.cuda()
                sim_score = sim_model(doc_in, code_in, nodes, adjs)
                sim_score = sim_score.tolist()
                ranked_list += list(zip(sim_score, code_seqs))
            ranked_list = sorted(ranked_list, reverse=True)
            doc_seq = doc_seq[0]
            code_seq = code_seq[0]
            rank = 0
            for k, (score, cand) in enumerate(ranked_list):
                if cand == code_seq and rank == 0:
                    rank = k + 1
                    break
            if not rank:
                count += 1
                continue
            rank_list.append([doc_seq, code_seq, rank])
            mrr += 1.0 / (rank)
            if rank < 50:
                ndcg += 1.0 / math.log2(rank + 1)
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            if rank <= 10:
                r10 += 1
            count += 1
    print(count)
    mrr /= count
    r1 /= count
    r5 /= count
    r10 /= count
    ndcg /= count
    print("MRR = ", mrr)
    print("Recall@1 = ", r1)
    print("Recall@5 = ", r5)
    print("Recall@10 = ", r10)
    print("NDCG@50 = ", ndcg)
    df = pandas.DataFrame(rank_list, columns=['Query', 'Gold', 'Rank'])
    df.to_pickle(f"../results/results_graph_{cfg['dataset']}.pkl")
    with open(f"../results/results_graph_{cfg['dataset']}.txt", "w") as f:
        f.write(f"MRR = {mrr}\n")
        f.write(f"Recall@1 = {r1}\n")
        f.write(f"Recall@5 = {r5}\n")
        f.write(f"Recall@10 = {r10}\n")
        f.write(f"NDCG@50 = {ndcg}\n")
start_time = time.time()
if model_type == 'base':
    eval_base()
elif model_type == 'REModel':
    eval_cress()
elif model_type == 'graph':
    eval_graph()
print(f"Time taked to evaluate {model_type}: {(time.time()-start_time)} seconds.")
