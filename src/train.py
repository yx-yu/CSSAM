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


def load_DF(fname):
    codes = ''
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            codes += line
    # print(codes)
    return codes


trainDF={}
trainDF['doc'] = load_DF(f'../dataset/all.docstring')
trainDF['code'] = load_DF(f'../dataset/all.code')
# trainDF['ast'] = load_DF(f'../dataset/all.ast')
print(1)

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
        embedding_vector = np.array(embeddings_index.get_word_vector(word), dtype='float32')
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
    train_dataset = json.load(open('../dataset/labelled_dataset_train.json', 'rb'))
    #$test_dataset = json.load(open('../dataset/labelled_dataset_test.json', 'rb'))

    train_dataset = read_json(train_dataset, 'train')
    #test_dataset = read_json(test_dataset,'test')
    codes_train, asts_train, docs_train, docs2_train = zip(*train_dataset)
    #codes_test, asts_test, docs_test, codes2_test, asts2_test = zip(*test_dataset)
    #codes_train, docs_train, docs2_train = zip(*train_dataset)
    # codes_test, docs_test, codes2_test = zip(*test_dataset)
    #print(type(eval(asts_train[0])))
    if cfg["model"] == 'base':
        train_dataset = list(zip(codes_train, docs_train, docs2_train))
    elif cfg["model"] == 'REModel':
        train_dataset = list(zip(codes_train, docs_train, docs2_train))
    elif cfg["model"] == 'graph':
        train_dataset = list(zip(codes_train, asts_train, docs_train, docs2_train))


##############################################################
#                     load dataset                           #
##############################################################
random_seed = cfg["random_seed"]
np.random.seed(random_seed)
embedding_dim = cfg["embedding_dim"]
learning_rate = cfg["learning_rate"]

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
    print("==================================================================================")
    print(len(word_to_ix_doc), weights_matrix_doc.shape)
    word_to_ix_code, weights_matrix_code = create_embeddings(f'../trained_models/code.bin', 'code')
    print(len(word_to_ix_code), weights_matrix_code.shape)
    print("==================================================================================")
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
if cfg['model'] == "base":
    sim_model = BaseModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code)
elif cfg['model'] == 'REModel':
    sim_model = REModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code)
elif cfg['model'] == 'graph':
    sim_model = GraphModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code)
if use_softmax_classifier:
    criterion = nn.CrossEntropyLoss()
else:
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()

if use_adam:
    opt = torch.optim.Adam(sim_model.parameters(), lr=learning_rate)
else:
    opt = torch.optim.SGD(sim_model.parameters(), lr=learning_rate, momentum=0.9)

# Training
iter = 0
best_loss = 10000.0
sim_model.train()
#loss_file = open(f"losses/loss_file_{model_type}_{output_dim}.csv", 'a+')
start_time = time.time()

##############################################################
#                     Train  model                           #
##############################################################
if torch.cuda.is_available() and use_cuda:
    sim_model.cuda()
    if use_parallel:
        sim_model = nn.DataParallel(sim_model)
        #sim_model=torch.nn.parallel.DistributedDataParallel(sim_model)
#train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
                                           # pin_memory=True, drop_last=True,sampler=train_sampler)
#sim_model.load_state_dict(torch.load(f"../{save_path}/sim_model_base_final_200"))
if model_type == 'base':
    sim_model.load_state_dict(torch.load(f"../{save_path}/sim_model_base_150_200"))
    for epoch in trange(num_epochs):
        epoch += 1
        batch_iter = 0
        loss_epoch = 0.0

        for code_sequence, doc_sequence, doc_sequence_neg in tqdm(train_loader):
            #doc_neg_batch = []
            #for neg_id in range(cfg["negative_examples"]):
            #    doc_neg_list = []
            #    for batch_id in range(len(doc_sequence)):
            #        code2, doc2, count2 = random.choice(train_dataset)
            #        doc_neg_list.append(doc2)
            #    doc_neg_batch.append(prepare_sequence(doc_neg_list, seq_len_doc, word_to_ix_doc))
            #doc_neg_batch = torch.stack(doc_neg_batch)
            doc_in = prepare_sequence(doc_sequence, seq_len_doc, word_to_ix_doc)
            code_in = prepare_sequence(code_sequence, seq_len_code, word_to_ix_code)
            doc_in_neg = prepare_sequence(doc_sequence_neg, seq_len_doc, word_to_ix_doc)

            if torch.cuda.is_available() and use_cuda:
                sim_model.zero_grad()
                sim_score = sim_model(doc_in.cuda(), code_in.cuda())
                sim_score_neg = sim_model(doc_in_neg.cuda(), code_in.cuda())
                loss = 0.05 - sim_score + sim_score_neg
                loss[loss < 0] = 0.0
                loss = torch.sum(loss)
                loss.backward()
                opt.step()
                loss_epoch += loss
            else:
                sim_score = sim_model(doc_in, code_in)
                #sim_score_neg = sim_model(doc_in_neg, code_in)

            iter += 1
            batch_iter += 1
        loss_epoch /= batch_iter
        tqdm.write(f"Epoch: {epoch}. Loss: {loss_epoch}. Model: {model_type}")
        loss_file = open(f"losses/loss_file_{model_type}_{output_dim}.csv", 'a+')
        loss_file.write(f"{epoch},{loss_epoch}\n")
        loss_file.close()
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(sim_model.module.state_dict(), f"../{save_path}/sim_model_base_final_{output_dim}")
        if (epoch) % cfg["save_every"] == 0:
            torch.save(sim_model.module.state_dict(), f"../{save_path}/sim_model_base_{(epoch)}_{output_dim}")

if model_type == 'REModel':
    sim_model.load_state_dict(torch.load(f"../{save_path}/sim_model_maxpool_re_final_200"))
    for epoch in trange(0,num_epochs):
        epoch += 1
        batch_iter = 0
        loss_epoch = 0.0

        for code_sequence, doc_sequence, doc_sequence_neg in tqdm(train_loader):
            #doc_neg_batch = []
            #for neg_id in range(cfg["negative_examples"]):
            #    doc_neg_list = []
            #    for batch_id in range(len(doc_sequence)):
            #        code2, doc2, count2 = random.choice(train_dataset)
            #        doc_neg_list.append(doc2)
            #    doc_neg_batch.append(prepare_sequence(doc_neg_list, seq_len_doc, word_to_ix_doc))
            #doc_neg_batch = torch.stack(doc_neg_batch)
            doc_in = prepare_sequence(doc_sequence, seq_len_doc, word_to_ix_doc)
            code_in = prepare_sequence(code_sequence, seq_len_code, word_to_ix_code)
            doc_in_neg = prepare_sequence(doc_sequence_neg, seq_len_doc, word_to_ix_doc)

            if torch.cuda.is_available() and use_cuda:
                sim_model.zero_grad()
                sim_score = sim_model(doc_in.cuda(), code_in.cuda())
                sim_score_neg = sim_model(doc_in_neg.cuda(), code_in.cuda())
                loss = 0.05 - sim_score + sim_score_neg
                loss[loss < 0] = 0.0
                loss = torch.sum(loss)
                loss.backward()
                opt.step()
                loss_epoch += loss
            else:
                sim_score = sim_model(doc_in, code_in)
                #sim_score_neg = sim_model(doc_in_neg, code_in)

            iter += 1
            batch_iter += 1
        loss_epoch /= batch_iter
        tqdm.write(f"Epoch: {epoch}. Loss: {loss_epoch}. Model: {model_type}")
        loss_file = open(f"losses/loss_file_{model_type}_{output_dim}_maxpool.csv", 'a+')
        loss_file.write(f"{epoch},{loss_epoch}\n")
        loss_file.close()
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(sim_model.module.state_dict(), f"../{save_path}/sim_model_maxpool_re_final_{output_dim}")
        if (epoch) % cfg["save_every"] == 0:
            torch.save(sim_model.module.state_dict(), f"../{save_path}/sim_model_maxpool_re_{(epoch)}_{output_dim}")

if model_type == 'graph':
    for epoch in trange(num_epochs):
        epoch += 1
        batch_iter = 0
        loss_epoch = 0.0

        for code_sequence, ast, doc_sequence,  doc_sequence_neg in tqdm(train_loader):
            # doc_neg_batch = []
            # for neg_id in range(cfg["negative_examples"]):
            #     doc_neg_list = []
            #     for batch_id in range(len(doc_sequence)):
            #         code2,  ast2, doc2 ,count2 = random.choice(train_dataset)
            #         doc_neg_list.append(doc2)
            #     doc_neg_batch.append(prepare_sequence(doc_neg_list, seq_len_doc, word_to_ix_doc))
            # doc_neg_batch = torch.stack(doc_neg_batch)
            doc_in = prepare_sequence(doc_sequence, seq_len_doc, word_to_ix_doc)
            code_in = prepare_sequence(code_sequence, seq_len_code, word_to_ix_code)
            doc_in_neg = prepare_sequence(doc_sequence_neg, seq_len_doc, word_to_ix_doc)
            #print(ast[0])
            nodes,adjs = get_Graph_info(ast, word_to_ix_code, 100, 100)
            #print(nodes.shape,adjs.shape)
            if torch.cuda.is_available() and use_cuda:
                # for neg_id in range(cfg["negative_examples"]):
                #     sim_model.zero_grad()
                #     sim_score = sim_model(doc_in.cuda(), code_in.cuda(), nodes.cuda(),adjs.cuda())
                #     #sim_score_neg = sim_model(doc_neg_batch[neg_id].cuda(), code_in.cuda(),nodes.cuda(),adjs.cuda())
                #     #print(sim_score_neg,sim_score)
                #     loss = 0.05 - sim_score + sim_score_neg
                #     loss[loss < 0] = 0.0
                #     loss = torch.sum(loss)
                #     loss.backward()
                #     opt.step()
                #     loss_epoch += loss
                # # sim_score_neg, _, _ = sim_model(doc_in_neg.cuda(), code_in.cuda())
                sim_model.zero_grad()
                sim_score = sim_model(doc_in.cuda(), code_in.cuda(), nodes.cuda(),adjs.cuda())
                sim_score_neg = sim_model(doc_in_neg.cuda(), code_in.cuda(),nodes.cuda(),adjs.cuda())
                loss = 0.05 - sim_score + sim_score_neg
                loss[loss < 0] = 0.0
                loss = torch.sum(loss)
                loss.backward()
                opt.step()
                loss_epoch += loss
                # sim_score_neg, _, _ = sim_model(doc_in_neg.cuda(), code_in.cuda())
            else:
                sim_score = sim_model(doc_in, code_in)
                #sim_score_neg = sim_model(doc_in_neg, code_in)

            iter += 1
            batch_iter += 1
        loss_epoch /= batch_iter
        tqdm.write(f"Epoch: {epoch}. Loss: {loss_epoch}. Model: {model_type}")
        loss_file = open(f"losses/loss_file_{model_type}_F_{output_dim}.csv", 'a+')
        loss_file.write(f"{epoch},{loss_epoch}\n")
        loss_file.close()
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_F_final_{output_dim}")
        if (epoch) % cfg["save_every"] == 0:
            torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_F_{(epoch)}_{output_dim}")