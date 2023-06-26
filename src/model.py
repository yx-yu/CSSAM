import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import TransformerEncoderLayer
from modules import Module, ModuleList, ModuleDict
from modules.embedding import Embedding
from modules.encoder import Encoder
from modules.alignment import Alignment as alignment
from modules.fusion import FullFusion as fusion
from modules.pooling import Pooling
import networkx as nx
import numpy as np
import torch.nn.functional as F
import yaml
import torch_geometric.nn as pyg
import sys

with open("../config.yml", 'r') as config_file:
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)

hidden_size = cfg["hidden_size"]
dense_dim = cfg["dense_dim"]
output_dim = cfg["output_dim"]
num_layers_lstm = cfg["num_layers_lstm"]
use_cuda = cfg["use_cuda"]
use_bidirectional = cfg["use_bidirectional"]



def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class CrossBlock(Module):
    def __init__(self,doc_weights_matrix,code_weights_matrix,output_dim,dropout=0.2,hidden_size=200,device="cuda",blocks=2):
        super().__init__()
        self.docembedding, num_embeddings, embedding_dim = create_emb_layer(doc_weights_matrix, True)
        self.codembedding, _,_ = create_emb_layer(code_weights_matrix, True)
        self.dropout = dropout
        self.device = device
        self.blocks = ModuleList([ModuleDict({
            'alignment': alignment(embedding_dim + hidden_size),
            'fusion': fusion(
                embedding_dim),
            'connection': nn.Linear(hidden_size * 2, embedding_dim),
        }) for i in range(blocks)])

        self.crossA = nn.Linear(embedding_dim, embedding_dim)
        self.crossB = nn.Linear(embedding_dim, embedding_dim)
        self.pooling = Pooling()
        self.Linear = nn.Linear(2*output_dim, output_dim)

    def forward(self, a, b):
        mask_a = torch.ne(a, 0).unsqueeze(2).to(self.device).bool()
        mask_b = torch.ne(b, 0).unsqueeze(2).to(self.device).bool()
        a = self.docembedding(a)
        b = self.codembedding(b)
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = block['connection'](a)
                b = block['connection'](b)
                res_a, res_b = a, b

            a_enc = self.crossA(a)  # embedding_dim
            b_enc = self.crossB(b)  # embedding_dim
            a = a_enc * res_a + a
            b = b_enc * res_b + b
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        a = self.Linear(a)
        b = self.Linear(b)
        return a, b

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, outdim, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, outdim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LSTMModel(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, dense_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        if use_bidirectional:
            self.num_layers = num_layers * 2
        else:
            self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=use_bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(dense_dim*(int(use_bidirectional)+1), dense_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(dense_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if torch.cuda.is_available() and use_cuda:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Initialize cell state
        if torch.cuda.is_available() and use_cuda:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        else:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(self.embedding(x), (h0, c0))
        # out = self.fc(out[:, -1, :])
        out, _ = torch.max(out, dim=1, keepdim=False, out=None)
        out = self.fc(out)

        return out

class SelfAttnModel(nn.Module):
    def __init__(self, weights_matrix, hidden_size, dense_dim, output_dim):
        super(SelfAttnModel, self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=hidden_size,
            dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        out = self.transformer_encoder(self.embedding(x))
        out, _ = torch.max(out, dim=1, keepdim=False, out=None)
        out = self.fc(out)
        return out

class BaseModel(nn.Module):
    def __init__(self, weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code):
        super(BaseModel, self).__init__()
        if cfg["encoder"]=='Transformer':
            self.doc_model = SelfAttnModel(weights_matrix_doc, hidden_size, dense_dim, output_dim)
            self.code_model = SelfAttnModel(weights_matrix_code, hidden_size, dense_dim, output_dim)
        elif cfg["encoder"]=='LSTM':
            self.doc_model = LSTMModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim)
            self.code_model = LSTMModel(weights_matrix_code, hidden_size, num_layers_lstm, dense_dim, output_dim)
        else:
            print("Encoder must be Transformer or LSTM")
            exit()
        self.dist = nn.modules.distance.PairwiseDistance(p=2, eps=1e-10)

    def forward(self, doc_in, code_in):
        doc_vector = self.doc_model(doc_in)
        code_vector = self.code_model(code_in)
        sim_score = 1.0-self.dist(doc_vector, code_vector)
        return sim_score


class REModel(nn.Module):
    def __init__(self, weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim,weights_matrix_code):
        super(REModel, self).__init__()
        if cfg["encoder"] == 'Transformer':
            self.doc_model = SelfAttnModel(weights_matrix_doc, hidden_size, dense_dim, output_dim)
            self.code_model = SelfAttnModel(weights_matrix_code, hidden_size, dense_dim, output_dim)
        elif cfg["encoder"] == 'LSTM':
            self.doc_model = LSTMModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim)
            self.code_model = LSTMModel(weights_matrix_code, hidden_size, num_layers_lstm, dense_dim, output_dim)
        else:
            print("Encoder must be Transformer or LSTM")
            exit()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.Re2Block = CrossBlock(weights_matrix_doc,weights_matrix_code,output_dim)
        self.dist = nn.modules.distance.PairwiseDistance(p=2, eps=1e-10)
        #self.pool = nn.MaxPool1d(2,stride=2)

    def forward(self, doc_in, code_in):
        doc_vector1 = self.doc_model(doc_in)
        code_vector1 = self.code_model(code_in)
        
        doc_vector2, code_vector2 = self.Re2Block(doc_in, code_in)
        doc_vector = torch.cat((doc_vector1,doc_vector2),1)
        code_vector = torch.cat((code_vector1, code_vector2),1)
        # sim_score = 1.0 - self.dist(doc_vector, code_vector)
        sim_score = self.cos(doc_vector, code_vector)  
        return sim_score


class GraphModel(nn.Module):
    def __init__(self, weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code, nfeat=128, nhid=128):
        super(GraphModel, self).__init__()
        if cfg["encoder"] == 'Transformer':
            self.doc_model = SelfAttnModel(weights_matrix_doc, hidden_size, dense_dim, output_dim)
            self.code_model = SelfAttnModel(weights_matrix_code, hidden_size, dense_dim, output_dim)
        elif cfg["encoder"] == 'LSTM':
            self.doc_model = LSTMModel(weights_matrix_doc, hidden_size, num_layers_lstm, dense_dim, output_dim)
            self.code_model = LSTMModel(weights_matrix_code, hidden_size, num_layers_lstm, dense_dim, output_dim)
        else:
            print("Encoder must be Transformer or LSTM")
            exit()
        self.Re2Block = CrossBlock(weights_matrix_doc,weights_matrix_code,output_dim)
        self.GatLayer = pyg.GATConv(128,50,4)
        #self.dist = nn.modules.distance.PairwiseDistance(p=2, eps=1e-10)
        self.attn = nn.Parameter(torch.randn(200,200))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,output_dim),
            nn.ReLU(),
        )
        self.W_doc = nn.Linear(output_dim, output_dim)
        self.W_docre = nn.Linear(output_dim, output_dim)
        self.W_code = nn.Linear(output_dim, output_dim)
        self.W_codere = nn.Linear(output_dim, output_dim)
        self.W_ast = nn.Linear(output_dim, output_dim)
        self.W_docs = nn.Linear(output_dim * 2, output_dim * 2)
        self.W_codes = nn.Linear(output_dim * 3, output_dim * 2)
        self.W_a = nn.Linear(output_dim, 1)

    def forward(self, doc_in, code_in, nodes, adjs):
        batch_size = len(doc_in)
        doc_vector1 = self.doc_model(doc_in)
        code_vector1 = self.code_model(code_in)
        doc_vector1_d = self.W_doc(doc_vector1)
        code_vector1_d = self.W_code(code_vector1)
        doc_vector2, code_vector2 = self.Re2Block(doc_in, code_in)
        doc_vector2_d = self.W_docre(doc_vector2)
        code_vector2_d = self.W_codere(code_vector2)
        adj = None
        offset = 0
        nfeats = nodes.view(-1,128)
        for ad in adjs:
            if adj == None:
                adj = torch.nonzero(ad).contiguous().t() 
            else:
                adj = torch.cat((adj, torch.nonzero(ad).contiguous().t()+offset), dim = 1)
            offset += 100
        code_vector3 = self.GatLayer(nfeats, adj)
        code_vector3 = code_vector3.view(batch_size, -1, 200)
        code_vector3 = code_vector3.mean(1)
        #for nfeat,adj in zip(nodes,adjs):
        #    #nfeat, adj = get_Graph_info(eval(g),self.node_embed_matrix)
        #    graphvec = self.GatLayer(nfeat.cuda(),adj.cuda())
        #    if code_vector3 == None:
        #        code_vector3 = graphvec.mean(0).unsqueeze(0)
        #    else:
        #        code_vector3 = torch.cat((code_vector3,graphvec.mean(0).unsqueeze(0)),0)
        code_vector3_d = self.W_ast(code_vector3)
        doc_vector1_tanh = torch.tanh(doc_vector1_d)
        doc_vector2_tanh = torch.tanh(doc_vector2_d)
        code_vector1_tanh = torch.tanh(code_vector1_d)
        code_vector2_tanh = torch.tanh(code_vector2_d)
        code_vector3_tanh = torch.tanh(code_vector3_d)
        #print(code_vector2.shape,code_vector1.shape)
        #code_vector = torch.cat((code_vector1, code_vector2), 1)
        #code_vector = self.pool(code_vector)
        #code_vector2 = self.fc(torch.tanh(code_vector2))
        #doc_vector = self.fc(torch.tanh(doc_vector1 + doc_vector2))
        #code_vector = self.fc(torch.tanh(code_vector1 + code_vector2 + code_vector3))
        #code_vector = self.fc(torch.tanh(code_vector))
        #print(code_vector,doc_vector)
        doc_vector1_scalar = self.W_a(F.dropout(doc_vector1_tanh, 0.2))
        doc_vector2_scalar = self.W_a(F.dropout(doc_vector2_tanh, 0.2))
        code_vector1_scalar = self.W_a(F.dropout(code_vector1_tanh,0.2))
        code_vector2_scalar = self.W_a(F.dropout(code_vector2_tanh, 0.2))
        code_vector3_scalar = self.W_a(F.dropout(code_vector3_tanh, 0.2))
        doc_attn_catted = torch.cat([doc_vector1_scalar,doc_vector2_scalar],1)
        code_attn_catted = torch.cat([code_vector1_scalar,code_vector2_scalar,code_vector3_scalar],1)
        doc_atten_weight = F.softmax(doc_attn_catted, dim=1)
        code_atten_weight = F.softmax(code_attn_catted, dim=1)

        doc_vector1 = torch.bmm(doc_atten_weight[:,0].reshape(batch_size,1,1),doc_vector1.reshape(batch_size,1,200)).reshape(batch_size,200)
        doc_vector2 = torch.bmm(doc_atten_weight[:,1].reshape(batch_size,1,1),doc_vector2.reshape(batch_size,1,200)).reshape(batch_size,200)

        code_vector1 = torch.bmm(code_atten_weight[:,0].reshape(batch_size,1,1), code_vector1.reshape(batch_size,1,200)).reshape(batch_size,200)
        code_vector2 = torch.bmm(code_atten_weight[:,1].reshape(batch_size,1,1), code_vector2.reshape(batch_size,1,200)).reshape(batch_size,200)
        code_vector3 = torch.bmm(code_atten_weight[:,2].reshape(batch_size,1,1), code_vector3.reshape(batch_size,1,200)).reshape(batch_size,200)

        doc_vector = self.W_docs(torch.cat((doc_vector1,doc_vector2),1))
        code_vector = self.W_codes(torch.cat((code_vector1,code_vector2,code_vector3),1))
        sim_score = self.cos(doc_vector, code_vector)
        if self.training:
            return sim_score
        else:
            return doc_vector, code_vector

