import matplotlib.pyplot as plt
import networkx as nx
from tree_sitter import Language, Parser
from preprocess import str_process, brief_code_process
import numpy as np
import scipy as sp
import json
from graphviz import Digraph

JAVA_LANGUAGE = Language('../build/my-languages.so', 'java')


def ast_parse(str):
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    tree = parser.parse(bytes(
        "class tmp{" + str + '}',
        'utf8'))
    return tree


def pure_DFS(node,depth):
    if node is None:
        return
    print('\t'*depth,node.type)
    for child in node.children:
        pure_DFS(child,depth+1)


def DFS(node,depth,node_seq,original_code_str):
    if node is None:
        return
    if 'identifier' in node.type:
        print('\t'*depth,node.type,original_code_str[node.start_point[1]:node.end_point[1]])
        node_seq.append(original_code_str[node.start_point[1]:node.end_point[1]])
    else:
        print('\t'*depth,node.type)
        node_seq.append(node.type)
    for child in node.children:
        DFS(child,depth+1,node_seq,original_code_str)


class graph_generation():
    def __init__(self, tree, original_code_str=None, mapping_dict=None):
        self.mapping_dict = mapping_dict
        self.original_code_string = original_code_str
        #self.G = nx.MultiDiGraph()
        self.G = nx.DiGraph()
        method_begin_node = self.get_declaration_node(tree.root_node)
        # self.dfs_graph_create(tree.root_node)
        self.dfs_graph_create(method_begin_node, original_code_str=self.original_code_string)
        #self.G = self.token_chains(self.G, self.original_code_string)
        # ============================
        #DFS(method_begin_node)
        # ============================

    def dfs_graph_create(self, node, father=None, original_code_str=None):
        if node is None:
            return
        if father is None:
            self.G.add_node(node.type)
        else:
            if 'identifier' in node.type:
                # self.G.add_edge(father.type, node.type, weight=0.8)
                # self.G.add_edge(node.type, original_code_str[node.start_point[1]:node.end_point[1]], weight=0.8)

                self.G.add_edge(father.type, original_code_str[node.start_point[1]:node.end_point[1]], weight=0.8)
            else:
                self.G.add_edge(father.type, node.type, weight=0.8)
            # self.G.add_edge(father.type, node.type, weight=0.65)
        for child in node.children:
            self.dfs_graph_create(child, node, original_code_str)

    def node_relabel(self, mapping_dict):
        return nx.relabel_nodes(self.G, mapping_dict, copy=False)

    def token_chains(self, graph, original_code):
        token_chain = []
        code_token = str_process(original_code,'code').split()[3:-1]
        for i in code_token:
            if i in graph.nodes:
                token_chain.append(i)
        for i in range(len(token_chain) - 1):
            graph.add_edge(token_chain[i], token_chain[i + 1], weight=0.2)
        return graph

    def get_declaration_node(self, node):
        #if node.type == 'method_declaration':
        if ('declaration' in node.type) and ('class' not in node.type):
            return node
        if node is None:
            return
        else:
            for child in node.children:
                tmp = self.get_declaration_node(child)
                if tmp is None:
                    continue
                else:
                    return tmp

    def node_matrix(self):
        return self.G.nodes, nx.convert_matrix.to_numpy_array(self.G)

    def get_graph(self):
        return self.G


def draw_graph(graph):
    edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
    plt.figure(figsize=(14, 14))
    plt.subplot(111)
    #nx.draw(graph, pos=nx.spring_layout(graph, k=0.3, scale=0.7), node_color='r', edge_color='#000000', with_labels=True)
    nx.draw(graph, pos=nx.kamada_kawai_layout(graph, scale=1), node_color='r', edge_color=weights, with_labels=True)
    plt.show()


def token_chains(graph, original_code):
    token_chain = []
    code_token = str_process(original_code).split()[3:-1]
    for i in code_token:
        if i in graph.nodes:
            token_chain.append(i)
    for i in range(len(token_chain) - 1):
        graph.add_edge(token_chain[i], token_chain[i + 1], weight=0.35)
    return graph


def extract_var_from_ast(root_node):
    data_flow_seq=[]

    def dfs(node, depth, node_seq, original_code_str):
        if node is None:
            return
        if 'identifier' in node.type and node.children is None:
            print('\t' * depth, node.type, original_code_str[node.start_point[1]:node.end_point[1]])
            node_seq.append(original_code_str[node.start_point[1]:node.end_point[1]])
        else:
            print('\t' * depth, node.type)
            node_seq.append(node.type)
        for child in node.children:
            DFS(child, depth + 1, node_seq, original_code_str)
    return


if __name__ == '__main__':
    # print(tree.root_node.start_point[1])
    # DFS(tree.root_node)
    sample = '@private int findPLV(int M_PriceList_ID){\n  Timestamp priceDate=null;\n  String dateStr=Env.getContext(Env.getCtx(),p_WindowNo,\"DateOrdered\");\n  if (dateStr != null && dateStr.length() > 0)   priceDate=Env.getContextAsDate(Env.getCtx(),p_WindowNo,\"DateOrdered\");\n else {\n    dateStr=Env.getContext(Env.getCtx(),p_WindowNo,\"DateInvoiced\");\n    if (dateStr != null && dateStr.length() > 0)     priceDate=Env.getContextAsDate(Env.getCtx(),p_WindowNo,\"DateInvoiced\");\n  }\n  if (priceDate == null)   priceDate=new Timestamp(System.currentTimeMillis());\n  log.config(\"M_PriceList_ID=\" + M_PriceList_ID + \" - \"+ priceDate);\n  int retValue=0;\n  String sql=\"SELECT plv.M_PriceList_Version_ID, plv.ValidFrom \" + \"FROM M_PriceList pl, M_PriceList_Version plv \" + \"WHERE pl.M_PriceList_ID=plv.M_PriceList_ID\"+ \" AND plv.IsActive=\'Y\'\"+ \" AND pl.M_PriceList_ID=? \"+ \"ORDER BY plv.ValidFrom DESC\";\n  try {\n    PreparedStatement pstmt=DB.prepareStatement(sql,null);\n    pstmt.setInt(1,M_PriceList_ID);\n    ResultSet rs=pstmt.executeQuery();\n    while (rs.next() && retValue == 0) {\n      Timestamp plDate=rs.getTimestamp(2);\n      if (!priceDate.before(plDate))       retValue=rs.getInt(1);\n    }\n    rs.close();\n    pstmt.close();\n  }\n catch (  SQLException e) {\n    log.log(Level.SEVERE,sql,e);\n  }\n  Env.setContext(Env.getCtx(),p_WindowNo,\"M_PriceList_Version_ID\",retValue);\n  return retValue;\n}\n'
    sample_comment = " Find Price List Version and update context"

    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    tree = parser.parse(bytes(
        "class tmp{" + r'@Override public int runCommand(boolean mergeErrorIntoOutput,String... commands) throws IOException, InterruptedException { return runCommand(mergeErrorIntoOutput,new ArrayList<String>(Arrays.asList(commands))); }}' + '}',
        'utf8'))

    original_code = 'class tmp{@Override public int runCommand(boolean mergeErrorIntoOutput,String... commands) throws IOException, InterruptedException { return runCommand(mergeErrorIntoOutput,new ArrayList<String>(Arrays.asList(commands))); }} }'


    tree = ast_parse('@Override public int runCommand(boolean mergeErrorIntoOutput,String... commands) throws IOException, InterruptedException { return runCommand(mergeErrorIntoOutput,new ArrayList<String>(Arrays.asList(commands)));}')
    graph = graph_generation(tree, 'class tmp{@Override public int runCommand(boolean mergeErrorIntoOutput,String... commands) throws IOException, InterruptedException { return runCommand(mergeErrorIntoOutput,new ArrayList<String>(Arrays.asList(commands)));}}').get_graph()

    tree_2=ast_parse("void foo(){int x=source();if(x<max){int y=2*x;sink(y);}}")
    node_seq = []
    DFS(tree.root_node,0,node_seq,original_code)
    #pure_DFS(tree.root_node,0)
    node_seq2=[]
    pure_DFS(tree_2.root_node,0)
    DFS(tree_2.root_node,0,node_seq2,"class tmp{void foo(){int x=source();if(x<max){int y=2*x;sink(y);}}}")
    #print(str_process(original_code, 'code').split()[3:-1])
    node_seq=node_seq[6:-1]
    tmp=[]
    print(node_seq)
    for i in node_seq:
        if i in original_code:
            tmp.append(i)
        else:
            continue
    length=len(tmp)
    for index in range(length-1):
        tmp.append((tmp[index],tmp[index+1],0.2))
    tmp=tmp[length:]
    print(tmp)
    graph.add_weighted_edges_from(tmp)
    print(graph.nodes)
    draw_graph(graph)