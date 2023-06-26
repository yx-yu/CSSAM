import collections
import re
from tree_sitter import Language, Parser
import networkx as nx
import matplotlib.pyplot as plt
from DFG import extract_dfg

JAVA_LANGUAGE = Language('../build/my-languages.so', 'java')
PYTHON_LANGUAGE = Language('../build/my-languages.so', 'python')


class TCG():
    def __init__(self, code, language='java',DiGraph=True):
        """
            the connection of AST nodes is BiDirection

            the format of the code is a string
            example:
            for java code:
                int f(int a,int b){ return a+b;}

            for python:
                def f(a,b):\n\treturn a+b\n

                \t can not be replaced by "SPACE" it must be the \t
        """
        assert language in ['java', 'python']
        if DiGraph:
            self.G=nx.DiGraph()
        else:
            self.G = nx.Graph()
        ast_parsers = self.parser_init()
        self.language = language

        self.node_list = []
        self.type_list = []
        self.leaf_list = []

        if self.language == 'java':
            self.tree = ast_parsers['java'](bytes("class tmp{" + code + '}', 'utf8'))
            self.root_node = self.get_declaration_node(self.tree.root_node)
            self.java_DFS(self.root_node, self.node_list, self.type_list, self.leaf_list, "class tmp{" + code + '}')
            self.create_graph(self.G, self.root_node, self.node_list)
            """
                we add the dataflow and the token chains into the AST
            """

            code_token, dfg = extract_dfg(code)

            """insert dfg edge (Direction)"""
            try:
                for i in dfg:
                    if i[3] and i[4]:
                        for j in i[4]:
                            self.G.add_edge(self.node_list.index(self.leaf_list[j - 1]),
                                        self.node_list.index(self.leaf_list[i[1] - 1]), edge_type=2)
            except Exception:
                print(code)
            finally:
                pass
            """insert token edge (biDirection)"""
            for i in range(len(self.leaf_list) - 1):
                self.G.add_edge(self.node_list.index(self.leaf_list[i]), self.node_list.index(self.leaf_list[i + 1]),
                                edge_type=3)
            """=============================="""
        else:
            self.tree = ast_parsers['python'](bytes(code, 'utf_8'))
            self.root_node = self.tree.root_node

            function_split = self.replaceTable(code).strip().split('\n')

            for index, i in enumerate(function_split):
                function_split[index] = re.split('', i)[1:-1]
            self.python_DFS(self.root_node, self.node_list, self.type_list, self.leaf_list, function_split)
            self.create_graph(self.G, self.root_node, self.node_list)

            python_code_token, py_dfg = extract_dfg(code, 'python')

            """insert dfg edge (Direction)"""
            for i in py_dfg:
                if i[3] and i[4]:
                    for j in i[4]:
                        self.G.add_edge(self.node_list.index(self.leaf_list[j]),
                                        self.node_list.index(self.leaf_list[i[1]]),
                                        edge_type=2)
            """insert token edge (biDirection)"""
            for i in range(len(self.leaf_list) - 1):
                self.G.add_edge(self.node_list.index(self.leaf_list[i]), self.node_list.index(self.leaf_list[i + 1]),
                                edge_type=3)
                self.G.add_edge(self.node_list.index(self.leaf_list[i+1]), self.node_list.index(self.leaf_list[i]),
                                edge_type=3)
            """=============================="""

    def create_graph(self, G, node, node_list, parent=None):
        if node is None:
            return
        if parent is None:
            G.add_node(node_list.index(node))

        else:
            G.add_edge(node_list.index(parent), node_list.index(node), edge_type=1)

        for index, child in enumerate(node.children):
            self.create_graph(G, child, node_list, node)

    def get_declaration_node(self, node):
        # if node.type == 'method_declaration':
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

    def parser_init(self):
        java_parser = Parser()
        python_parser = Parser()
        java_parser.set_language(JAVA_LANGUAGE)
        python_parser.set_language(PYTHON_LANGUAGE)
        ast_parsers = {'java': java_parser.parse, 'python': python_parser.parse}
        return ast_parsers

    def java_DFS(self, node, node_seq, type_seq, leaf_seq, original_code_str):
        if node is None:
            return
        if 'identifier' in node.type:
            # print('\t'*depth,node,original_code_str[node.start_point[1]:node.end_point[1]])
            type_seq.append(original_code_str[node.start_point[1]:node.end_point[1]])
            node_seq.append(node)
        else:
            # print('\t'*depth,node)
            type_seq.append(node.type)
            node_seq.append(node)
        if not node.children:
            leaf_seq.append(node)
        for child in node.children:
            self.java_DFS(child, node_seq, type_seq, leaf_seq, original_code_str)

    def python_DFS(self, node, node_seq, type_seq, leaf_seq, split_code):
        if node is None:
            return
        if 'identifier' in node.type:
            # print(node,''.join(code[node.start_point[0]][node.start_point[1]:node.end_point[1]]))
            node_seq.append(node)
            type_seq.append(''.join(split_code[node.start_point[0]][node.start_point[1]:node.end_point[1]]))
        else:
            # print(node)
            node_seq.append(node)
            type_seq.append(node.type)
        if not node.children:
            leaf_seq.append(node)
        for child in node.children:
            self.python_DFS(child, node_seq, type_seq, leaf_seq, split_code)

    def preOrder(self):
        def preorder(node, depth):
            if node is None:
                return
            print('\t' * depth, node)
            for child in node.children:
                preorder(child, depth + 1)

        preorder(self.root_node, 0)

    def levelOrder(self):
        def levelorder(root):
            if root is None:
                return []
            result = []
            queue = collections.deque([root])
            while queue:
                for _ in range(len(queue)):
                    node = queue.popleft()
                    result.append(node)
                    queue.extend(node.children)
            return result
        return levelorder(self.root_node)

    def nodes(self):
        return self.type_list

    def edges(self,label=False):
        if label:
            return self.G.edges(data='edge_type')
        else:
            edges=[]
            for i in self.G.edges(data='edge_type'):
                edges.append((self.type_list[i[0]],self.type_list[i[1]],i[2]))
            return edges

    def adj(self):
        return nx.convert_matrix.to_numpy_array(self.G)

    def sparse_matrix(self):

        return

    def replaceTable(self, code_sentence):
        result = re.sub('    ', '\t', code_sentence)
        result = re.sub('\t', ' ', result)
        return result


if __name__=='__main__':

    java_code = 'class tmp{@Override @SuppressWarnings("unchecked") public boolean preVisit2(final ASTNode ¢){if ($.get() != null)   return false;if (!clazz.isAssignableFrom(¢.getClass()))   return true;$.set((N)¢);return false;}}'
    python_code = "def f(a,b):\n\tx=10\n\tif a>b:\n\t\treturn a\n\telse:\n\t\treturn b\n"
    # print(java_code)
    g = TCG(java_code)
    token, dfg = extract_dfg(python_code)
    edges, type = zip(*nx.get_edge_attributes(g.G, 'edge_type').items())
    plt.figure(figsize=(14, 14))
    plt.subplot(111)
    nx.draw(g.G, pos=nx.kamada_kawai_layout(g.G, scale=1), edge_color=type, with_labels=True)
    plt.show()
    print(token)
    print(dfg)