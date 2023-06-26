import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_node("funcdef",name="add")
G.add_node("body",name="x")
G.add_edge("funcdef","body")
print(list(G.nodes))
print(list(G.edges))



nx.draw(G, with_labels=True, font_weight='bold')
plt.show()
#nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')