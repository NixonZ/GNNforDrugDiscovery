from graph import sequence_on_graph
import networkx as nx
import matplotlib.pyplot as plt

G = nx.path_graph(4)

print(sequence_on_graph(G))
plt.figure()
nx.draw(G)
plt.show()