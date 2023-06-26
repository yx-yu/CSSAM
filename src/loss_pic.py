import matplotlib.pyplot as plt
import numpy as np

with open(f"losses/loss_file_graph_all_200.csv",encoding='utf-8') as f:
    data = np.loadtxt(f,str,delimiter=',')
    data[data == ''] = 0.0
    data = data.astype(np.float)
    #print(data[:,1])

x = np.linspace(0,100,100)
y = data[:,1]
fig, ax = plt.subplots()
plt.plot(x,y)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig('loss_graph.pdf', dpi=600, format='pdf')
plt.show()