import matplotlib.pyplot as plt
import os

project_root = r"C:\Users\Karthik S\Documents\Infosys-Internship\circuitguard"
graph_path = os.path.join(project_root, "training_performance.png")

img = plt.imread(graph_path)
plt.imshow(img)
plt.axis('off')
plt.show()
