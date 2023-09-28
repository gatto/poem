import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

relationships = [
    ("TreePoint", "Domain", {"what": "input"}),
    ("TreePoint", "Latent", {"what": "input"}),
    ("TreePoint", "LatentDT", {"what": "input"}),
    ("TreePoint", "BlackboxPD", {"what": "computed"}),
    ("LatentDT", "ComplexRule", {"what": "computed"}),
    ("Domain", "AE", {"what": "loaded"}),
    ("Domain", "Blackbox", {"what": "loaded"}),
    ("Domain", "explanation_base", {"what": "loaded"}),
    ("ComplexRule", "Condition", {"what": "input"}),  # several
]

G.add_edges_from(relationships)

# Define custom positions
pos = {
    "TreePoint": (-1, -1.5),
    "Domain": (0, -1),
    "Latent": (0, -2),
    "LatentDT": (0, 0),
    "BlackboxPD": (0, -3),
    "ComplexRule": (1, 0),
    "AE": (1, -2),
    "Blackbox": (1, -1),
    "explanation_base": (1, -3),
    "Condition": (2, 0),
}

fig = plt.figure(1, figsize=(10, 4))

# Draw the graph
nx.draw(
    G,
    pos,
    with_labels=False,
    node_size=700,
    node_color="orangered",
    font_size=10,
    font_weight="bold",
)

# Adjust label positions
label_pos = {
    "TreePoint": (-1.1, -1.25),
    "Domain": (0, -0.75),
    "Latent": (0, -1.75),
    "LatentDT": (-0.23, 0),
    "BlackboxPD": (0.12, -2.75),
    "ComplexRule": (1, -0.3),
    "AE": (1.13, -2),
    "Blackbox": (1.23, -1),
    "explanation_base": (1.35, -3),
    "Condition": (2, -0.3),
}

nx.draw_networkx_labels(G, label_pos)
plt.show()
