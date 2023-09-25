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

#    ("TreePoint", "id: int", {"what": "input"}),
#    ("TreePoint", "a: np.ndarray", {"what": "input"}),
#    ("LatentDT", "predicted_class: str", {"what": "computed"}),
#    ("LatentDT", "fidelity: float", {"what": "computed"}),
#    ("LatentDT", "model: sklearn.DecisionTreeClassifier", {"what": "computed"}),
#    ("Domain", "dataset_name", {"what": "input"}),
#    ("Domain", "bb_type", {"what": "input"}),
#    ("Domain", "metadata", {"what": "input"}),
#    ("Domain", "classes", {"what": "input"}),
#    ("Latent", "a: np.ndarray", {"what": "input"}),
#    ("Latent", "margins: np.ndarray", {"what": "input"}),

G.add_edges_from(relationships)


# draw it
subax1 = plt.subplot(121)
nx.draw_shell(G, with_labels=True)

plt.show()
