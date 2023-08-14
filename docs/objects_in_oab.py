import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

relationships = [
    ("Explainer", "TestPoint", {"what": "input"}),
    ("Explainer", "TreePoint", {"what": "computed"}),
    ("Explainer", "ImageExplanation", {"what": "output"}),  # many
    ("TestPoint", "Domain", {"what": "input"}),
    ("TestPoint", "BlackboxPD", {"what": "computed"}),
    ("TestPoint", "Latent", {"what": "computed"}),
    ("ImageExplanation", "Latent", {"what": "input"}),
    ("ImageExplanation", "BlackboxPD", {"what": "computed"}),
    ("TreePoint", "Domain", {"what": "input"}),
    ("TreePoint", "Latent", {"what": "input"}),
    ("TreePoint", "LatentDT", {"what": "input"}),
    ("TreePoint", "BlackboxPD", {"what": "computed"}),
    ("LatentDT", "ComplexRule", {"what": "computed"}),
    ("LatentDT", "Rule", {"what": "computed"}),  # several
    ("Domain", "AE", {"what": "loaded"}),
    ("Domain", "Blackbox", {"what": "loaded"}),
    ("ComplexRule", "Rule", {"what": "input"}),  # several
]

G.add_edges_from(relationships)


# draw it
subax1 = plt.subplot(121)
nx.draw_shell(G, with_labels=True)

plt.show()
