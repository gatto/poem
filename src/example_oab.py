import oab
from parameters import *
from pathlib import Path


if __name__ == "__main__":
    """
    1. call explanation (autoencoder, bb, and explain)
    2. call oab and import the explanation into it
    """

    oab.load(156)


    something = oab.explain(Path("data/file/to/explain.xxx"))
    type(something) == oab.Explainer

