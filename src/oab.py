import io
import json
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import ClassVar

import abele
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn_json as skljson
from attrs import define, field, validators
from mnist import get_autoencoder, get_data, get_dataset_metadata, run_explain
from rich import print
from rich.console import Console
from rich.table import Table
from sklearn.neighbors import NearestNeighbors

data_table_structure = (
    "id int",
    "a array",
    "latent array",
    "DTpredicted int",
    "DTmodel dictionary",
    "DTfidelity float",
    "srules str",
    "scounterrules str",
    "BBpredicted int",
    "classes str",
)

data_path = Path("./data/oab")
data_table_path = data_path / "mnist.db"
operators = [">=", "<=", ">", "<"]


@define
class Rule:
    """
    structure of a Rule:
    feature:int  the latent feature the rules checks
    operator:str  between the following choices:
        - >
        - >=
        - <
        - <=
    value:float  the value of the latent feature
    target_class:str the class this targets

    methods:
    .marginal_apply(Latent, eps=0.01)
    returns a Latent object with the rule applied marginally
    e.g. if the rule is 2 > 5.0
    returns Latent with Latent.a feature 2 == 5 + eps
    (regardless of what feature 2's value was in Latent)
    **This should be useful for counterfactual image generation**

    .respect(Latent)
    returns a Latent object with the rule respected
    on feature, but still varying the feature by **at least** some margin
    **this is not the correct approach for factual generation**
    """

    feature: int
    operator: str
    value: float
    # TODO: make target_class an index of Domain.classes again?
    # remember to correct the rule/counterrules extraction in LatentDT
    target_class: str


@define
class Domain:
    classes: list[str]


@define
class LatentDT:
    predicted_class: int  # index of classes, refers to Domain.classes
    model: sklearn.tree._classes.DecisionTreeClassifier
    fidelity: float
    s_rules: str
    s_counterrules: str
    model_json: dict = field(
        init=False,
        repr=lambda value: f"{type(value)}",
    )
    rules: list[Rule] = field(init=False)
    counterrules: list[Rule] = field(init=False)

    @model_json.default
    def _model_json_default(self):
        return skljson.to_dict(self.model)

    @rules.default
    def _rules_default(self):
        """
        This converts s_rules:str to rules:list[Rule]
        """
        results = []
        """print(f"pre work: {self.s_rules}")
        working = self.s_rules.translate(str.maketrans("", "", "{} ")).split(",")
        print(working)
        for my_rule in working:
            print(my_rule)
        exit(1)
        """
        return results

    @counterrules.default
    def _counterrules_default(self):
        """
        This converts s_counterrules:str to counterrules:list[Rule]
        """
        results = []
        print(f"pre work: {self.s_counterrules}")
        # str.maketrans's third argument indicates characters to remove with str.translate(•)
        all_rules = self.s_counterrules.translate(str.maketrans("", "", "{} ")).split(
            ","
        )
        print(all_rules)

        if all_rules:
            for my_rule in all_rules:
                print(my_rule)
                parts = my_rule.split("-->")
                print(parts)
                parts[1] = parts[1][parts[1].find(":") + 1 :]
                print(parts)
                for operator in operators:
                    if operator in parts[0]:
                        parts[0] = parts[0].split(operator)
                        break
                results.append(
                    Rule(
                        feature=int(parts[0][0]),
                        operator=operator,
                        value=float(parts[0][1]),
                        target_class=parts[1],
                    )
                )

            print(results)
        return results


@define
class Latent:
    """
    a: the record in latent representation
    """

    a: np.ndarray = field(
        validator=validators.instance_of(np.ndarray),
        repr=lambda value: str(value) if len(value) < 10 else str(type(value)),
    )
    # TODO
    # margins: np.ndarray with feature, min, max
    # space: bool


@define
class Blackbox:
    predicted_class: int  # index of classes, refers to Domain.classes


@define
class TreePoint:
    """
    TreePoint.id is the index of the record in the passed dataset
    TreePoint.a is the original array in real space
    """

    id: int = field(validator=validators.instance_of(int))
    a: np.ndarray = field(
        validator=validators.instance_of(np.ndarray),
        repr=lambda value: f"{type(value)}",
    )
    latent: Latent
    latentdt: LatentDT
    blackbox: Blackbox
    domain: Domain
    # true_class: int  # index of classes, refers to Domain.classes

    def save(self):
        con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()

        # TODO: can i automate this also based on db schema?
        data = (
            self.id,
            self.a,
            self.latent.a,
            self.latentdt.predicted_class,
            self.latentdt.model_json,
            self.latentdt.fidelity,
            self.latentdt.s_rules,
            self.latentdt.s_counterrules,
            self.blackbox.predicted_class,
            self.domain.classes,
        )
        cur.execute(f"INSERT INTO data VALUES {_data_table_structure_query()}", data)
        con.commit()
        con.close()


@define
class TestPoint:
    a: np.ndarray = field(
        validator=validators.instance_of(np.ndarray),
        repr=lambda value: f"{type(value)}",
    )
    blackbox: Blackbox
    domain: Domain
    latent: Latent = field()

    @latent.default
    def _latent_default(self):
        """
        encodes the TestPoint.a to build TestPoint.Latent.a
        """
        mtda = get_dataset_metadata()
        (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = get_data()

        ae: abele.adversarial.AdversarialAutoencoderMnist = get_autoencoder(
            np.expand_dims(self.a, axis=0),
            mtda["ae_name"],
            mtda["dataset"],
            mtda["path_aemodels"],
        )
        ae.load_model()

        miao = ae.encode(np.expand_dims(self.a, axis=0))
        return Latent(a=miao[0])

    @classmethod
    def generate_test(cls):
        """
        can use TestPoint.generate_test to get a TestPoint usable for testing
        (it's the point with id=0 in the sql db)
        """
        my_point = load(0)
        return cls(
            a=my_point.a,
            blackbox=Blackbox(predicted_class=my_point.blackbox.predicted_class),
            domain=Domain(classes=my_point.domain.classes),
        )


@define
class Explainer:
    """
    this is what oab.py returns when you ask an explanation
    """

    testpoint: TestPoint
    target: TreePoint = field(init=False)
    counterfactuals: list[np.ndarray] = field(init=False)
    factuals: list[np.ndarray] = field(init=False)

    @target.default
    def _closest_neighbor_default(self) -> TreePoint:
        return knn(self.testpoint)

    @counterfactuals.default
    def _counterfactuals_default(self):
        print("Doing [green]counterfactuals[/]")
        try:
            # TODO: understand this
            # what do i have?
            # i have the treepoint from which to draw rules, crules
            # i have said rules, crules
            # i have the testpoint on which to apply rules, crules
            cfactuals = get_counterfactual_prototypes(eps=0.01)
            for rule in target.rules:
                pass

            for i, cpimg in enumerate(cfactuals):
                bboc = bb_predict(np.array([cpimg]))[0]
                plt.imshow(cpimg)
                plt.title("cf - black box %s" % bboc)
                plt.savefig(
                    data_path / f"counter_{i}.png",
                    dpi=150,
                )
        except:
            print("very bad during counterfactuals")
            exit(1)
        print(f"I made #{i+1} counterfactuals.")
        return "something xxx"

    @factuals.default
    def _factuals_default(self):
        print("Doing [green]factuals[/]")
        try:
            # TODO: understand this
            factuals = exp.get_prototypes_respecting_rule(num_prototypes=3)
            for i, pimg in enumerate(factuals):
                bbo = bb_predict(np.array([pimg]))[0]
                if use_rgb:
                    plt.imshow(pimg)
                else:
                    plt.imshow(pimg.astype("uint8"), cmap="gray")
                plt.title(f"prototype {bbo}")
                plt.savefig(
                    data_path / f"factual_{i}.png",
                    dpi=150,
                )
        except:
            print("very bad during factuals")
            exit(1)
        print(f"I made #{i+1} factuals.")

        return "something xxx"

    @classmethod
    def from_file(cls, my_path: Path):
        """
        This is the main method that should be exposed externally.
        intended usage:

        from oab import Explainer
        explanation = Explainer.from_file(<path_to_image>)

        the format of the Explainer object is still to be defined (TODO)

        TODO: structure of this method is
        1. create a TestPoint from file in my_path
        2. explain somehow
        3. generate Explainer which will contain TestPoint
        """

        return cls()

    @classmethod
    def get_counterfactual_prototypes(cls, eps=0.01, interp=0):
        cprototypes = list()
        for delta in self.deltas:
            limg_new = self.limg.copy()
            for p in delta:
                if p.op == ">":
                    limg_new[p.att] = p.thr + eps
                else:
                    limg_new[p.att] = p.thr - eps

            img_new = self.autoencoder.decode(limg_new.reshape(1, -1))[0]
            cprototypes.append(img_new)

        return cprototypes

    @classmethod
    def _expl(cls, a: np.ndarray):
        """
        this is the private funct that actually explains
        TODO: should this be here or in TestPoint?
        """

        pass


def decode_rules(str) -> list[Rule]:
    pass


def knn(point: TestPoint) -> TreePoint:
    """
    this returns only the closest TreePoint to the inputted point `a`
    (in latent space representation)
    """

    neigh = NearestNeighbors(n_neighbors=1)

    points: list[TreePoint] = load_all()
    latent_arrays: list[np.ndarray] = [point.latent.a for point in points]

    # I train this on the np.ndarray latent repr of the points,
    neigh.fit(latent_arrays)

    fitted_model = neigh.kneighbors([point.latent.a])
    # if I need the distance it's here…
    distance: np.float64 = fitted_model[0][0][0]
    index: np.int64 = fitted_model[1][0][0]

    # I return the entire TreePoint though
    return points[index]


def list_all() -> list[int]:
    """
    Returns a list of TreePoint ids that are in the db.
    """
    con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    rows = cur.execute("SELECT id FROM data")
    rows = rows.fetchall()
    con.close()
    return sorted([x["id"] for x in rows])


def load(id: int) -> None | TreePoint:
    """
    Loads a TrainPoint if you pass an id:int
    TODO: Loads a set of TrainPoint if you pass an id: collection
    """
    if isinstance(id, int):
        con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        row = cur.execute("SELECT * FROM data WHERE id = ?", (id,))
        # print([element[0] for element in res.description]) # this gives table column names
        row = row.fetchall()
        con.close()

        if len(row) == 0:
            return None
        elif len(row) > 1:
            raise Exception(
                f"the id {id} is supposed to be unique but it's not in this database"
            )
        else:
            row = row[0]  # there is only one row anyway
            assert id == row["id"]

            # TODO: rebuild LatentDT.model from json
            rebuilt_dt = skljson.from_dict(row["DTmodel"])

            # TODO: can i automate this based on the db schema?
            return TreePoint(
                id=id,
                a=row["a"],
                latent=Latent(a=row["latent"]),
                latentdt=LatentDT(
                    predicted_class=row["DTpredicted"],
                    model=rebuilt_dt,
                    fidelity=row["DTfidelity"],
                    s_rules=row["srules"],
                    s_counterrules=row["scounterrules"],
                ),
                blackbox=Blackbox(predicted_class=row["BBpredicted"]),
                domain=Domain(classes=row["classes"]),
            )
    else:
        raise ValueError(f"id was not an int: {id}")


def load_all() -> list[TreePoint]:
    """
    Returns a list of all TreePoints that are in the sql db
    """
    results = []

    for i in list_all():
        results.append(load(i))
    return results


def _data_table_structure_query() -> str:
    """
    Creates the table structure query for data INSERT
    """
    my_query = "("
    for column in data_table_structure:
        my_query = f"{my_query}?, "
    return f"{my_query[:-2]})"


def _connect():
    # TODO: insert a check that the table exists using
    # res = cur.execute("SELECT name FROM sqlite_master WHERE name='spam'")
    # res.fetchone() is None
    pass
    # return cur


def _delete_create_table() -> None:
    data_table_path.unlink(missing_ok=True)

    data_table_string = "("
    for column in data_table_structure:
        data_table_string = f"{data_table_string}{column}, "
    data_table_string = f"{data_table_string[:-2]})"

    con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    cur.execute(f"CREATE TABLE data{data_table_string}")
    # must be smth like "CREATE TABLE data(id, original array)"
    con.close()


def _adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


"""
con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("create table test (arr array)")
"""

console = Console()
sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("array", _convert_array)
sqlite3.register_adapter(dict, lambda d: json.dumps(d).encode("utf8"))
sqlite3.register_converter("dictionary", lambda d: json.loads(d.decode("utf8")))


if __name__ == "__main__":
    """
    This should train the offline system
    1. explain each of the data points in tree_set
    2. save them as TrainPoint
    we have a sqlite populated with trainpoints.
    """
    try:
        run_options = sys.argv[1]
    except IndexError:
        raise Exception(
            """possible runtime arguments are:\n
            (testing) delete-all, run-tests, test-train,\n
            (production) train, list, explain <path of img to explain>"""
        )

    if run_options == "delete-all":
        # delete create table
        _delete_create_table()
        print(f"done {run_options}")
    elif run_options == "run-tests":
        _delete_create_table()
        miao = TreePoint(156, np.array(([4, 5], [1, 2])))
        miao.save()

        # visualization
        table = Table(title="TrainPoint schema")
        table.add_column("attribute", style="cyan", no_wrap=True)
        table.add_column("dType", style="magenta")
        for attribute in [
            a
            for a in dir(load(156))
            if not a.startswith("__") and not callable(getattr(load(156), a))
        ]:
            table.add_row(f"{attribute}", f"{type(attribute)}")
        console.print(table)
        print(f"[red]Example:[/] {load(156)}")
    elif run_options in (
        "train",
        "test-train",
    ):  # test-train should be used until real train run
        (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = get_data()

        if run_options == "test-train":
            # only for test purposes
            X_tree = X_tree[:100]
            Y_tree = Y_tree[:100]

        for i, point in enumerate(X_tree):
            try:
                with open(
                    Path(get_dataset_metadata()["path_aemodels"])
                    / f"explanation/{i}.pickle",
                    "rb",
                ) as f:
                    tosave = pickle.load(f)
            except FileNotFoundError:
                tosave = run_explain(i, X_tree, Y_tree)

            # the following creates the actual data point
            # TODO: can i automate this also based on db schema?
            miao = TreePoint(
                id=i,
                a=point,
                latent=Latent(a=tosave["limg"]),
                latentdt=LatentDT(
                    predicted_class=tosave["dt_pred"],
                    model=tosave["dt"],
                    fidelity=tosave["fidelity"],
                    s_rules=str(tosave["rstr"]),
                    s_counterrules=tosave["cstr"],
                ),
                blackbox=Blackbox(predicted_class=tosave["bb_pred"]),
                domain=Domain(classes="test xxx"),
            )
            miao.save()

        if run_options == "test-train":
            # only for test purposes
            print(load(0))
            print(load(1))
    elif run_options == "list":
        all_records = list_all()
        if all_records:
            print(all_records)
            print(f"We have {len(all_records)} TreePoints in the database.")
        else:
            print("[red]No records")
    elif run_options == "explain":
        pass
        # this should allow the upload of a new data point
        # and explain it
        # miao = explain(Path"something something")
        # print(miao)

"""
con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
uuu = cur.execute(f"SELECT * FROM sqlite_master").fetchall()
print(f"metadata:\n{uuu}")
"""
