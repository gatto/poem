import io
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
from attrs import define, field, validators
from mnist import get_data, run_explain
from rich import print
from rich.console import Console
from rich.table import Table
import sklearn
import sklearn_json as skljson

data_table_structure = (
    "id int",
    "a array",
    "latent array",
    "DTpredicted int",
    "DTmodel dictionary",
    "DTfidelity float",
    "BBpredicted int",
    "classes str",
)

data_path = Path("./data")
data_table_path = data_path / "treepoints.db"


@define
class Domain:
    classes: list[str]


@define
class LatentDT:
    predicted_class: int  # index of classes, refers to Domain.classes
    model: sklearn.tree._classes.DecisionTreeClassifier
    fidelity: float
    model_json: dict = field(init=False)

    @model_json.default
    def _model_json_default(self):
        hi = skljson.to_dict(self.model)
        print("heeeeeeeeeeeeey", type(hi), hi)
        return hi


@define
class Latent:
    a: np.ndarray = field(
        validator=validators.instance_of(np.ndarray),
        repr=lambda value: f"{type(value)}",
    )  # record in latent space
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
            self.blackbox.predicted_class,
            self.domain.classes,
        )
        cur.execute(f"INSERT INTO data VALUES {_data_table_structure_query()}", data)
        con.commit()
        con.close()


@define
class Explainer:
    """
    this is what oab.py returns when you ask an explanation
    """


def list_all() -> list[int]:
    """
    Returns a tuple of TreePoint ids that are in the db.
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
    Loads a set of TrainPoint if you pass an id: collection
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
                ),
                blackbox=Blackbox(predicted_class=row["BBpredicted"]),
                domain=Domain(classes=row["classes"]),
            )
    else:
        raise ValueError(f"id was not an int: {id}")


def explain(my_path: Path) -> Explainer:
    pass


def _data_table_structure_query() -> str:
    """
    creates the table structure query for data INSERT
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
sqlite3.register_adapter(dict, lambda d: json.dumps(d).encode('utf8'))
sqlite3.register_converter("dictionary", lambda d: json.loads(d.decode('utf8')))


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
            X_tree = X_tree[:2]
            Y_tree = Y_tree[:2]

        for i, point in enumerate(X_tree):
            try:
                with open(
                    data_path / f"aemodels/mnist/aae/explanation/{i}.pickle", "rb"
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
