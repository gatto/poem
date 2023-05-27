import io
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from attrs import define, field, validators
from rich import print
from rich.console import Console
from rich.table import Table

data_table_structure = ("id int", "a array")

data_path = Path("./data")
data_table_path = data_path / "tutorial.db"


def load(id: int):
    """
    Loads a TrainPoint if you pass an id:int
    Loads a set of TrainPoint if you pass an id: collection
    """
    con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
    #  asks the connection to return Row objects instead of tuples
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    if isinstance(id, int):
        row = cur.execute("SELECT * FROM data WHERE id = ?", (id,))
        # print([element[0] for element in res.description]) # this gives table column names
        row = row.fetchall()
        con.close()
        assert len(row) == 1
        row = row[0]  # there is only one row anyway
        assert id == row["id"]

        # TODO: can i automate this based on the db schema?
        return TrainPoint(id=id, a=row["a"])


@define
class Latent:
    a: np.ndarray  # record in latent space
    space: bool


@define
class Blackbox:
    classes: list[str]
    true_class: str  # or int? index of classes
    predicted_class: str  # or int? index of classes


@define
class TrainPoint:
    """
    TrainPoint.a is the original array in real space
    """

    id: int = field(validator=validators.instance_of(int))
    a: np.ndarray = field(validator=validators.instance_of(np.ndarray))
    # encoded: Latent
    # bb: Blackbox

    def save(self):
        con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()

        data = (self.id, self.a)
        cur.execute(f"INSERT INTO data VALUES {_data_table_structure_query()}", data)
        con.commit()
        con.close()


def _data_table_structure_query() -> str:
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


def _delete_create_table():
    data_table_path.unlink(missing_ok=True)

    data_table_string = "("
    for column in data_table_structure:
        data_table_string = f"{data_table_string}{column}, "
    data_table_string = f"{data_table_string[:-2]})"
    # must be smth like "CREATE TABLE data(id, original array)"

    con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute(f"CREATE TABLE data{data_table_string}")
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


if __name__ == "__main__":
    console = Console()
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter("array", _convert_array)

    # delete create table and commit to db a trainpoint
    _delete_create_table()
    mu = np.array(((4, 5), (1, 2)))
    miao = TrainPoint(10, mu)
    miao.save()

    # visualization
    table = Table(title="TrainPoint schema")
    table.add_column("attribute", style="cyan", no_wrap=True)
    table.add_column("dType", style="magenta")
    for attribute in [
        a
        for a in dir(load(10))
        if not a.startswith("__") and not callable(getattr(load(10), a))
    ]:
        table.add_row(f"{attribute}", f"{type(attribute)}")
    console.print(table)
    print(f"[red]Example:[/] {load(10)} {type(load(10))}")

    # understanding the explanation file

    with open("./data/aemodels/mnist/aae/explanation/156.pickle", "rb") as f:
        element = pickle.load(f)

    print(element)
    print(type(element))

"""
con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
uuu = cur.execute(f"SELECT * FROM sqlite_master").fetchall()
print(f"metadata:\n{uuu}")
"""
