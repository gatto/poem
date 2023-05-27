import io
import sqlite3
from pathlib import Path

import numpy as np
from attrs import Factory, asdict, define, field, make_class, validators
from rich import print

data_table_structure = ("id int", "a array")

data_path = Path("./data")
data_table_path = data_path / "tutorial.db"


def load(id: int):
    con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    res = cur.execute("SELECT * FROM data")
    # print([element[0] for element in res.description]) # this gives table column names
    res = res.fetchall()
    con.close()
    return res


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
    encoded: Latent
    bb: Blackbox

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
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter("array", _convert_array)

    _delete_create_table()
    mu = np.array(((4, 5), (1, 2)))
    miao = TrainPoint(10, mu, mu, True)
    miao.save()

    print("[red]Values:[/]")
    for i, value in enumerate(load(0)[0]):
        print(f"{i}: {value} {type(value)}")

"""
con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
uuu = cur.execute(f"SELECT * FROM sqlite_master").fetchall()
print(f"metadata:\n{uuu}")
"""
