import copy
import io
import json
import logging
import pickle
import random
import sqlite3
import sys
from collections import UserList
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

## CODE EXECUTED BEFORE LIBRARY DEFINITION
classes_mnist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
example_record_mnist = np.zeros((28, 28, 3))
data_table_structure = (
    "id int",
    "a array",
    "latent array",
    "margins array",
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
geq = {">=", ">"}


# TODO: class Condition anziché Rule mantengo l'attributo "is_continuous"
# e class Rule diventa l'intero insieme di predicate da rispettare e/o falsificare
# TODO: rendere eps / 10 * numero random
# in modo da poter avere multipli controesemplari diversi
# TODO: eps non può essere numero statico ma deve confrontarsi con varianza della distribuzione tree_set
# TODO: trovare un modo più semplice possibile per la generazione dei prototipi positivi accantonando
# il metodo di abele per la generazione.
# ad esempio: ho un array [1, 2, 5, 3] ci aggiungo un epsilon nella direzione delle regole positive (rispettandole quindi)


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
    """

    feature: int
    operator: str
    value: float
    # is_continuous: bool TODO: add this
    # TODO: make target_class an index of Domain.classes again?
    # remember to correct the rule/counterrules extraction in LatentDT
    target_class: str

    def respects_rule(self, my_value) -> bool:
        match self.operator:
            case ">":
                return True if my_value > self.value else False
            case ">=":
                return True if my_value >= self.value else False
            case "<":
                return True if my_value < self.value else False
            case "<=":
                return True if my_value <= self.value else False


def _converter_complexrule(rules):
    if type(rules) == list:
        my_results = {}
        for rule in rules:
            try:
                my_results[rule.feature].append(rule)
            except KeyError:
                my_results[rule.feature] = [rule]
        return my_results


@define
class ComplexRule:
    """
    Has
    .rules - dict of Rule.
    Key of the item is the feature number, in the latent space,
    value of the dict is a list with all Rule objects operating on that feature.
    """

    rules = field(converter=_converter_complexrule)


@define
class AAE:
    """
    Methods:
    .encode(Point)
    .decode(Point)
    .discriminate(Point)
    """

    model = field(init=False, repr=lambda value: f"{type(value)}")

    @model.default
    def _model_default(self):
        mtda = get_dataset_metadata()

        ae: abele.adversarial.AdversarialAutoencoderMnist = get_autoencoder(
            np.expand_dims(example_record_mnist, axis=0),
            mtda["ae_name"],
            mtda["dataset"],
            mtda["path_aemodels"],
        )
        ae.load_model()
        return ae

    def discriminate(self, point) -> float:
        """
        pass a Point with latent.a
        returns a probability:float
        """
        return self.model.discriminate(np.expand_dims(point.latent.a, axis=0))[0][0]

    def encode(self, point):
        return self.model.encode(np.expand_dims(point.a, axis=0))[0]

    def decode(self, point):
        return self.model.decode(np.expand_dims(point.latent.a, axis=0))[0]


@define
class Domain:
    classes: list[str] = field()
    aae: AAE = field(init=False, default=AAE())


@define
class LatentDT:
    predicted_class: str  # TODO: do i need to validate this against Domain.classes?
    model: sklearn.tree._classes.DecisionTreeClassifier
    fidelity: float
    # TODO: set s_rules and s_counterrules to repr=False
    s_rules: str = field()
    s_counterrules: str = field()
    model_json: dict = field(
        init=False,
        repr=lambda value: f"{type(value)}",
    )
    rules: ComplexRule = field(init=False)
    counterrules: list[Rule] = field(init=False)

    @model_json.default
    def _model_json_default(self):
        return skljson.to_dict(self.model)

    @rules.default
    def _rules_default(self):
        """
        This converts s_rules:str to rules:ComplexRule

        REMEMBER THAT positive rules mean you're getting
        the `target_class` by 'applying' **ALL** the positive rules.

        THIS IS DIFFERENT THAN COUNTERRULES where you get the target_class
        by falsifying only one of the counterrules.
        """
        results = []
        # str.maketrans's third argument indicates characters to remove with str.translate(•)
        all_rules = self.s_rules.translate(str.maketrans("", "", "{} "))
        if all_rules:
            all_rules = all_rules.split("-->")

            target_class = all_rules[1][all_rules[1].find(":") + 1 :]

            all_rules = all_rules[0].split(",")
            for rule in all_rules:
                for operator in operators:
                    if operator in rule:
                        rule = rule.split(operator)
                        break
                results.append(
                    Rule(
                        feature=int(rule[0]),
                        operator=operator,
                        value=float(rule[1]),
                        target_class=target_class,
                    )
                )
        return ComplexRule(results)

    @counterrules.default
    def _counterrules_default(self):
        """
        This converts s_counterrules:str to counterrules:list[Rule]
        """
        results = []
        # str.maketrans's third argument indicates characters to remove with str.translate(•)
        all_rules = self.s_counterrules.translate(str.maketrans("", "", "{} "))

        if all_rules:
            all_rules = all_rules.split(",")
            for my_rule in all_rules:
                parts = my_rule.split("-->")
                parts[1] = parts[1][parts[1].find(":") + 1 :]
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
    margins: np.ndarray | None
    # space: bool

    # TODO: a validator for self.margins that checks that len(margins) == len(a)

    def __contains__(self, test_point) -> bool:
        """
        ** used for poliedro check **
        returns True if test_point:TestPoint is in the margins of self (which will be a TreePoint)
        False otherwise
        """

        for i, boundary in enumerate(self.margins):
            if not (min(boundary) < test_point.latent.a[i] < max(boundary)):
                # if the feature in test is outside of the boundaries, return bad
                return False
        return True


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
            self.latent.margins,
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
class ImageExplanation:
    """
    The main purpose of this is to automate the decoding of the latent.a:np.ndarray
    into an a:np.ndarray in the real space
    """

    latent: Latent
    blackbox: Blackbox  # TODO: insert predicted class
    a: np.ndarray = field(
        init=False,
        repr=lambda value: f"{type(value)}",
    )

    @a.default
    def _a_default(self):
        """
        I generated a latent.a as either an exemplar or a counterexemplar,
        so now I have to decode it to get the real-space representation of the image
        """
        return my_domain.aae.decode(self)


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
        return Latent(a=my_domain.aae.encode(self), margins=None)

    def marginal_apply(self, rule: Rule, eps=0.01) -> ImageExplanation:
        """
        **Used for counterfactual image generation**

        is used to apply marginally a Rule on a new TestPoint object,
        Returns an ImageExplanation

        e.g. if the rule is 2 > 5.0
        returns a TestPoint with Latent.a[2] == 5.0 + eps
        (regardless of what feature 2's value was in the original TestPoint)

        TODO: eps possibly belonging to Domain? Must calculate it feature
        by feature or possible to have one eps for entire domain?
        """

        debug_results = ""
        # cycles UP TO 40 times to get one point passing the discriminator
        for i in range(40):
            value_to_overwrite = (
                rule.value + eps if rule.operator in geq else rule.value - eps
            )

            # THIS IS THE IMAGEEXPLANATION GENERATION
            new_point = ImageExplanation(
                latent=Latent(a=copy.deepcopy(self.latent.a), margins=None),
                blackbox=None,
            )
            new_point.latent.a[rule.feature] = value_to_overwrite

            # static set discriminator probability at 0.5
            # passes discriminator? Return it immediately.
            # No? start again with entire point generation
            if my_domain.aae.discriminate(new_point) > 0.5:
                if debug_results:
                    logging.debug(debug_results)
                return new_point
            else:
                debug_results = (
                    f"{debug_results} {my_domain.aae.discriminate(new_point)}"
                )
        # we arrive here if we didn't get a valid point after 40 tries
        raise Exception(
            f"apparently we had lots of trouble with .marginal_apply() on this point:\n{self}"
        )

    def perturb(
        self, complexrule: ComplexRule, eps=0.01, old_method=False
    ) -> ImageExplanation:
        """
        **Used for factual image generation**

        returns one TestPoint object with the entire ComplexRule respected,
        but still varying the features by **at least** some margin eps
        """

        debug_results = ""
        # cycles UP TO 40 times to get one point passing the discriminator
        for i in range(40):
            my_generated_record = []
            for feature_id in range(self.latent.a.shape[0]):
                # this is the perturbation step, feature by feature
                generated = False
                failures_counter = 0
                while not generated:
                    # OLD_METHOD:
                    # take value of feature_id in testpoint
                    # perturb it with random.uniform gives variations on the eps value,
                    # random.randrange returns -1 or 1 randomly so the effect
                    # is to either subtract or add eps to value
                    # NEW METHOD:
                    # generate any random value out of a gaussian. We rely solely on
                    # the neighbour-distance ranking to get a link with our original testpoint
                    if old_method:
                        generated_value = self.latent.a[
                            feature_id
                        ] + eps * random.uniform(0.1, 10) * random.randrange(-1, 1, 2)
                    else:
                        generated_value = random.gauss(mu=0.0, sigma=1.0)

                    # validate it according to ComplexRule
                    rules_satisfied = 0
                    for rule in complexrule.rules[feature_id]:
                        if rule.respects_rule(generated_value):
                            rules_satisfied += 1
                        else:
                            pass
                    if rules_satisfied == len(complexrule.rules[feature_id]):
                        generated = True
                    else:
                        failures_counter += 1
                if failures_counter > 0:
                    logging.debug(
                        f"(debug) {failures_counter} failures for feature {feature_id}"
                    )

                my_generated_record.append(generated_value)

            new_point = ImageExplanation(
                latent=Latent(a=np.asarray(my_generated_record), margins=None),
                blackbox=None,
            )
            # static set discriminator probability at 0.5
            # passes discriminator? Return it immediately.
            # No? start again with entire point generation
            if my_domain.aae.discriminate(new_point) > 0.5:
                if debug_results:
                    logging.debug(debug_results)
                return new_point
            else:
                debug_results = (
                    f"{debug_results} {my_domain.aae.discriminate(new_point)}"
                )
        # we arrive here if we didn't get a valid point after 40 tries
        raise Exception(
            f"apparently we had lots of trouble perturbing this point:\n{self}"
        )

    @classmethod
    def generate_test(cls):
        """
        can use TestPoint.generate_test() to get a TestPoint usable for testing
        (it's the point with id=0 in the sql db)
        """
        my_point = load(0)
        return cls(
            a=my_point.a,
            blackbox=Blackbox(predicted_class=my_point.blackbox.predicted_class),
            domain=Domain(classes=my_point.domain.classes),
        )


def ranking_knn(
    target: TestPoint, my_points: list[ImageExplanation]
) -> list[tuple[float, int]]:
    """
    outputs a list of ImageExplanation
    in ascending order of distance from target
    closest point is index=0, farthest point is index=len(my_points)
    """

    neigh = NearestNeighbors(n_neighbors=len(my_points))
    neigh.fit([x.latent.a for x in my_points])
    results = neigh.kneighbors([target.latent.a])
    # results: tuple(np.ndarray, np.ndarray)
    # results[0].shape = (1, n_neighbors) are the distances
    # results[1].shape = (1, n_neighbors) are the indexes
    results = zip(results[0][0], results[1][0])
    results = list(sorted(results))
    return results


@define
class Explainer:
    """
    this is what oab.py returns when you ask an explanation

    testpoint: an input point to explain
    howmany:int how many prototypes (positive exemplars) to generate. The number of counterfactuals
        generated is always == the number of counterrules.
    save: whether to save generated exemplars to data_path (for testing purposes)
    target: the TreePoint most similar to testpoint
    """

    testpoint: TestPoint
    howmany: int = field(default=3)
    save: bool = field(default=False)
    target: TreePoint = field(init=False)
    counterfactuals: list[ImageExplanation] = field(init=False)
    eps_factuals: list[ImageExplanation] = field(init=False)
    factuals: list[ImageExplanation] = field(init=False)

    @howmany.validator
    def _howmany_validator(self, attribute, value):
        if value < 0:
            raise ValueError(f"{attribute} should be positive, instead it was {value}")

    @target.default
    def _target_default(self) -> TreePoint:
        return knn(self.testpoint)

    @counterfactuals.default
    def _counterfactuals_default(self):
        logging.info(f"Doing counterfactuals with target point id={self.target.id}")
        # for now, set epsilon statically. TODO: do a hypoteses test for an epsilon
        # statistically *slightly* bigger than zero
        results = []

        for i, rule in enumerate(self.target.latentdt.counterrules):
            point: ImageExplanation = self.testpoint.marginal_apply(rule)

            if self.save:
                plt.imshow(point.a.astype("uint8"), cmap="gray")
                plt.title(
                    f"counterfactual - black box predicted class: xxx"
                )  # TODO: substitute xxx -> point.blackbox.predicted_class
                plt.savefig(data_path / f"counter_{i}.png", dpi=150)

            results.append(point)

        logging.info(f"I made {len(results)} counterfactuals.")
        return results

    @eps_factuals.default
    def _eps_factuals_default(self):
        logging.info(f"Doing epsilon-factuals with target point id={self.target.id}")
        results = []

        for factual in range(self.howmany):
            point: ImageExplanation = self.testpoint.perturb(
                self.target.latentdt.rules, old_method=True
            )
            results.append(point)

        if self.save:
            for i, point in enumerate(results):
                plt.imshow(point.a.astype("uint8"), cmap="gray")
                plt.title(
                    f"epsilon factual - black box predicted class: xxx"
                )  # TODO: substitute xxx -> point.blackbox.predicted_class
                plt.savefig(data_path / f"fact_{i}.png", dpi=150)

        logging.info(f"I made {len(results)} epsilon-factuals.")
        return results

    @factuals.default
    def factuals_default(self):
        logging.info(f"Doing factuals with target point id={self.target.id}")
        results = []

        for factual in range(self.howmany * 10):
            point: ImageExplanation = self.testpoint.perturb(self.target.latentdt.rules)
            results.append(point)

        # take the last how_many points, last because I'd like the farthest points
        ranking = ranking_knn(self.target, results)[-self.howmany :]

        # take the index in the tuple(distance, index)
        indexes_to_take = [x[1] for x in ranking]
        results = [results[i] for i in indexes_to_take]

        if self.save:
            for i, point in enumerate(results):
                plt.imshow(point.a.astype("uint8"), cmap="gray")
                plt.title(
                    f"factual - black box predicted class: xxx"
                )  # TODO: substitute xxx -> point.blackbox.predicted_class
                plt.savefig(data_path / f"new_fact_{i}.png", dpi=150)

        logging.info(f"I made {len(results)} factuals.")
        return results

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
    def from_array(cls, my_array: np.ndarray):
        """
        This is the main method that should be exposed externally.
        intended usage:

        from oab import Explainer
        explanation = Explainer.from_file(<path_to_image>)
        """
        return cls()

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

    points: list[TreePoint] = load_all()
    latent_arrays: list[np.ndarray] = [point.latent.a for point in points]
    while True:
        # this while loop's purpose is to continue looking for 1-nn sample points
        # if the first sample point result `points[index]` is discarded because TestPoint
        # is not in the sampled point's margins
        # TODO: ensure a stopping condition exists for this while loop
        neigh = NearestNeighbors(n_neighbors=1)

        # I train this on the np.ndarray latent repr of the points,
        neigh.fit(latent_arrays)

        fitted_model = neigh.kneighbors([point.latent.a])
        # if I need the distance it's here…
        distance: np.float64 = fitted_model[0][0][0]
        index: np.int64 = fitted_model[1][0][0]

        # check the margins of the latent space (poliedro check)
        if point in points[index].latent:
            # if it's in the margins
            break
        else:
            # otherwise, pop that point (don't need it)
            # and start again
            print("popped a point")
            points.pop(index)
            latent_arrays.pop(index)

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
                latent=Latent(
                    a=row["latent"],
                    margins=row["margins"],
                ),
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

## CODE EXECUTED AFTER LIBRARY DEFINITION
console = Console()
sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("array", _convert_array)
sqlite3.register_adapter(dict, lambda d: json.dumps(d).encode("utf8"))
sqlite3.register_converter("dictionary", lambda d: json.loads(d.decode("utf8")))
my_domain = Domain(classes_mnist)


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
            (testing) delete-all, run-tests, test-train <how many to load>,\n
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
            X_tree = X_tree[: sys.argv[2]]
            Y_tree = Y_tree[: sys.argv[2]]

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
                latent=Latent(
                    a=tosave["limg"],
                    margins=tosave["neigh_bounding_box"].transpose(),
                ),
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
