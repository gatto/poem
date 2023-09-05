import copy
import io
import json
import logging
import pickle
import random
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn_json as skljson
from attrs import define, field, validators
from mnist import (
    get_autoencoder,
    get_black_box,
    get_data,
    get_dataset_metadata,
    run_explain,
)
from rich import print
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.neighbors import NearestNeighbors

## CODE EXECUTED BEFORE LIBRARY DEFINITION
"""
Places to update when adding something to the sql db:
- data_table_structure
- TreePoint.save()
- .load()
- in main, in `elif run_option in ("train", "test-train"):`
"""
data_table_structure = (
    "id int",
    "a array",
    "latent array",
    "margins array",
    "DTpredicted str",
    "DTmodel dictionary",
    "DTfidelity float",
    "srule str",
    "scounterrules str",
    "BBpredicted str",
    "dataset str",
)

data_path = Path("./data/oab")
operators = [">=", "<=", ">", "<"]
geq = {">=", ">"}


# TODO: eps non può essere numero statico ma deve confrontarsi con varianza della distribuzione tree_set


@define
class Condition:
    """
    structure of a Condition:
    feature:int  the latent feature the condition checks
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
    target_class: str
    # is_continuous: bool TODO: add this

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


def _converter_complexrule(conditions):
    if isinstance(conditions, list):
        my_results = {}
        for condition in conditions:
            try:
                my_results[condition.feature].append(condition)
            except KeyError:
                my_results[condition.feature] = [condition]
        return my_results


@define
class ComplexRule:
    """
    Has
    .conditions - dict of Condition.
    Key of the item is the feature number, in the latent space,
    value of the dict is a list with all Condition objects operating on that feature.
    """

    conditions = field(converter=_converter_complexrule)

    def __contains__(self, point) -> bool:
        """
        this is used like this:
        if Point in my_treepoint.latentdt.rule:
            pass  # Point respects the rule for that my_treepoint
        else:
            pass  # Point does not respect the rule for that my_treepoint

        Point must be either TestPoint or TreePoint or ImageExplanation: anything that has .latent.a
        """

        features_failing = []
        for feature_id in range(point.latent.a.shape[0]):
            # validate it according to ComplexRule
            rules_satisfied = 0
            try:
                for condition in self.conditions[feature_id]:
                    if condition.respects_rule(point.latent.a[feature_id]):
                        rules_satisfied += 1
                if rules_satisfied != len(self.conditions[feature_id]):
                    features_failing.append(feature_id)
            except KeyError:
                # We get here if for a certain feature_id there is no self.conditions[feature_id]
                # If there are no conditions on a feature, then the feature is automatically respecting the rule
                # So we need to do nothing except keep going to the next features
                pass

        if len(features_failing) == 0:
            return True
        else:
            my_message = "Features failing are:\n"
            for feature_id in features_failing:
                my_message = f"{my_message}feature {feature_id}: it's {point.latent.a[feature_id]} and conditions are {self.conditions[feature_id]}\n"
            logging.warning(my_message)
            return False


@define
class Blackbox:
    """
    Usage:
    Blackbox.predict(point) returns a class prediction of type str
    """

    dataset_name: str = field(repr=False, validator=validators.instance_of(str))
    bb_type: str = field(validator=validators.instance_of(str))
    model: dict = field(init=False, repr=lambda value: f"{value.keys()}")

    @model.default
    def _model_default(self) -> dict:
        match self.dataset_name:
            case "mnist":
                use_rgb = False  # g with mnist dataset
                black_box_filename = f"./data/models/mnist_{self.bb_type}"

                results = dict()
                results["predict"], results["predict_proba"] = get_black_box(
                    self.bb_type, black_box_filename, use_rgb
                )
                return results
                # the original usage is: Y_pred = bb_predict(X_test)

            case "fashion":
                use_rgb = False
                black_box_filename = f"./data/models/fashion_{self.bb_type}"

                results = dict()
                results["predict"], results["predict_proba"] = get_black_box(
                    self.bb_type, black_box_filename, use_rgb
                )
                return results
                # the original usage is: Y_pred = bb_predict(X_test)
            case "custom":
                raise NotImplementedError

    def predict(self, a):
        return self.model["predict"](np.expand_dims(a, axis=0))[0]


@define
class BlackboxPD:
    predicted_class: str = field(converter=str)


@define
class AE:
    """
    Methods:
    .encode(Point)
    .decode(Point)
    .discriminate(Point)
    """

    dataset_name: str = field(repr=False, validator=validators.instance_of(str))
    metadata: dict = field(repr=False)
    model = field(init=False, repr=lambda value: f"{type(value)}")

    @model.default
    def _model_default(self):
        # ae: abele.adversarial.AdversarialAutoencoderMnist
        ae = get_autoencoder(
            np.expand_dims(np.zeros(self.metadata["shape"]), axis=0),
            self.metadata["ae_name"],
            self.metadata["dataset"],
            self.metadata["path_aemodels"],
        )
        ae.load_model()
        return ae

    def discriminate(self, point) -> float:
        """
        pass a Point containing latent.a
        returns a probability:float
        """
        return self.model.discriminate(np.expand_dims(point.latent.a, axis=0))[0][0]

    def encode(self, point):
        return self.model.encode(np.expand_dims(point.a, axis=0))[0]

    def decode(self, point):
        return self.model.decode(np.expand_dims(point.latent.a, axis=0))[0]


@define
class Domain:
    """
    EXCEPT FOR A "CUSTOM" DATASET, WHICH WILL NOT BE IMPLEMENTED,

    Automatically knows everything it needs to know just by constructing it like:
    Domain(dataset_name="mnist | xxx", bb_type="RF | xxx")
    Each dataset has possible bb_types

    Domain has metadata set, clases set, fetches ae and blackbox from files
    """

    dataset_name: str = field()
    bb_type: str = field()
    metadata: dict = field()
    classes: list[str] = field(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(str),
            iterable_validator=validators.instance_of(list),
        )
    )
    ae: AE = field()
    blackbox: Blackbox = field()
    explanation_base: list = field(init=False, default=None)

    @dataset_name.validator
    def _dataset_validator(self, attribute, value):
        possible_datasets = {"mnist", "fashion", "custom"}
        if value not in possible_datasets:
            raise ValueError(f"Dataset {value} not implemented.")

    @bb_type.validator
    def _bb_string_validator(self, attribute, value):
        match self.dataset_name:
            case "mnist":
                possible_bb = {"RF", "DNN"}
            case "fashion":
                possible_bb = {"RF", "DNN"}
            case "custom":
                raise NotImplementedError
            case _:
                raise ValueError
        if value not in possible_bb:
            raise ValueError(
                f"Blackbox type bb_type={value} not implemented. Possible are: {possible_bb}"
            )

    @metadata.default
    def _metadata_default(self):
        results = dict()
        match self.dataset_name:
            case "mnist":
                results["ae_name"] = "aae"
                match self.bb_type:
                    case "RF":
                        pass
                    case "DNN":
                        pass
                    case _:
                        raise ValueError
                results["dataset"] = "mnist"
                results["shape"] = (28, 28, 3)
                results[
                    "path_aemodels"
                ] = f"./data/aemodels/{results['dataset']}/{results['ae_name']}/"
            case "fashion":
                results["ae_name"] = "aae"
                match self.bb_type:
                    case "RF":
                        pass
                    case "DNN":
                        pass
                    case _:
                        raise ValueError
                results["dataset"] = "fashion"
                results["shape"] = (28, 28, 3)
                results[
                    "path_aemodels"
                ] = f"./data/aemodels/{results['dataset']}/{results['ae_name']}/"
            case "custom":
                raise NotImplementedError
            case _:
                raise ValueError
        return results

    @classes.default
    def _classes_default(self):
        match self.dataset_name:
            case "mnist":
                return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            case "fashion":
                return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            case "custom":
                raise NotImplementedError
            case _:
                raise ValueError

    @ae.default
    def _ae_default(self):
        """
        this works already
        """
        match self.dataset_name:
            case "mnist":
                pass
            case "fashion":
                pass
            case "custom":
                raise NotImplementedError
            case _:
                raise ValueError
        return AE(dataset_name=self.dataset_name, metadata=self.metadata)

    @blackbox.default
    def _blackbox_default(self):
        """
        this works already too
        """
        match self.dataset_name:
            case "mnist":
                pass
            case "fashion":
                pass
            case "custom":
                raise NotImplementedError
            case _:
                raise ValueError
        return Blackbox(dataset_name=self.dataset_name, bb_type=self.bb_type)

    def load(self):
        logging.info(
            f"start loading the explanation base for {self.dataset_name}, {self.bb_type}"
        )
        print(
            f"start loading the explanation base for {self.dataset_name}, {self.bb_type}"
        )
        self.explanation_base = load_all(self)
        logging.info(f"loaded {len(self.explanation_base)} in explanation base")
        print(f"loaded {len(self.explanation_base)} in explanation base")


@define
class LatentDT:
    """
    All attributes non-optional except .model_json, .rule and .counterrules
    """

    predicted_class: str = field(converter=str)
    model: sklearn.tree._classes.DecisionTreeClassifier
    fidelity: float
    s_rules: str = field()
    s_counterrules: str = field()
    model_json: dict = field(init=False, repr=False)
    rule: ComplexRule = field(init=False)
    counterrules: list[Condition] = field(init=False)

    @model_json.default
    def _model_json_default(self):
        return skljson.to_dict(self.model)

    # TODO: remember to correct the rule/counterrules extraction in LatentDT: counterrules may be both simple and ComplexRule
    @rule.default
    def _rules_default(self):
        """
        This converts s_rules:str to rule:ComplexRule

        REMEMBER THAT positive conditions mean you're getting
        the `target_class` by 'applying' **ALL** the positive conditions.

        THIS IS DIFFERENT THAN COUNTERRULES where you can get the target_class
        by falsifying only one of the conditions (as long as its a one-condition counterrule).
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
                    Condition(
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
        This converts s_counterrules:str to counterrules:list[Condition]
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
                    Condition(
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
    margins: the margins of the neighborhood that was generated around a, if any. Is None if I generate myself the latent.a instead
    of getting it from Abele: in this case there never was a generated neighborhood for the point.

    Construction: Latent(a:np.ndarray, margins=np.ndarray | None, )
    """

    a: np.ndarray = field(
        validator=validators.instance_of(np.ndarray),
        repr=lambda value: str(value) if len(value) < 10 else str(type(value)),
    )
    margins: np.ndarray | None = field(default=None)

    @margins.validator
    def _margins_validator(self, attribute, value):
        """
        This checks that len(margins) == len(a). This must be true or there is a coherency error somewhere.
        """
        if value is not None:  # if margins is None, doesn't matter
            if len(value) != len(self.a):
                raise ValueError(
                    f"The len of {attribute}:{type(value)} == {value}\nis not equal to the len of the array a."
                )

    def __contains__(self, test_point) -> bool:
        """
        ** used for poliedro check **
        returns True if test_point:TestPoint is in the margins of self (which will be a TreePoint)
        False otherwise
        """
        # return True  # temporary fix because the poliedro boundaries are all wrong
        for i, boundary in enumerate(self.margins):
            if not (min(boundary) < test_point.latent.a[i] < max(boundary)):
                # if the feature in test is outside of the boundaries, return bad
                return False
        return True


@define
class TreePoint:
    """
    TreePoint.id is the index of the record in the passed dataset
    TreePoint.a is the original array in real space
    TreePoint.domain is the Domain object representing the domain of the problem. See Domain for construction
    TreePoint.latent is the point's latent space containing its repr and its margins. See Latent

    Construction: TreePoint(id:int, a:np.ndarray, domain=Domain(•), latent=Latent(•), latentdt=LatentDT)
    """

    id: int = field(validator=validators.instance_of(int))
    a: np.ndarray = field(
        validator=validators.instance_of(np.ndarray),
        repr=lambda value: f"{type(value)}",
    )
    domain: Domain = field(repr=lambda value: f"{type(value)}")
    latent: Latent
    latentdt: LatentDT = field()
    blackboxpd: BlackboxPD = field(init=False)
    # true_class: str  # TODO: do i need to validate this against Domain.classes?

    @latentdt.validator
    def _latentdt_validator(self, attribute, value):
        if value.predicted_class not in self.domain.classes:
            raise ValueError(
                f"The {value.predicted_class} predicted_class of {attribute} is not in the domain. Its type is {type(value.predicted_class)}"
            )

    @blackboxpd.default
    def _blackboxpd_default(self):
        return BlackboxPD(predicted_class=self.domain.blackbox.predict(self.a))

    @blackboxpd.validator
    def _blackboxpd_validator(self, attribute, value):
        if value.predicted_class not in self.domain.classes:
            raise ValueError(
                f"The {value.predicted_class} predicted_class of {attribute} is not in the domain. Its type is {type(value.predicted_class)}"
            )

    def save(self):
        with Connection(self.domain.dataset_name, self.domain.bb_type) as con:
            cur = con.cursor()

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
                self.blackboxpd.predicted_class,
                self.domain.dataset_name,
            )
            cur.execute(
                f"INSERT INTO data VALUES {_data_table_structure_query()}", data
            )
            con.commit()

    def keys(self):
        return [x for x in dir(self) if x[:1] != "_"]


@define
class ImageExplanation:
    """
    The main purpose of this is to automate the decoding of the latent.a:np.ndarray
    into an a:np.ndarray in the real space
    Also, to get the blackbox prediction for this

    Construction: ImageExplanation(latent=Latent(•))
    """

    latent: Latent
    domain: Domain
    a: np.ndarray = field(
        init=False,
        repr=lambda value: f"{type(value)}",
    )
    blackboxpd: BlackboxPD = field(init=False)

    @a.default
    def _a_default(self):
        """
        I generated a latent.a as either an exemplar or a counterexemplar,
        so now I have to decode it to get the real-space representation of the image
        """
        return self.domain.ae.decode(self)

    @blackboxpd.default
    def _blackboxpd_default(self):
        return BlackboxPD(predicted_class=self.domain.blackbox.predict(self.a))


@define
class TestPoint:
    a: np.ndarray = field(
        validator=validators.instance_of(np.ndarray),
        repr=lambda value: f"{type(value)}",
    )
    domain: Domain
    blackboxpd: BlackboxPD = field(init=False)
    latent: Latent = field(init=False)

    @blackboxpd.default
    def _blackboxpd_default(self):
        return BlackboxPD(predicted_class=self.domain.blackbox.predict(self.a))

    @latent.default
    def _latent_default(self):
        """
        encodes the TestPoint.a to build TestPoint.Latent.a.
        Has no margins bc as a test point we did not and will not generate a neighborhood
        """
        return Latent(a=self.domain.ae.encode(self))

    def marginal_apply(
        self, rule: Condition, eps=0.04, more=False
    ) -> ImageExplanation | None:
        """
        **Used for counterfactual image generation**

        is used to apply marginally a Condition on a new TestPoint object,
        Returns an ImageExplanation

        e.g. if the condition is 2 > 5.0
        returns a TestPoint with Latent.a[2] == 5.0 + eps
        (regardless of what feature 2's value was in the original TestPoint)

        If the record is then rejected by discriminator, try again with
        Latent.a[2] == 5.0 + eps * 2
        then Latent.a[2] == 5.0 + eps * 3
        up to 40 tries. If still fails discrim, then fail and return None

        TODO: eps possibly belonging to Domain? Must calculate it feature
        by feature or possible to have one eps for entire domain?
        """

        debug_results = ""
        # cycles UP TO 40 times to get one point passing the discriminator
        for i in range(1, 41):
            # if I'm in the "more" counterfactual generation, I randomly take a multiplicator i
            # for the epsilon eps to get further away from the decision boundary than usual.
            # we're still trying up to 40 times, that doesn't change. Except the i is random
            # instead of being incremental.
            if more:
                i = random.randrange(1, 51)

            # I multiply by i because if the discriminator doesn't accept the record then I
            # start getting further and further from the decision boundary
            # hoping that at some point the value will be accepted by discriminator.
            value_to_overwrite = (
                rule.value + eps * i if rule.operator in geq else rule.value - eps * i
            )

            # THIS IS THE IMAGEEXPLANATION GENERATION
            my_a = copy.deepcopy(self.latent.a)
            my_a[rule.feature] = value_to_overwrite
            new_point = ImageExplanation(latent=Latent(my_a), domain=self.domain)

            # static set discriminator probability at 0.35
            # passes discriminator? Return it immediately.
            # No? start again with entire point generation
            if discr_value := self.domain.ae.discriminate(new_point) >= 0.35:
                if debug_results:
                    logging.warning(debug_results)
                return new_point
            else:
                debug_results = f"{debug_results} {discr_value}"
        # we arrive here if we didn't get a valid point after 40 tries
        if debug_results:
            logging.error(
                f"we would have runtimerrror here in .marginal_apply() with {debug_results}"
            )
        return None

    # TODO: fare notebook con 10 immagini per record

    def perturb(
        self, complexrule: ComplexRule, eps=0.04, old_method=False
    ) -> ImageExplanation | None:
        """
        **Used for factual image generation**

        returns one ImageExplanation object with the entire ComplexRule respected,
        but still varying the features by **at least** some margin eps
        It can return None if we tried 40 times to pass the discriminator and we failed every time
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
                        ] + eps * random.uniform(1, 10) * random.randrange(-1, 2, 2)
                    else:
                        generated_value = random.gauss(mu=0.0, sigma=1.0)

                    # validate it according to ComplexRule
                    rules_satisfied = 0
                    try:
                        # try because there might not be any condition insisting on any specific feature
                        for rule in complexrule.conditions[feature_id]:
                            if rule.respects_rule(generated_value):
                                rules_satisfied += 1
                            else:
                                pass
                        if rules_satisfied == len(complexrule.conditions[feature_id]):
                            generated = True
                        else:
                            failures_counter += 1
                    except KeyError:
                        generated = True
                if failures_counter > 0:
                    logging.warning(
                        f"{failures_counter} failures for feature {feature_id}"
                    )

                my_generated_record.append(generated_value)

            new_point = ImageExplanation(
                latent=Latent(a=np.asarray(my_generated_record)), domain=self.domain
            )
            # static set discriminator probability at 0.35
            # passes discriminator? Return it immediately.
            # No? start again with entire point generation
            if discr_value := self.domain.ae.discriminate(new_point) > 0.35:
                if debug_results:
                    logging.warning(debug_results)
                return new_point
            else:
                debug_results = f"{debug_results} {discr_value}"
        # we arrive here if we didn't get a valid point after 40 tries
        logging.error(
            f"we would have runtimerrror here in .perturb() with {debug_results}"
        )
        pass

    @classmethod
    def generate_test(cls, my_domain):
        """
        can use TestPoint.generate_test() to get a TestPoint usable for testing
        (it's the point with id=0 in the sql db)
        """
        my_point = load(0)
        return cls(a=my_point.a, domain=my_domain)


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
    dataset: name of the standard dataset (supported datasets are in Domain._dataset_validator)
    howmany:int how many prototypes (positive exemplars) to generate. The number of counterfactuals
        generated is always == the number of counterrules.
    save: whether to save generated exemplars to data_path (for testing purposes)
    target: the TreePoint most similar to testpoint

    Offers methods:
    .more_factuals(): generate more factuals, return them, and also extend .factuals with them.
    .from_array(
        a: np.ndarray,
        dataset: str,
        howmany: int = 3,
        save: bool = False
        ): intended entry point to Explainer, explain from array a
    .from_file(): intended entry point to Explainer, explain from file
    """

    testpoint: TestPoint
    howmany: int = field(default=3)
    save: bool = field(default=False)
    target: TreePoint = field(init=False)
    counterfactuals: list[ImageExplanation] = field(init=False)
    # eps_factuals: list[ImageExplanation] = field(init=False)
    factuals: list[ImageExplanation] = field(init=False)

    @howmany.validator
    def _howmany_validator(self, attribute, value):
        if value < 0:
            raise ValueError(f"{attribute} should be positive, instead it was {value}")

    @target.default
    def _target_default(self) -> TreePoint:
        return knn(self.testpoint)

    @counterfactuals.default
    def _counterfactuals_default(self, more=False):
        logging.info(f"Doing counterfactuals with target point id={self.target.id}")
        # for now, set epsilon statically. TODO: do a hypoteses test for an epsilon
        # statistically *slightly* bigger than zero
        results = []

        for i, rule in enumerate(self.target.latentdt.counterrules):
            point: ImageExplanation = self.testpoint.marginal_apply(rule, more=more)

            # it might be that we can't get an image accepted by the
            # discriminator. Therefore might be that point is None
            if point:
                results.append(point)
                if self.save:
                    plt.imshow(point.a.astype("uint8"), cmap="gray")
                    plt.title(
                        f"counterfactual - black box predicted class: {point.blackboxpd.predicted_class}"
                    )
                    plt.savefig(data_path / f"counter_{i}.png", dpi=150)

        logging.info(f"I made {len(results)} counterfactuals.")
        return results

    # @eps_factuals.default
    def _eps_factuals_default(self):
        logging.info(f"Doing epsilon-factuals with target point id={self.target.id}")
        results = []

        for factual in range(self.howmany):
            point: ImageExplanation = self.testpoint.perturb(
                self.target.latentdt.rule, old_method=True
            )
            if point:
                results.append(point)

        if self.save:
            for i, point in enumerate(results):
                plt.imshow(point.a.astype("uint8"), cmap="gray")
                plt.title(
                    f"epsilon factual - black box predicted class: {point.blackboxpd.predicted_class}"
                )
                plt.savefig(data_path / f"fact_{i}.png", dpi=150)

        logging.info(f"I made {len(results)} epsilon-factuals.")
        return results

    @factuals.default
    def _factuals_default(self, closest=False):
        """
        This generates 10 times the number of factuals requested with self.howmany,
        then returns the .howmany **farthest** from the testpoint.

        For debug purposes or if you have specific needs you can set closest=True
        to return the .howmany **closest** from testpoint.
        """
        logging.info(f"Doing factuals with target point id={self.target.id}")
        results = []

        for factual in range(self.howmany * 10):
            point: ImageExplanation = self.testpoint.perturb(self.target.latentdt.rule)
            if point:
                # check if the point is classified same as class of testpoint
                if point.blackboxpd == self.target.blackboxpd:
                    results.append(point)

        logging.info(
            f"there are, after checks, {len(results)} points among which to choose {self.howmany}."
        )

        # take the last how_many points, last because I'd like the farthest points
        if not closest:
            ranking = ranking_knn(self.target, results)[-self.howmany :]
        elif closest:
            ranking = ranking_knn(self.target, results)[: self.howmany]

        # take the index in the tuple(distance, index)
        indexes_to_take = [x[1] for x in ranking]
        results = [results[i] for i in indexes_to_take]

        if self.save:
            for i, point in enumerate(results):
                plt.imshow(point.a.astype("uint8"), cmap="gray")
                plt.title(
                    f"factual - black box predicted class: {point.blackboxpd.predicted_class}"
                )
                plt.savefig(data_path / f"new_fact_{i}.png", dpi=150)

        logging.info(f"I made {len(results)} factuals.")
        return results

    def more_factuals(self) -> list[ImageExplanation]:
        more = self._factuals_default()
        self.factuals = self.factuals.extend(more)
        return more

    def more_counterfactuals(self) -> list[ImageExplanation]:
        more = self._counterfactuals_default(more=True)
        self.counterfactuals = self.counterfactuals.extend(more)
        return more

    def keys(self):
        return [x for x in dir(self) if x[:1] != "_"]

    @classmethod
    def from_array(
        cls, a: np.ndarray, domain: Domain, howmany: int = 3, save: bool = False
    ):
        """
        This is the main method that should be exposed externally.
        intended usage:

        from oab import Explainer
        explanation = Explainer.from_array(a:np.ndarray)
        """

        return cls(
            testpoint=TestPoint(a=a, domain=domain),
            howmany=howmany,
            save=save,
        )

    @classmethod
    def from_file(
        cls, my_path: Path, domain: Domain, howmany: int = 3, save: bool = False
    ):
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

        return cls.from_array("xxx")


def knn(point: TestPoint) -> TreePoint:
    """
    this returns only the closest TreePoint to the inputted point `a`
    (in latent space representation)
    """
    logging.info("start of knn")

    points: list[TreePoint] = point.domain.explanation_base
    logging.info("loaded all")
    latent_arrays: list[np.ndarray] = [point.latent.a for point in points]
    logging.info("done latent_arrays")
    while True:
        if not points:
            raise RuntimeError("We've run out of tree points during knn")
        # this while loop's purpose is to continue looking for 1NN sample points
        # if the first sample point result `points[index]` is discarded because TestPoint
        # is not in the sampled point's margins
        neigh = NearestNeighbors(n_neighbors=1)

        # I train this on the np.ndarray latent repr of the points,
        neigh.fit(latent_arrays)
        logging.info("fitted KNN")

        fitted_model = neigh.kneighbors([point.latent.a])
        # if I need the distance it's here… fitted_model[0][0][0]: np.float64
        index: np.int64 = fitted_model[1][0][0]

        # check the margins of the latent space (poliedro check)
        if point in points[index].latent:
            # if it's in the margins
            logging.info("checked margins")
            pass  # go to next check
        else:
            # otherwise, pop that point (don't need it) and start again
            logging.warning("popped a point bc of margins")
            points.pop(index)
            latent_arrays.pop(index)
            continue  # start over

        # check if bb predicted class match of testpoint and selected treepoint
        if point.blackboxpd == points[index].blackboxpd:
            logging.info("checked class")
            break  # we done
        else:
            # otherwise, pop that point (don't need it) and start again
            logging.warning("popped a point bc of BlackboxPD mismatch")
            points.pop(index)
            latent_arrays.pop(index)
            continue  # start over

    # I return the entire TreePoint though
    return points[index]


def list_all(dataset_name, bb_type) -> list[int]:
    """
    Returns a list of TreePoint ids that are in the db.
    """
    with Connection(dataset_name, bb_type) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        rows = cur.execute("SELECT id FROM data")
        rows = rows.fetchall()
    return sorted([x["id"] for x in rows])


def load(domain: Domain, id: int | set | list | tuple) -> None | TreePoint:
    """
    Loads a TrainPoint if you pass an id:int
    Loads a list of TrainPoints ordered by id if you pass a collection
    """
    if isinstance(id, int):
        with Connection(domain.dataset_name, domain.bb_type) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            row = cur.execute("SELECT * FROM data WHERE id = ?", (id,))
            # print([element[0] for element in res.description]) # this gives table column names
            row = row.fetchall()

        if len(row) == 0:
            return None
        elif len(row) > 1:
            raise Exception(
                f"the id {id} is supposed to be unique but it's not in this database"
            )
        else:
            row = row[0]  # there is only one row anyway
            assert id == row["id"]

            rebuilt_dt = skljson.from_dict(row["DTmodel"])

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
                    s_rules=row["srule"],
                    s_counterrules=row["scounterrules"],
                ),
                domain=domain,
            )  # blackboxpd=BlackboxPD(predicted_class=row["BBpredicted"]),

    elif isinstance(id, set) or isinstance(id, list) or isinstance(id, tuple):
        to_load = sorted([x for x in id])
        results = []
        for i in to_load:
            results.append(load(domain, i))
        return results
    else:
        raise TypeError(f"id was of an unrecognized type: {type(id)}")


def load_all(domain: Domain) -> list[TreePoint]:
    """
    Returns a list of all TreePoints that are in the sql db
    """
    results = []

    for i in list_all(domain.dataset_name, domain.bb_type):
        results.append(load(domain, i))
    return results


def _data_table_structure_query() -> str:
    """
    Creates the table structure query for data INSERT
    """
    my_query = "("
    for column in data_table_structure:
        my_query = f"{my_query}?, "
    return f"{my_query[:-2]})"


@define
class Connection:
    # TODO: insert a check that the table exists using
    # res = cur.execute("SELECT name FROM sqlite_master WHERE name='spam'")
    # res.fetchone() is None
    # return cur
    dataset_name: str = field()
    bb_type: str = field()
    method: str = field(default="sqlite")
    path: Path = field(init=False)
    connection = field(init=False, repr=lambda value: f"{type(value)}", default=None)

    @dataset_name.validator
    def _dataset_name_validator(self, attribute, value):
        possible_datasets = {"mnist", "fashion", "custom"}
        if value not in possible_datasets:
            raise NotImplementedError(f"Dataset {value} not implemented.")

    @bb_type.validator
    def _bb_type_validator(self, attribute, value):
        match self.dataset_name:
            case "mnist":
                possible_bb = {"RF", "DNN"}
            case "fashion":
                possible_bb = {"RF", "DNN"}
            case "custom":
                raise NotImplementedError
            case _:
                raise ValueError
        if value not in possible_bb:
            raise NotImplementedError(
                f"Blackbox type bb_type={value} not implemented. Possible are: {possible_bb}"
            )

    @method.validator
    def _method_validator(self, attribute, value):
        implemented_protocols = {"sqlite"}
        if value not in implemented_protocols:
            raise ValueError(
                f"The implemented sql protocols right now are {implemented_protocols}"
            )

    @path.default
    def _path_default(self) -> Path:
        data_path = Path("./data/oab")
        match self.dataset_name:
            case "mnist":
                match self.bb_type:
                    case "RF":
                        data_table_path = data_path / "mnist.db"
                    case "DNN":
                        data_table_path = data_path / "mnist-dnn.db"
                    case _:
                        raise NotImplementedError
            case "fashion":
                match self.bb_type:
                    case "RF":
                        data_table_path = data_path / "fashion-rf.db"
                    case "DNN":
                        data_table_path = data_path / "fashion-dnn.db"
                    case _:
                        raise NotImplementedError
            case "custom":
                raise NotImplementedError
            case _:
                raise ValueError
        return data_table_path

    def __enter__(self):
        self.connection = sqlite3.connect(
            self.path, detect_types=sqlite3.PARSE_DECLTYPES
        )
        return self.connection

    def __exit__(self, *args):
        self.connection.close()


def _delete_create_table(dataset_name, bb_type) -> None:
    Connection(dataset_name, bb_type).path.unlink(missing_ok=True)

    data_table_string = "("
    for column in data_table_structure:
        data_table_string = f"{data_table_string}{column}, "
    data_table_string = f"{data_table_string[:-2]})"

    with Connection(dataset_name, bb_type) as con:
        cur = con.cursor()

        # must be smth like "CREATE TABLE data(id, original array)"
        cur.execute(f"CREATE TABLE data{data_table_string}")


def _adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


## CODE EXECUTED AFTER LIBRARY DEFINITION
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
        dataset = sys.argv[1]
        bb_type = sys.argv[2].upper()
        run_option = sys.argv[3]
    except IndexError:
        raise Exception(
            """possible runtime arguments are:
            (dataset) mnist
            (blackbox type) rf | dnn
            and then:
            (testing) delete-all, run-tests, test-train <how many to load>,
            (production) train, list
            
            examples:
            python oab.py mnist rf delete-all
            python oab.py mnist rf test-train 8000

            to use this to explain a record, must use it in a python script (refer to documentation)"""
        )

    match dataset:
        case "mnist":
            pass
        case "fashion":
            pass
        case _:
            raise NotImplementedError

    if run_option == "delete-all":
        # delete create table
        _delete_create_table(dataset, bb_type)
        print(f"done {dataset} {bb_type} {run_option}")
    elif run_option == "run-tests":
        raise NotImplementedError
        miao = load(0)
        miao.id = 12155
        miao.save()

        # visualization
        table = Table(title="TrainPoint schema")
        table.add_column("attribute", style="cyan", no_wrap=True)
        table.add_column("dType", style="magenta")
        for attribute in [
            a
            for a in dir(load(12155))
            if not a.startswith("__") and not callable(getattr(load(12155), a))
        ]:
            table.add_row(f"{attribute}", f"{type(attribute)}")
        console.print(table)
        print("[red]Example:[/]")
        print(load(12155))
    elif run_option in ("train", "test-train"):
        # test-train should be used until real train run
        my_domain = Domain(dataset, bb_type)
        (
            (X_train, Y_train),
            (X_test, Y_test),
            (X_tree, Y_tree),
        ) = get_data(dataset)
        if run_option == "test-train":
            # only for test purposes
            X_tree = X_tree[: int(sys.argv[4])]
            Y_tree = Y_tree[: int(sys.argv[4])]
        for i, point in enumerate(track(X_tree, description="Loading on sql…")):
            try:
                with open(
                    Path(get_dataset_metadata(dataset)["path_aemodels"])
                    / f"explanation/{i}.pickle",
                    "rb",
                ) as f:
                    tosave = pickle.load(f)
            except FileNotFoundError:
                tosave = run_explain(i, X_tree, Y_tree, dataset, bb_type)
        # the following creates the actual data point
        miao = TreePoint(
            id=i,
            a=point,
            latent=Latent(
                a=tosave["limg"],
                margins=tosave["neigh_bounding_box"].transpose(),
            ),
            latentdt=LatentDT(
                predicted_class=str(tosave["dt_pred"]),
                model=tosave["dt"],
                fidelity=tosave["fidelity"],
                s_rules=str(tosave["rstr"]),
                s_counterrules=tosave["cstr"],
            ),
            domain=my_domain,
        )
        # blackboxpd=BlackboxPD(predicted_class=str(tosave["bb_pred"])),
        # TODO: check if this is the same conceptually of what I extract myself
        miao.save()

    elif run_option == "list":
        all_records = list_all(dataset, bb_type)
        if all_records:
            print(all_records)
            print(f"We have {len(all_records)} TreePoints in the database.")
        else:
            print("[red]No records")

"""
con = sqlite3.connect(data_table_path, detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
uuu = cur.execute(f"SELECT * FROM sqlite_master").fetchall()
print(f"metadata:\n{uuu}")
"""
