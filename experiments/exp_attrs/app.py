from attrs import Factory, define, field, validators
import random


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

    counterfactuals: list[int] = field(init=False)

    @counterfactuals.default
    def _counterfactuals_default(self, more=False):
        results = []

        for rule in range(1):
            print(point := random.choice([0, 1]))

            if point:
                results.append(point)

        return results


miao = Explainer()
gegiu = Explainer()

print(f"{miao=}")
miao.counterfactuals.append(5)
print(f"{miao=}")

print(f"{gegiu=}")
