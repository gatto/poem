# Possible points of failure

## knn

The function takes a TestPoint and looks for the closest TreePoint whose margins contain the TestPoint. If the testpoint is not within these margins, the TreePoint is popped and knn runs again. This can mean that all points are popped because the testpoint falls within no margins.

### current strategy

Right now checks ALL TreePoints (e.g. 10k) and raise RuntimeError if no suitable TreePoint is found.

Possible alternatives:

- check only a certain number of TreePoints before failing or a certain ratio of treepoints
- Do not fail at all: if I would fail, throw away the margins check and take the closest 1-nn treepoint, regardless of margins.

## counterfactual generation

`TestPoint.marginal_apply()` is a loop. First run, set the feature value on the decision boundary. Checks discriminator, if I don't pass it, run again, up to 40. Each run getting further and further from the decision boundary. If I reach 40 times and discriminator doesn't like it, fail.

### current strategy

return None, skipping that specific counterrule for generation of counterexemplars

Possible alternatives:

- throw exception? (blargh)
- keep going more? (must be limited though)

## factual generation

1. for each feature:
    1. take random (gauss) value
    2. validate it according to complexrule for feature in question
    3. keep going until get a value accepted by complexrule

2. I have a point, validate it to discriminator
3. pass? return
4. no pass? start again from 1, but only up to 40 times
5. I tried 40 times? Fail.

### current strategy

if fail, `raise RuntimeError`

Possible alternatives:

- keep going after 40 tries? How many?
- return None, leaving the user freedom to call it again or not?
- done: implement a method with the explainer that generates more factuals. Happy :) it's Explainer.refresh_prototypes()
