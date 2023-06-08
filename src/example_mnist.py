import sys
from parameters import path_aemodels
from rich import print

if __name__ == "__main__":
    try:
        run_options = sys.argv[1]
    except IndexError:
        raise Exception(
            """possible runtime arguments are:
            <how many images to explain>
            example: python example_mnist.py 100
            will explain 100 images not already explained"""
        )
    from mnist import run_explain, get_data

    run_options = int(run_options)
    my_counter = 0
    explanation_path = path_aemodels / "explanation"
    # check what I've already done
    max_i = int(max(explanation_path.glob("*.pickle")).stem)
    
    (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = get_data()

    # do some more
    for i in range(max_i+1, max_i+1+run_options):
        run_explain(i, X_tree, Y_tree)
        my_counter += 1

    print(
        f"Explained instances from {run_options[0]} to {run_options[1]} amounting to {my_counter} instances."
    )
