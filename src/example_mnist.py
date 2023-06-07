from mnist import run_explain
import sys
from rich import print

if __name__ == "__main__":
    try:
        run_options = sys.argv[1]
    except IndexError:
        raise Exception(
            """possible runtime arguments are:\n
            <start_index_image_to_explain>-<end_index_image_to_explain>\n
            example: python example_mnist.py 0-100"""
        )

run_options = run_options.split("-")
my_counter = 0
for i in range(run_options[0], run_options[1]):
    run_explain(i)
    my_counter += 1

print(f"Explained instances from {run_options[0]} to {run_options[1]} amounting to {my_counter} instances.")
