import time
from rich.progress import Progress
from watchfiles import watch


if __name__ == "__main__":
    with Progress() as progress:
        task1 = progress.add_task("[red]👀...", total=200)

        for changes in watch("data/progress/progr.pickle"):
            print(changes)
