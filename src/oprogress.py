import pickle

from rich.progress import Progress
from watchfiles import watch

if __name__ == "__main__":
    print("[gray]Ctrl-c to quit")
    with Progress() as progress:
        task1 = progress.add_task("[purple]👀...", total=None)

        for _ in watch("data/progress/progr.pickle"):
            with open("data/progress/progr.pickle", "rb") as f:
                d = pickle.load(f)
            if d["good"]:
                progress.update(task1, total=d["total"], advance=1)
            else:
                progress.stop()
                print("[red]ERROR.")
