import pickle

from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from watchfiles import watch

if __name__ == "__main__":
    print("[bright_black]Ctrl-C to quit")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task1 = progress.add_task("👀", total=None)
        progress.start_task(task1)

        for _ in watch("data/progress/progr.pickle"):
            with open("data/progress/progr.pickle", "rb") as f:
                d = pickle.load(f)
            if d["good"]:
                progress.update(task1, total=d["total"], completed=d["current"])
            else:
                progress.stop()
                print("[red]ERROR.")
