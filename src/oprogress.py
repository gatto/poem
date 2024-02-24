import pickle

from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from watchfiles import watch

if __name__ == "__main__":
    first_run = True
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        overall = progress.add_task("🧺️", total=5000)
        task1 = progress.add_task("👀", total=None)
        time = progress.add_task("⏳", total=1)
        progress.start_task(task1)

        for _ in watch("data/progress/progr.pickle"):
            progress.reset(time)
            with open("data/progress/progr.pickle", "rb") as f:
                d = pickle.load(f)
            if d["good"]:
                if first_run:
                    progress.console.print(
                        f"[bright_black]{d['dataset']}, {d['model']} - Ctrl-C to quit"
                    )
                    first_run = False
                progress.update(task1, total=d["total"], completed=d["current"])
                progress.update(overall, completed=d["current_index"] + 1)
            else:
                progress.stop()
                print("[red]ERROR.")
