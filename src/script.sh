#!/usr/bin/bash

cd ~/repos/tesi
source venv/bin/activate
cd src

for ((i=540; i<5000; i = i + 5))
do
	/home/frusso/miniconda3/condabin/conda run -n tesi /home/frusso/repos/tesi/venv/bin/python oab.py fashion dnn explain "$i"
done

# conda run -n tesi python -c "from pathlib import Path; print(Path.cwd())"
