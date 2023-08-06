#!/usr/bin/bash

cd ~/repos/tesi
source venv/bin/activate
cd src

for ((i=0; i<6; i++))
do
	/home/frusso/miniconda3/condabin/conda run -n tesi /home/frusso/repos/tesi/venv/bin/python mnist.py explain 100
done

# conda run -n tesi python -c "from pathlib import Path; print(Path.cwd())"
