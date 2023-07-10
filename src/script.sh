#!/usr/bin/bash

cd ~/repos/tesi
source venv/bin/activate
cd src

for ((i=0; i<4; i++))
do
	conda run -n tesi python mnist.py explain 100
done

# conda run -n tesi python -c "from pathlib import Path; print(Path.cwd())"
