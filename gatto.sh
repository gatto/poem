#!/usr/bin/bash
#
# Start script by gatto v0.2
#
# To use it:
# chmod +x gatto.sh
# mv -f gatto.sh ~/.local/bin/gatto
# gatto
#
cd ~/repos/tesi
source ~/repos/tesi/venv/bin/activate
source ~/miniconda3/bin/activate tesi
jupyter notebook --ip 0.0.0.0 --no-browser
