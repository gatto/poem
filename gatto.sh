# start script by gatto v0.1
#
# to use it:
# chmod +x gatto.sh
# mv gatto.sh ~/.local/bin/gatto
cd repos/tesi
source venv/bin/activate
conda activate tesi
jupyter notebook --ip 0.0.0.0 --no-browser
