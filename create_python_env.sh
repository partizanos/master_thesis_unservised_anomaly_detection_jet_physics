# https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b

# virtualenv venv_master_thesis_exp
virtualenv -p python3 venv_master_thesis_exp


# custom shell
# source venv/bin/activate.fish
source venv_master_thesis_exp/bin/activate


# deactivate




# pip install numpy

# pip install pandas
## pytz-2020.4-py2.py3-none-any.whl
## Installing collected packages: six, pytz, python-dateutil, pandas

pip install matplotlib
## Installing collected packages: pyparsing, pillow, kiwisolver, cycler, matplotlib

pip install seaborn
pip install tensorflow==2.2.0
pip install keras