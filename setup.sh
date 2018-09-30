# create virtual environment
conda create --yes --name venv-jvcr python=2.7

# activate virtual environment 
source activate venv-jvcr

pip install --upgrade pip

# install requirements
pip install -r requirements.txt
# conda install --yes --file requirements.txt

# no args behaviour of training.py is to download dataset.
python ./src/training.py
