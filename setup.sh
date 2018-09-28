# create virtual environment
conda create --name venv python=2.7

# activate virtual environment 
source activate venv

# install requirements
pip install -r requirements.txt

# no args behaviour of training.py is to download dataset.
python ./src/training.py