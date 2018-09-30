sh setup.sh

# activate virtual environment 
source activate venv-jvcr

# Trains Hourglass network --pre_train_mode is specified.
python ./src/training.py -mode=pre-train --pre_train_mode=hourglass --optimizer=RMSPROP
#python ./src/training.py -mode=pre-train --pre_train_mode=hourglass --optimizer=RMSPROP &
#backgroundProcess=$!

# Trains coordinate regression if --pre_train_mode is not specified.
python ./src/training.py --mode=pre-train --optimizer=RMSPROP
#python training.py --mode=pre-train --optimizer=RMSPROP &
#backgroundProcess=$!

# deactivate virtual environment 
source deactivate venv-jvcr
