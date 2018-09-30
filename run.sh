sh setup.sh

# activate virtual environment 
source activate venv-jvcr

# Trains Hourglass network --pre_train_mode is specified.
python ./src/training.py -mode=pre-train --pre_train_mode=hourglass --optimizer=SGD --momentum=0.1 &
backgroundProcess=$!

# use the process id stored in backgroundProcess to stop the process.
echo "background process is $backgroundProcess"

# Trains coordinate regression if --pre_train_mode is not specified.
python ./src/training.py -mode=pre-train --optimizer=SGD --momentum=0.1 &
backgroundProcess=$!

# use the process id stored in backgroundProcess to stop the process.
echo "background process is $backgroundProcess"

tensorboard --logdir='/tmp/JVCR/logs' --port=6006

# deactivate virtual environment 
source deactivate venv-jvcr
