sh setup.sh

# activate virtual environment 
source activate venv-jvcr

# no args behaviour of training.py is to download dataset.
python ./src/training.py -mode=pre-train --pre_train_mode=hourglass --optimizer=SGD --momentum=0.1 &

backgroundProcess=$!

# use the process id stored in backgroundProcess to stop the process.
echo "bacg process is $backgroundProcess"

# deactivate virtual environment 
source deactivate venv-jvcr