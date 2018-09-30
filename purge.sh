
source deactivate

conda remove --yes --name venv-jvcr --all

# remove the downloaded packages
conda info --envs

# remove temp files
rm -rf /tmp/JVCR/*