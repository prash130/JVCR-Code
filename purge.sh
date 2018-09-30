
# remove the downloaded packages
conda remove --yes --name venv-jvcr --all

#check if the venv-jvcr doesn't exist
conda info --envs

# remove temp files
rm -rf /tmp/JVCR/*