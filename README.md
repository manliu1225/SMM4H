# SMM4H
pipenv install
pip3 install git+https://www.github.com/keras-team/keras-contrib.git
pip install -U scikit-learn
sudo pip install Cython
pip install pyjnius
pipenv shell


# prepare data

# convert data
python convert_file.py

# preprocess data
python data_preprocessing.py
