# Biological Age Prediction
In this project, we propose a prediction ML model which predicts medical outcomes (e.g. Death and
Biological Age) of a patient. We utilize the provided datasets related to Opportunistic Cardiometabolic
Screening (CT Data and Clinical Data ) to train our model and predict certain medical outcomes with
a low error and high accuracy

## Install

```
python3 -m venv 760
source ./760/bin/activate
which pip
pip install -r requirements.txt
```


## Train
Training models for the 3 problems have been divided as part1, part2 and part3 corresponding to each problem posited in the project. All files are in "Train Src"

```
which python
cd "Train Src"
python part1.py
python part2.py
python part3.py
```

## Demo
The driver code can be used to run 2 examples with CT data drawn from the given sample
```
python Driver.py 
```
