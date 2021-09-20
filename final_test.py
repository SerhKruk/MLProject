import json
import requests
import pandas as pd

from utils import DataLoader
from settings.constants import TRAIN_CSV, VAL_CSV

train = pd.read_csv(TRAIN_CSV, header=0)
val = pd.read_csv(VAL_CSV, header=0)

X_raw = train.drop("Attrition_Flag", axis=1)
loader = DataLoader()
loader.fit(X_raw)
metrics = "accuracy_score"
X = loader.load_data()
y = train["Attrition_Flag"]

req_data = {'data': json.dumps(X.to_dict())}
response = requests.get('http://127.0.0.1:5000/predict', data=req_data)

api_predict = response.json()['predictions']
print('predict: ', api_predict[:10])

api_score = eval(metrics)(y, api_predict)
print('accuracy: ', api_score)