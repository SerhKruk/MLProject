import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from utils.dataloader import DataLoader
from settings.constants import TRAIN_CSV

train = pd.read_csv(TRAIN_CSV, header = 0)
X_raw = train.drop("Attrition_Flag", axis=1)
y = train["Attrition_Flag"]
loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()

classifiers = {
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

for (name, model) in classifiers.items():
    m = model
    m.fit(X, y)

    with open('models/{}.pickle'.format(name), 'wb') as f:
        pickle.dump(m, f)