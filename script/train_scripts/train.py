import sys
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model(df: pd.DataFrame):
    x = df.drop('Survived',axis=1)
    y = df.Survived
    cate_features_index = np.where(x.dtypes != float)[0]
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.85,random_state=1234)

    model = CatBoostClassifier(eval_metric='Accuracy', use_best_model=True, random_seed=42)
    model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))

    print('Test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))

    model_directory = "data/models/"
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, "titanic_catboost_model.cbm")
    model.save_model(model_path)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        df = pd.read_csv(csv_file)
        train_model(df)
    else:
        print("Please provide the CSV file path as an argument.")