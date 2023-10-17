import sys
import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model(df: pd.DataFrame):
    x = df.drop('Survived',axis=1)
    y = df.Survived
    cate_features_index = np.where(x.dtypes != float)[0]
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.85,random_state=1234)
    model = CatBoostClassifier(eval_metric='Accuracy', use_best_model=True, random_seed=42)
    model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
    cv_data = cv(model.get_params(),Pool(x,y,cat_features=cate_features_index),fold_count=10)
    print('Best CV Accuracy: {}'.format(np.max(cv_data["b'Accuracy'_test_avg"])))
    print('Test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))
    model.save_model('../../data/models/titanic_catboost_model.cbm')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        train_model(csv_file)
    else:
        print("Please provide the CSV file path as an argument.")