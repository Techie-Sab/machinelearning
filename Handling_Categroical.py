from statistics import mode
import pandas as pd
import numpy as np
from sklearn import linear_model,metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import sparse
from sklearn import ensemble
from sklearn import decomposition

import warnings
warnings.filterwarnings("ignore")


def run(df,fold):

    features=[each for each in df.columns if each not in ['age','kfold']]

    for each in features:

        df.loc[:,each]=df[each].astype(str).fillna('NONE')

        df_train=df.loc[df['kfold']!=fold].reset_index(drop=True)
        df_valid=df.loc[df['kfold']==fold].reset_index(drop=True)

        ohe=preprocessing.OneHotEncoder()

        full_data=pd.concat([df_train[features],df_valid[features]])

        ohe.fit(full_data)

        x_train=ohe.transform(df_train[features])

        y_train=df_train.income.values

        x_valid=ohe.transform(df_valid[features])

        model=linear_model.LogisiticRegression()

        model.fit(x_train,y_train)

        valid_preds=model.predict_log_proba(x_valid)[:,1]

        auc=metrics.auc_score(df_valid.income.values,valid_preds)

        print(f'Fold = {fold}, AUC = {auc}')


if __name__=="__main__":

    df=pd.read_csv('input.csv')

    for each in range(5):
        run(df,each)