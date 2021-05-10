import pandas as pd
import os


train_set = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/raw_data/train.csv"


def get_data():
    '''returns training titanic set DataFrames'''
    df = pd.read_csv(train_set)
    return df


def clean_data(df):
    ''' return clean dataframe '''
    df = df.drop(["Name",
                  "PassengerId",
                  "Ticket",
                  "Embarked",
                  "Parch",
                  "SibSp",
                  "Cabin"], axis=1)
    return df


if __name__ == '__main__':
    df = get_data()
    print("Data loaded. Shape:" + str(df.shape))
