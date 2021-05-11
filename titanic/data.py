import pandas as pd
import os


train_set = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/raw_data/train.csv"
test_set = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/raw_data/test.csv"


def get_data():
    '''returns training titanic set DataFrames'''
    df_train = pd.read_csv(train_set)
    df_test = pd.read_csv(test_set)
    return df_train, df_test


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
    df_train, df_test = get_data()
    print("Train dataframe loaded. Shape:" + str(df_train.shape))
    print("Test dataframe loaded. Shape:" + str(df_test.shape))
