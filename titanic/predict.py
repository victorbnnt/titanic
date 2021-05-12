# imports
from titanic.trainer import Trainer
from titanic.data import get_data
from titanic.data import clean_data
from titanic.parameters import *
import os
import joblib
import pandas as pd
import argparse
import subprocess
from termcolor import colored

PATH_TO_LOCAL_MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/models/"

def prepare_test_data(data_test=None):
    """ Prepare test set to be predicted """
    # get data
    if data_test is None:
        data_train, data_test = get_data()
    else:
        data_train, _ = get_data()

    # clean data
    data_train = clean_data(data_train)
    X_test = clean_data(data_test)

    # set X train and y train
    X_train = data_train.drop(["Survived"], axis=1)
    y_train = data_train["Survived"]

    # fit train data
    trainer = Trainer(X_train, y_train)
    trainer.encode_and_scale()

    # prepare test data
    X_test = trainer.encode_and_scale(X_test=X_test)

    return X_test, data_test



def get_model(model):
    return joblib.load(os.path.join(PATH_TO_LOCAL_MODEL, model + ".joblib"))



def generate_submission_csv(model="model", export_name="titanic_prediction"):
    """ Generate csv file to be sent to Kaggle """

    # X_test preparation
    X_test, data_test = prepare_test_data()

    # load model
    try:
        trained_model = get_model(model)
    except:
        print("Model named " + model + " not found in models/ folder. Please train a model first.")
        return

    # predict test set
    y_pred = trained_model.predict(X_test)

    # Format dataframe to be send to kaggle
    to_send_to_kaggle = pd.concat([data_test[["PassengerId"]],
                        pd.DataFrame(y_pred)],axis=1).rename(columns={0: "Survived"})

    # Write .csv file to be sent to kaggle competition
    to_send_to_kaggle.to_csv(export_name + ".csv", index=False)
    print(colored("Submission file saved locally under " + export_name + ".csv", "green"))



# terminal parameter definition
parser = argparse.ArgumentParser(description='Titanic victimes predict')
parser.add_argument('-m', action="store",
                          dest="modelname",
                          help='.joblib model to load for prediction - default: model',
                          default="model")
parser.add_argument('-s', action="store",
                          dest="tokaggle",
                          help='Kaggle submission csv name - default: titanic_prediction',
                          default="titanic_prediction")


if __name__ == "__main__":

    # getting optionnal arguments otherwise default
    results = parser.parse_args()

    generate_submission_csv(model=results.modelname, export_name=results.tokaggle)
    subprocess.run(["mv", results.tokaggle + ".csv", "kaggle"])
