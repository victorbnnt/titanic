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



def get_model(model):
    return joblib.load(os.path.join(PATH_TO_LOCAL_MODEL, model + ".joblib"))



def generate_submission_csv(model="model", export_name="titanic_prediction"):
    """ Generate csv file to be sent to Kaggle """

    # get test data
    _, data_test = get_data()
    X_test = data_test.copy()
    X_test = clean_data(X_test)

    # load model
    try:
        trained_model = get_model(model)
    except:
        print("Model named " + model + " not found in models/ folder. Please train a model first.")
        return

    # predict test set
    if "best_estimator_" in dir(trained_model):
        y_pred = trained_model.best_estimator_.predict(X_test)
    else:
        y_pred = trained_model.predict(X_test)

    # Format dataframe to be send to kaggle
    to_send_to_kaggle = pd.concat([data_test[["PassengerId"]],
                                  pd.DataFrame(y_pred)], axis=1).rename(columns={0: "Survived"})

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
