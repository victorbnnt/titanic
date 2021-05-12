# imports
import pandas as pd
import argparse
import subprocess
from termcolor import colored
from titanic.data import get_data
from titanic.data import clean_data
from titanic.parameters import *

# sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate

# mlflow
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import mlflow

# joblib
import joblib


# Update to change parameters to test
MODEL = model_GBC
GRID = grid_GBC


class Trainer():

    def __init__(self, X, y, params=params):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.model = None
        self.scaler = None
        self.X_test = None
        self.params = params
        self.X = X
        self.y = y
        self.baseline_accuracy = None
        self.optimized_accuracy = None
        self.experiment_name = EXPERIMENT_NAME

    def encode_and_scale(self, X_test=None):
        """ encode and scale dataframe """

        if X_test is None:
            df = self.X.copy()
        else:
            df = X_test.copy()

        # Binary encode Sex feature
        df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)

        # OneHotEncode Pclass feature
        if df.shape[0]!=1:
            feat = "Pclass"
            ohe = OneHotEncoder(sparse=False)
            ohe.fit(df[[feat]])
            col = list(ohe.get_feature_names([feat]))
            df[col] = ohe.transform(df[[feat]])
            df.drop(columns=feat, inplace=True)

        # impute for missing values in Age feature
        df["Age"] = SimpleImputer(strategy=self.params["imputer_strategy"]).fit_transform(df[["Age"]])

        # Scale dataframe
        if X_test is None:
            self.scaler = self.params["scaler"].fit(df)
            self.X = self.scaler.transform(df)

            # ### MLFLOW RECORDS
            self.mlflow_log_param("Numeric imputer", self.params["imputer_strategy"])
            self.mlflow_log_param("Scaler", self.params["scaler"])
        else:
            df["Fare"] = SimpleImputer(strategy=self.params["imputer_strategy"]).fit_transform(df[["Fare"]])
            return self.scaler.transform(df)



    def cross_validate_baseline(self, model=model_SVC, cv=20):
        """ compute chosen model baseline accuracy """

        baseline = cross_validate(model,
                                  self.X,
                                  self.y,
                                  scoring="accuracy",
                                  cv=cv)
        self.baseline_accuracy = round(baseline["test_score"].mean(), 3)
        print("Baseline " + type(model).__name__ + " model accuracy: " +
              str(self.baseline_accuracy*100) + "%")

        # ### MLFLOW RECORDS
        self.mlflow_log_metric("Baseline accuracy", self.baseline_accuracy)
        self.mlflow_log_param("Model", type(model).__name__)



    def titanic_train(self, grid=grid_svc, model=model_SVC):
        """training baseline model"""

        """search best parameters and train model"""
        self.model = RandomizedSearchCV(model,
                                        grid,
                                        scoring='accuracy',
                                        n_iter=500,
                                        cv=5,
                                        n_jobs=-1)
        self.model.fit(self.X, self.y)
        self.optimized_accuracy = round(self.model.best_score_, 3)
        print("Tuned " + type(model).__name__ + " model accuracy: " +
              str(round(self.optimized_accuracy*100, 3)) + "%")

        # ### MLFLOW RECORDS
        self.mlflow_log_metric("Optimized accuracy", self.optimized_accuracy)
        for k, v in self.model.best_params_.items():
            self.mlflow_log_param(k, v)



    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(CUSTOMURI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



    def save_model(self, model_name):
        """ Save the model into a .joblib format """
        joblib.dump(self.model.best_estimator_, model_name + ".joblib")
        print(colored("Trained model saved locally under " + model_name + ".joblib", "green"))


# terminal parameter definition
parser = argparse.ArgumentParser(description='Titanic trainer')
parser.add_argument('-m', action="store",
                          dest="modelname",
                          help='.joblib model name - default: model',
                          default="model")

if __name__ == "__main__":
    # getting optionnal arguments otherwise default
    results = parser.parse_args()

    # get data
    data_train, data_test = get_data()

    # clean data
    data_train = clean_data(data_train)

    # set X and y
    X = data_train.drop(["Survived"], axis=1)
    y = data_train["Survived"]

    # define trainer
    trainer = Trainer(X, y)
    trainer.encode_and_scale()

    # get accuracy
    trainer.cross_validate_baseline(model=MODEL)
    trainer.titanic_train(grid=GRID, model=MODEL)

    # saving trained model and moving it to models folder
    trainer.save_model(model_name=results.modelname)
    subprocess.run(["mv", results.modelname + ".joblib", "models"])


