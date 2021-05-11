# imports
import pandas as pd
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
MODEL = model_ADA
GRID = grid_ADA


class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.model = None
        self.scaler = None
        self.X_test = None
        self.X = X
        self.y = y
        self.baseline_accuracy = None
        self.optimized_accuracy = None
        self.experiment_name = EXPERIMENT_NAME

    def encode_and_scale(self, params=params, training_set=True):
        """ encode and scale dataframe """

        if training_set==True:
            df = self.X.copy()
        else:
            df = self.X_test.copy()

        # Binary encode Sex feature
        df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)

        # OneHotEncode Pclass feature
        feat = "Pclass"
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(df[[feat]])
        col = list(ohe.get_feature_names([feat]))
        df[col] = ohe.transform(df[[feat]])
        df.drop(columns=feat, inplace=True)

        # impute for missing values in Age feature
        df["Age"] = SimpleImputer(strategy=params["imputer_strategy"]).fit_transform(df[["Age"]])

        # Scale dataframe
        if training_set==True:
            self.scaler = params["scaler"].fit(df)
            self.X = self.scaler.transform(df)

            # ### MLFLOW RECORDS
            self.mlflow_log_param("Numeric imputer", params["imputer_strategy"])
            self.mlflow_log_param("Scaler", params["scaler"])
        else:
            df["Fare"] = SimpleImputer(strategy=params["imputer_strategy"]).fit_transform(df[["Fare"]])
            self.X_test = self.scaler.transform(df)



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
              str(self.optimized_accuracy*100) + "%")

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

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.model.best_estimator_, 'titanic.joblib')

    def generate_kaggle_submission(self, test_set, export_name="titanic_prediction"):
        self.X_test = test_set
        self.X_test = clean_data(self.X_test)

        # Encode and scale
        self.encode_and_scale(params=params, training_set=False)

        # prediction on test set with optimized model
        y_pred = self.model.best_estimator_.predict(self.X_test)

        # Format dataframe to be send to kaggle
        to_send_to_kaggle = pd.concat([test_set[["PassengerId"]],
                               pd.DataFrame(y_pred)],axis=1).rename(columns={0: "Survived"})

        # Write .csv file to be sent to kaggle competition
        to_send_to_kaggle.to_csv(export_name + ".csv", index=False)




if __name__ == "__main__":

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

    # generate kaggle submission file
    trainer.generate_kaggle_submission(data_test)
