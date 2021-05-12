# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier


# others
from scipy import stats

# MLFLOW PARAMETERS
MLFLOW_URI = "https://mlflow.lewagon.co/"
CUSTOMURI = ""
myname = "VictorBnnt"
EXPERIMENT_NAME = f"[FR] [Paris] [{myname}] Titanic"


# encoding and scaling parameters
params = {"imputer_strategy": "mean",
          "scaler": StandardScaler()}


# training parameters
######################################################
# SVC model
######################################################
grid_svc = {'kernel': ["rbf", "sigmoid"],#['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'C': stats.loguniform(1, 2),
            #'gamma': 'auto',
            'degree': stats.randint(1, 3)}
model_SVC = SVC()
######################################################

######################################################
# Random Forest Classifier model
######################################################
grid_RFC = {'n_estimators': stats.randint(1, 200),
            'max_depth': stats.randint(1, 40),
            'min_samples_split': [2, 4, 6, 8, 10],
            'criterion': ["gini", "entropy"]
            # 'degree': stats.randint(2, 3)
            }
model_RFC = RandomForestClassifier()
######################################################

######################################################
# GradientBoostingClassifier model
######################################################
grid_GBC = {'loss': ['exponential'],
            'learning_rate': stats.loguniform(0.01, 1),
            'n_estimators': stats.randint(1, 200)
            #'max_depth': stats.randint(1, 100)
            }
model_GBC = GradientBoostingClassifier()
######################################################

######################################################
# GradientBoostingClassifier model
######################################################
grid_ADA = {'n_estimators': stats.randint(1, 200),
            'learning_rate': stats.loguniform(0.5, 3)
            }
model_ADA = AdaBoostClassifier()
######################################################

######################################################
# Ridge classifier model
######################################################
grid_RIDGE = {'alpha': stats.loguniform(0.5, 3),
              'normalize': [True, False],
              'copy_X': [True, False],
              'tol': stats.loguniform(0.01, 1),
              'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
              }
model_RIDGE = RidgeClassifier()
######################################################
