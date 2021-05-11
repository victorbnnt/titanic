# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

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
