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


# training parameters
######################################################
# SVC model
######################################################
grid_svc = {'model__kernel': ["rbf", "sigmoid"],#['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'model__C': stats.loguniform(1, 2),
            #'gamma': 'auto',
            'model__degree': stats.randint(1, 3),
            "preprocessor__encoder__numeric__imputer__strategy": ["mean", "median"],
            "preprocessor__scaler__scaler": [StandardScaler(), RobustScaler()]
            }
#
params_SVC = {"random_grid_search": grid_svc,
              "model": SVC()}
######################################################

######################################################
# Random Forest Classifier model
######################################################
grid_RFC = {'model__n_estimators': stats.randint(1, 200),
            'model__max_depth': stats.randint(1, 40),
            'model__min_samples_split': [2, 4, 6, 8, 10],
            'model__criterion': ["gini", "entropy"],
            "preprocessor__encoder__numeric__imputer__strategy": ["mean", "median"],
            "preprocessor__scaler__scaler": [StandardScaler(), RobustScaler()]
            # 'degree': stats.randint(2, 3)
            }
params_RFC = {"random_grid_search": grid_RFC,
              "model": RandomForestClassifier()}
######################################################

######################################################
# GradientBoostingClassifier model
######################################################
grid_GBC = {'model__loss': ['exponential'],
            'model__learning_rate': stats.loguniform(0.01, 1),
            'model__n_estimators': stats.randint(1, 200),
            "preprocessor__encoder__numeric__imputer__strategy": ["mean", "median"],
            "preprocessor__scaler__scaler": [StandardScaler(), RobustScaler()]
            #'max_depth': stats.randint(1, 100)
            }
params_GBC = {"random_grid_search": grid_GBC,
              "model": GradientBoostingClassifier()}
######################################################

######################################################
# GradientBoostingClassifier model
######################################################
grid_ADA = {'model__n_estimators': stats.randint(1, 200),
            'model__learning_rate': stats.loguniform(0.5, 3),
            "preprocessor__encoder__numeric__imputer__strategy": ["mean", "median"],
            "preprocessor__scaler__scaler": [StandardScaler(), RobustScaler()]
            }
params_ADA = {"random_grid_search": grid_ADA,
              "model": AdaBoostClassifier()}
######################################################

######################################################
# Ridge classifier model
######################################################
grid_RIDGE = {'model__alpha': stats.loguniform(0.5, 3),
              'model__normalize': [True, False],
              'model__copy_X': [True, False],
              'model__tol': stats.loguniform(0.01, 1),
              'model__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
              "preprocessor__encoder__numeric__imputer__strategy": ["mean", "median"],
              "preprocessor__scaler__scaler": [StandardScaler(), RobustScaler()]
              }
params_RIDGE = {"random_grid_search": grid_RIDGE,
                "model": RidgeClassifier()}
######################################################
