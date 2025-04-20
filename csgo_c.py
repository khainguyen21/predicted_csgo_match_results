import pandas as pd
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv("csgo.csv")

# profile = ProfileReport(data, title = "CSGO Report", explorative=True)
# profile.to_file("csgo_report.html")

column_to_drop = ["day", "month", "year", "date"]
data = data.drop(columns=column_to_drop)

target = "result"

y = data[target]
x = data.drop(target, axis = 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)


# Obtain the name of numerical features
numerical_feature = ['wait_time_s', 'match_time_s', 'team_a_rounds', 'team_b_rounds', 'ping', 'kills', 'assists', 'deaths'
                     ,'mvps', 'hs_percent', 'points']

numerical_transformer = Pipeline([("imputer", SimpleImputer(strategy= "median")),
                                  ("scaler", StandardScaler())])

# Obtain the name of categorical features
categorical_feature = ['map']
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[("num_transformer", numerical_transformer, numerical_feature),
                                               ("cate_transformer", categorical_transformer, categorical_feature)])

reg = Pipeline ([("Preprocessing", preprocessor),
                 ("classifier", LGBMClassifier())
                 ])

# Hyperparameter for the Random Forest
# hyper_params = {"classification__n_estimators": [50, 100, 200],
#                 "classification__criterion": ["gini", "entropy", "log_loss"],
#                 "classification__max_depth": [10, 20, None]}

# # Hyperparameter for the Decision Tree
# hyper_params = {"classification__criterion": ["gini", "entropy", "log_loss"],
#                 "classification__splitter": ["best", "random"],
#                 "classification__max_depth": [None, 50, 100]}

# # Hyperparameter for LGBM
hyper_params = {"classifier__boosting_type": ['gbdt', 'dart', 'rf'],
                "classifier__num_leaves": [31, 11, 51]}

model = RandomizedSearchCV(estimator= reg,
                    param_distributions=hyper_params,
                    scoring="accuracy",
                    cv = 6,
                    n_iter=15,
                   verbose=1)


model.fit(x_train, y_train)
y_predicted = model.predict(x_test)


# for i, j in zip(y_predicted, y_test):
#     print(f"Prediction: {i}, Actual: {j}")

print(classification_report(y_test, y_predicted))
print(model.best_score_)
# print(model.best_params_)


