import pandas as pd
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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


# categorical_feature = ['Dust II', 'Mirage', 'Cache',
#                        'Cobblestone', 'Inferno',  'Overpass',
#                        'Austria', 'Nuke', 'Canals', 'Italy']
#
# map_transformer = Pipeline(steps= [("imputer", SimpleImputer(strategy="most_frequent")),
#                                    ("Scaler", OneHotEncoder(categories=[categorical_feature], handle_unknown="ignore"))])
#
# # Apply the transformer to 'map' column
# map_encoded = map_transformer.fit_transform(data[['map']])
#
# # Convert encoded output to DataFrame
# map_encoded_df = pd.DataFrame.sparse.from_spmatrix(
#     map_encoded,
#     columns=map_transformer.named_steps['Scaler'].get_feature_names_out(['map'])
# )
#
# # Drop original 'map' column and concatenate encoded data
# data = pd.concat([data.drop('map', axis=1), map_encoded_df], axis=1)

# Obtain the name of numerical column
# numerical_feature = data.select_dtypes(include=['int64', 'float64']).columns
numerical_feature = ['wait_time_s', 'match_time_s', 'team_a_rounds', 'team_b_rounds', 'ping', 'kills', 'assists', 'deaths'
                     ,'mvps', 'hs_percent', 'points']
#categorical_feature = data.select_dtypes(include=['category', 'object']).columns
categorical_feature = ['map']


y = data["result"]
x = data.drop("result", axis = 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)



numerical_transformer = Pipeline([("imputer", SimpleImputer(strategy= "median")),
                                  ("scaler", StandardScaler())])

categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, numerical_feature),
                                               ("cat", categorical_transformer, categorical_feature)])


x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

# num_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy= "median")),
#                                      ("scaler", StandardScaler())
#                             ])

# x_train = num_transformer.fit_transform(x_train)
# x_test = num_transformer.transform(x_test)

# paramsRandomForest = {"n_estimators": [50, 100, 200],
#           "criterion": ["gini", "entropy", "log_loss"],
#           "max_depth": [10, 20, None]
#           }

paramsDecisionTree = {"criterion": ["gini", "entropy", "log_loss"],
                      "splitter": ["best", "random"],
                      "max_depth": [None, 50, 100]}

model = GridSearchCV(estimator= DecisionTreeClassifier(random_state=42),
                     param_grid=paramsDecisionTree,
                     scoring="accuracy",
                     cv = 6)

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)


# for i, j in zip(y_predicted, y_test):
#     print(f"Prediction: {i}, Actual: {j}")

print(classification_report(y_test, y_predicted))

print(model.best_score_)
print(model.best_params_)


