import pandas as pd
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
data = pd.read_csv("csgo.csv")

# profile = ProfileReport(data, title = "CSGO Report", explorative=True)
# profile.to_file("csgo_report.html")

column_to_drop = ["map", "day", "month", "year", "date"]
data = data.drop(columns=column_to_drop)

y = data["result"]
x = data.drop("result", axis = 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)



# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)

num_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy= "median")),
                                     ("scaler", StandardScaler())
                            ])

x_train = num_transformer.fit_transform(x_train)
x_test = num_transformer.transform(x_test)

params = {"n_estimators": [50, 100, 200],
          "criterion": ["gini", "entropy", "log_loss"],
          #"max_depth": [10, 20, None]
          }


model = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                     param_grid=params,
                     scoring="accuracy",
                     cv = 6)

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)


for i, j in zip(y_predicted, y_test):
    print(f"Prediction: {i}, Actual: {j}")

# print(classification_report(y_test, y_predicted))

