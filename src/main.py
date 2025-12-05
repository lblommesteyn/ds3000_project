import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from data_prep import load_df, add_dep_hour, add_delay_label, hhmm_to_minutes

# basic statistics
df = load_df()
df.head()
df.info()

df = add_dep_hour(df)
df = add_delay_label(df)
df["day_of_week"] = df["day_of_week"].astype("Int8")
df["arr_hour"] = (df["crs_arr_time"].map(hhmm_to_minutes) // 60).astype("Int8")

print("Avg Arrival Delay:", df["arr_delay"].mean())
print("Percent Delayed >15:", df["is_delayed_15"].mean())
print("Number of Flights:", len(df))

# probability of delays by time
delay_by_hour = df.groupby("dep_hour")["is_delayed_15"].mean()

plt.figure(figsize = (10,4))
plt.plot(delay_by_hour.index, delay_by_hour.values, marker = "o")
plt.title("Probability of Delay >15 min by Departure Hour")
plt.xlabel("Scheduled Departure Hour")
plt.ylabel("Delay Rate")
plt.grid(True)
os.makedirs("plots/main", exist_ok=True)
plt.savefig("plots/main/delay_by_hour.png")

# top delays by airport
top_airports = (
    df.groupby("origin")["arr_delay"]
      .mean()
      .sort_values(ascending = False)
      .head(15)
)

print(top_airports)

pivot = (
    df.groupby(["origin", "month"])["arr_delay"]
      .mean()
      .unstack("month")
)

high_volume = df["origin"].value_counts().head(25).index
pivot = pivot.loc[high_volume]

plt.figure(figsize = (12,8))
plt.imshow(pivot, aspect = "auto")
plt.colorbar(label = "Average Arrival Delay (minutes)")
plt.xticks(range(12), range(1,13))
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title("Monthly Delay Patterns Across Major Airports")
plt.savefig("plots/main/delay_by_airport.png")

# logistic regression model
model_df = df[
    ["dep_hour","month","day_of_week","distance","origin","dest","op_unique_carrier","is_delayed_15"]
].dropna()

X = model_df.drop("is_delayed_15", axis = 1)
y = model_df["is_delayed_15"]

preprocess = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown = "ignore"), ["origin","dest","op_unique_carrier"]),
    ("num", "passthrough", ["dep_hour","month","day_of_week","distance"])
])

log_reg = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter = 200))
])

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y,test_size = 0.2)

log_reg.fit(X_train,y_train)
preds = log_reg.predict_proba(X_test)[:,1]

print("ROC-AUC:", roc_auc_score(y_test,preds))
print(classification_report(y_test,(preds > 0.5).astype(int)))

# basic xgboost
xgb = Pipeline([
    ("prep", preprocess),
    ("model", XGBClassifier(
        tree_method = "hist",
        n_estimators = 600,
        max_depth = 8,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        eval_metric = "auc",
        n_jobs = -1
    ))
])

xgb.fit(X_train, y_train)
xgb_preds = xgb.predict_proba(X_test)[:,1]

print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgb_preds))
