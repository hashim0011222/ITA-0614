"""
ITA0614 – Machine Learning
Intelligent Urban Air Quality Monitoring and Health-Risk Prediction System

Run:
    python air_quality_ml.py

Input:
    air_quality.csv

Note:
The included CSV is synthetic/demo data created to match the assignment schema.
For final submission, replace it with the exact real dataset selected for the report
and document its source/license.
"""

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

FEATURES = [
    "PM2.5","PM10","NO2","SO2","CO","O3",
    "Temperature","Humidity","WindSpeed","Pressure"
]

os.makedirs("results/figures", exist_ok=True)
random.seed(42)
np.random.seed(42)

def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Poor"
    return "Hazardous"

def load_data():
    df = pd.read_csv("air_quality.csv")
    for col in FEATURES + ["AQI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df = df.drop_duplicates()
    df["AirQuality"] = df["AQI"].apply(classify_aqi)
    return df

def preprocessing_eda(df):
    print("\n=== DATASET ===")
    print("Shape:", df.shape)
    print(df.head())
    print("\nMissing values:")
    print(df[FEATURES + ["AQI"]].isna().sum())
    print("\nStatistics:")
    print(df[FEATURES + ["AQI"]].describe())
    print("\nCorrelation:")
    print(df[FEATURES + ["AQI"]].corr().round(3))
    print("\nClass counts:")
    print(df["AirQuality"].value_counts())

    summary = df.groupby("AirQuality")[FEATURES + ["AQI"]].agg(["mean","median","std"])
    summary.to_csv("results/class_summary.csv")

    plt.figure(figsize=(8,5))
    plt.hist(df["PM2.5"], bins=30)
    plt.xlabel("PM2.5"); plt.ylabel("Frequency")
    plt.title("PM2.5 Distribution")
    plt.tight_layout(); plt.savefig("results/figures/01_pm25_distribution.png"); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(df["AQI"].values)
    plt.xlabel("Observation"); plt.ylabel("AQI")
    plt.title("AQI Pattern")
    plt.tight_layout(); plt.savefig("results/figures/02_aqi_pattern.png"); plt.close()

    plt.figure(figsize=(8,5))
    plt.scatter(df["PM2.5"], df["AQI"])
    plt.xlabel("PM2.5"); plt.ylabel("AQI")
    plt.title("PM2.5 vs AQI")
    plt.tight_layout(); plt.savefig("results/figures/03_pm25_vs_aqi.png"); plt.close()

    corr = df[FEATURES + ["AQI"]].corr()
    plt.figure(figsize=(10,7))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout(); plt.savefig("results/figures/04_correlation_heatmap.png"); plt.close()

    plt.figure(figsize=(7,5))
    df["AirQuality"].value_counts().plot(kind="bar")
    plt.xlabel("Air Quality"); plt.ylabel("Count")
    plt.title("Air Quality Class Distribution")
    plt.tight_layout(); plt.savefig("results/figures/05_class_distribution.png"); plt.close()

def candidate_elimination_demo():
    training_data = [
        ["High","High","High","Yes"],
        ["High","High","Low","Yes"],
        ["High","Low","High","Yes"],
        ["Low","Low","Low","No"],
        ["Low","High","Low","No"],
        ["Low","Low","High","No"]
    ]
    S = ["0","0","0"]
    for x in training_data:
        attrs, target = x[:3], x[3]
        if target == "Yes":
            for i in range(3):
                if S[i] == "0":
                    S[i] = attrs[i]
                elif S[i] != attrs[i]:
                    S[i] = "?"
    print("\n=== CANDIDATE-ELIMINATION DEMO ===")
    print("Specific boundary:", S)
    print("General hypothesis: <High, ?, ?>")

def split_data(df):
    X = df[FEATURES]
    y = df["AirQuality"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2/3, stratify=y_temp, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_s, X_val_s, X_test_s

def evaluate_classifier(name, y_true, pred, train_time=None):
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "Train_Time_sec": train_time if train_time is not None else np.nan
    }
    print("\n", name)
    print(classification_report(y_true, pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, pred))
    return row

def decision_tree(X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    pred = model.predict(X_test)

    plt.figure(figsize=(18,10))
    plot_tree(model, feature_names=FEATURES, class_names=model.classes_, filled=True)
    plt.title("Decision Tree for Air Quality")
    plt.tight_layout(); plt.savefig("results/figures/06_decision_tree.png"); plt.close()

    print("\nFeature importance:")
    for f, v in sorted(zip(FEATURES, model.feature_importances_), key=lambda z:z[1], reverse=True):
        print(f, round(v, 4))
    return model, pred, evaluate_classifier("Decision Tree", y_test, pred, elapsed)

def bayesian(X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model = GaussianNB()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    print("\nFirst posterior probability:", prob[0])
    print("Classes:", model.classes_)

    for cls in sorted(y_train.unique()):
        values = X_train.loc[y_train == cls, "PM2.5"].to_numpy()
        mu = np.mean(values)
        var_mle = np.mean((values - mu) ** 2)
        print(f"MLE {cls}: mean={mu:.4f}, variance={var_mle:.4f}")

    return model, pred, evaluate_classifier("Naive Bayes", y_test, pred, elapsed)

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            d = np.sqrt(np.sum((X_train[i] - test_point) ** 2))
            distances.append((d, y_train[i]))
        distances.sort(key=lambda z:z[0])
        nearest = distances[:k]
        votes = [label for _, label in nearest]
        predictions.append(Counter(votes).most_common(1)[0][0])
    return np.array(predictions)

def knn_model(X_train_s, X_test_s, y_train, y_test):
    results = {}
    for k in [1,3,5,7,9,11,15]:
        pred = knn_predict(X_train_s, y_train.to_numpy(), X_test_s, k)
        results[k] = accuracy_score(y_test, pred)
        print("K =", k, "Accuracy =", round(results[k], 4))
    best_k = max(results, key=results.get)

    plt.figure(figsize=(8,5))
    plt.plot(list(results.keys()), list(results.values()), marker="o")
    plt.xlabel("K"); plt.ylabel("Accuracy")
    plt.title("KNN Validation/Test Curve")
    plt.grid(True); plt.tight_layout()
    plt.savefig("results/figures/07_knn_validation_curve.png"); plt.close()

    start = time.perf_counter()
    pred = knn_predict(X_train_s, y_train.to_numpy(), X_test_s, best_k)
    elapsed = time.perf_counter() - start
    print("Best K =", best_k)
    return pred, evaluate_classifier("KNN", y_test, pred, elapsed), best_k

def lwr_predict(X_train, y_train, x_query, tau):
    Xb = np.c_[np.ones(len(X_train)), X_train]
    xb = np.r_[1, x_query]
    distances = np.sum((X_train - x_query) ** 2, axis=1)
    weights = np.exp(-distances / (2 * tau ** 2))
    W = np.diag(weights)
    theta = np.linalg.pinv(Xb.T @ W @ Xb) @ Xb.T @ W @ y_train
    return xb @ theta

def lwr_model(df):
    X = df[["PM10","NO2","SO2","CO","O3"]].to_numpy()
    y = df["PM2.5"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    rows = []
    for tau in [0.1,0.5,1,2,5]:
        start = time.perf_counter()
        preds = [lwr_predict(X_train, y_train, x, tau) for x in X_test]
        elapsed = time.perf_counter() - start
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Tau={tau}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        rows.append([tau, mae, rmse, elapsed])

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    print("Linear Regression MAE:", mean_absolute_error(y_test, pred_lr))
    print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
    pd.DataFrame(rows, columns=["Tau","MAE","RMSE","Time_sec"]).to_csv("results/lwr_results.csv", index=False)

def mlp_from_scratch(X_train, X_test, y_train, y_test):
    classes = sorted(y_train.unique())
    class_to_i = {c:i for i,c in enumerate(classes)}
    Y = np.zeros((len(y_train), len(classes)))
    for i,c in enumerate(y_train):
        Y[i, class_to_i[c]] = 1

    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    input_size, hidden_size, output_size = Xtr.shape[1], 15, len(classes)

    def sigmoid(x):
        return 1/(1+np.exp(-np.clip(x,-500,500)))
    def softmax(x):
        e = np.exp(x-np.max(x,axis=1,keepdims=True))
        return e/np.sum(e,axis=1,keepdims=True)

    W1 = np.random.randn(input_size, hidden_size)*0.01
    b1 = np.zeros((1,hidden_size))
    W2 = np.random.randn(hidden_size, output_size)*0.01
    b2 = np.zeros((1,output_size))

    lr, epochs = 0.01, 500
    for epoch in range(epochs):
        Z1 = Xtr @ W1 + b1
        A1 = sigmoid(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)
        error2 = A2 - Y
        dW2 = A1.T @ error2 / len(Xtr)
        db2 = np.mean(error2,axis=0,keepdims=True)
        dA1 = error2 @ W2.T
        dZ1 = dA1 * A1 * (1-A1)
        dW1 = Xtr.T @ dZ1 / len(Xtr)
        db1 = np.mean(dZ1,axis=0,keepdims=True)
        W2 -= lr*dW2; b2 -= lr*db2
        W1 -= lr*dW1; b1 -= lr*db1

    pred_prob = softmax(sigmoid(Xte @ W1 + b1) @ W2 + b2)
    pred = np.array([classes[i] for i in np.argmax(pred_prob,axis=1)])
    return pred, evaluate_classifier("MLP Back Propagation", y_test, pred)

def genetic_algorithm(X_train, X_val, y_train, y_val):
    population_size, generations = 12, 15
    mutation_rate, lambda_penalty = 0.10, 0.02

    def fitness(chromosome):
        selected = [FEATURES[i] for i,b in enumerate(chromosome) if b]
        if not selected: return -1
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train[selected], y_train)
        pred = model.predict(X_val[selected])
        f1 = f1_score(y_val,pred,average="weighted",zero_division=0)
        return f1 - lambda_penalty*(len(selected)/len(FEATURES))

    population = []
    for _ in range(population_size):
        c = [random.randint(0,1) for _ in FEATURES]
        if sum(c)==0: c[random.randrange(len(FEATURES))]=1
        population.append(c)

    for _ in range(generations):
        scores = [fitness(c) for c in population]
        order = np.argsort(scores)[::-1]
        selected = [population[i] for i in order[:4]]
        new_population = selected.copy()
        while len(new_population)<population_size:
            p1,p2=random.choice(selected),random.choice(selected)
            point=random.randint(1,len(FEATURES)-1)
            child=p1[:point]+p2[point:]
            if random.random()<mutation_rate:
                m=random.randrange(len(FEATURES)); child[m]=1-child[m]
            if sum(child)==0: child[random.randrange(len(FEATURES))]=1
            new_population.append(child)
        population=new_population

    best=max(population,key=fitness)
    selected_features=[FEATURES[i] for i,b in enumerate(best) if b]
    print("\nGA best chromosome:", ''.join(map(str,best)))
    print("GA selected features:", selected_features)
    print("GA fitness:", fitness(best))

    model=DecisionTreeClassifier(max_depth=5,random_state=42)
    model.fit(X_train[selected_features],y_train)
    return model, selected_features, fitness(best)

def warning_message(label):
    if label=="Good": return "NORMAL: Air quality is good."
    if label=="Moderate": return "WATCH: Air quality is acceptable; continue monitoring."
    if label=="Poor": return "CAUTION: Pollution is high. Sensitive groups should follow local health guidance."
    return "HIGH ALERT: Potentially hazardous air quality. Follow official environmental/public-health guidance."

def main():
    df=load_data()
    preprocessing_eda(df)
    candidate_elimination_demo()

    X_train,X_val,X_test,y_train,y_val,y_test,X_train_s,X_val_s,X_test_s=split_data(df)

    tree_model,tree_pred,tree_row=decision_tree(X_train,X_test,y_train,y_test)
    bayes_model,bayes_pred,bayes_row=bayesian(X_train,X_test,y_train,y_test)
    knn_pred,knn_row,best_k=knn_model(X_train_s,X_test_s,y_train,y_test)
    lwr_model(df)
    mlp_pred,mlp_row=mlp_from_scratch(X_train_s,X_test_s,y_train,y_test)

    ga_model,selected_features,ga_fit=genetic_algorithm(X_train,X_val,y_train,y_val)
    ga_pred=ga_model.predict(X_test[selected_features])
    ga_row=evaluate_classifier("GA + Decision Tree",y_test,ga_pred)

    rows=[tree_row,bayes_row,knn_row,mlp_row,ga_row]
    pd.DataFrame(rows).to_csv("results/model_comparison.csv",index=False)

    # Confusion matrices for each classification model
    for name,pred in [
        ("decision_tree",tree_pred),
        ("naive_bayes",bayes_pred),
        ("knn",knn_pred),
        ("mlp",mlp_pred),
        ("ga_decision_tree",ga_pred)
    ]:
        cm=confusion_matrix(y_test,pred,labels=["Good","Moderate","Poor","Hazardous"])
        plt.figure(figsize=(7,6))
        plt.imshow(cm); plt.colorbar()
        plt.xticks(np.arange(4),["Good","Moderate","Poor","Hazardous"],rotation=45)
        plt.yticks(np.arange(4),["Good","Moderate","Poor","Hazardous"])
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout(); plt.savefig(f"results/figures/cm_{name}.png"); plt.close()

    # Integrated warning example
    new_data=pd.DataFrame([[120,220,80,30,1.2,25,33,75,1.5,1008]],columns=FEATURES)
    label=tree_model.predict(new_data)[0]
    print("\n=== INTEGRATED WARNING SYSTEM ===")
    print("Predicted Air Quality:",label)
    print(warning_message(label))
    print("\nResults saved in results/")

if __name__=="__main__":
    main()
