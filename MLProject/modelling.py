import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telco Churn Prediction Basic")

DATA_PATH = 'dataset_preprocessing/data_clean.csv'

print("Memulai proses training model...")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset berhasil dimuat dari '{DATA_PATH}'.")
except FileNotFoundError:
    print(f"Error: File '{DATA_PATH}' tidak ditemukan.")
    exit()

print("\nKolom dalam dataset:")
print(df.columns)

if 'Churn_Yes' in df.columns:
    target_col = 'Churn_Yes'

elif 'Churn' in df.columns:
    target_col = 'Churn'
    
    # convert jika masih Yes/No
    if df[target_col].dtype == 'object':
        print("Mengubah label Yes/No menjadi 1/0...")
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

else:
    raise ValueError("Kolom target tidak ditemukan. Cek dataset preprocessing kamu.")

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Data berhasil dibagi.")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="telco-logistic-regression"):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("\nTraining selesai dan tercatat di MLflow.")
    print(f"Akurasi: {accuracy:.4f}")