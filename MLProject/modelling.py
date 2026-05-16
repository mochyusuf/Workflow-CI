import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    mlflow.sklearn.autolog()

    # Load Data
    df = pd.read_csv("data_rumah_preprocessing.csv")

    # Split
    X = df[["Luas Bangunan","Luas Tanah", "Kamar Tidur", "Kamar Mandi", "Garasi"]].values
    y = df["HARGA"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="Data_Rumah_RandomForest_MLProject") as run:
        print(f"🎯 MLflow Run ID: {run.info.run_id}")
    
        mlflow.sklearn.autolog()
    
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
        # --- Train
        model.fit(X_train, y_train)
    
        # --- Predict
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"✅ RMSE: {rmse:.4f}")
        
        mlflow.sklearn.log_model(model, "model", registered_model_name="data_rumah_model")
    
    # Model otomatis terekam via autolog
    print("🚀 Training & Logging selesai.")