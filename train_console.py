import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from house_price_prediction.core.data_loader import load_dataset, split_features_target
from house_price_prediction.core.model_utils import train_linear_regression, model_summary, save_model_zip


def main():
    parser = argparse.ArgumentParser(description="Train Linear Regression for House Price Prediction")
    parser.add_argument("--data", required=True, help="Path to dataset CSV (USA Housing style)")
    parser.add_argument("--out", default="house_price_prediction/models/house_price_model.zip", help="Output zip path for saved model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    args = parser.parse_args()

    df = load_dataset(args.data)

    # In ra head và describe theo đúng hướng dẫn
    print("\n=== Head (5 dòng đầu) ===")
    print(df.head())
    print("\n=== Describe ===")
    print(df.describe())

    # Vẽ heatmap tương quan và lưu ảnh
    os.makedirs("house_price_prediction/images", exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="magma")
    plt.tight_layout()
    corr_path = "house_price_prediction/images/correlation.png"
    plt.savefig(corr_path)
    plt.close()
    print(f"\nHeatmap correlation saved: {os.path.abspath(corr_path)}")
    X, y = split_features_target(df)
    model, (mae, mse, rmse), (X_train, X_test, y_train, y_test, y_pred) = train_linear_regression(X, y, test_size=args.test_size)

    print("\n=== Model Summary ===")
    print(model_summary(model, list(X.columns)))
    # In coefficients dạng DataFrame giống tài liệu
    coef_df = pd.DataFrame(model.coef_, list(X.columns), columns=['Coefficient'])
    print("\nCoefficients DataFrame:")
    print(coef_df)
    print("\n=== Evaluation ===")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # ensure output dir exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_model_zip(model, list(X.columns), args.out)
    print(f"\nModel saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()