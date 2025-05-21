import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_performance_comparison():
    """過去バージョンのモデルと比較して性能劣化がないか検証"""
    # 現在のモデルをロード
    if not os.path.exists(MODEL_PATH):
        pytest.skip("現在のモデルファイルが存在しないためスキップします")

    with open(MODEL_PATH, "rb") as f:
        current_model = pickle.load(f)

    # 過去バージョンのモデルのパス
    OLD_MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model_previous.pkl")

    # 過去バージョンのモデルが存在しない場合は現在のモデルをコピーしてスキップ
    if not os.path.exists(OLD_MODEL_PATH):
        # テスト目的でモデルをコピー保存（実際の運用ではこのロジックは変更が必要）
        with open(OLD_MODEL_PATH, "wb") as f:
            pickle.dump(current_model, f)
        pytest.skip("過去バージョンのモデルが存在しないためスキップします")

    # 過去バージョンのモデルをロード
    with open(OLD_MODEL_PATH, "rb") as f:
        previous_model = pickle.load(f)

    # テストデータを準備
    data = pd.read_csv(DATA_PATH)
    X = data.drop("Survived", axis=1)
    y = data["Survived"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 両方のモデルで予測
    current_pred = current_model.predict(X_test)
    previous_pred = previous_model.predict(X_test)

    # 性能指標を計算
    current_accuracy = accuracy_score(y_test, current_pred)
    previous_accuracy = accuracy_score(y_test, previous_pred)

    # 現在のモデルが過去のモデルより悪くないことを確認
    assert (
        current_accuracy >= previous_accuracy * 0.95
    ), f"モデルの性能が著しく低下しています: 現在={current_accuracy:.4f}, 過去={previous_accuracy:.4f}"

    # 推論時間の比較（オプション）
    start_time = time.time()
    current_model.predict(X_test)
    current_inference_time = time.time() - start_time

    start_time = time.time()
    previous_model.predict(X_test)
    previous_inference_time = time.time() - start_time

    # 推論時間が著しく悪化していないか確認（例: 1.5倍以上遅くなっていないか）
    assert (
        current_inference_time <= previous_inference_time * 1.5
    ), f"推論時間が著しく悪化しています: 現在={current_inference_time:.4f}秒, 過去={previous_inference_time:.4f}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"
