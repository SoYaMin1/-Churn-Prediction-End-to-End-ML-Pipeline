# src/train.py
"""
Обучение ML-пайплайна для предсказания оттока клиентов.

Что делает скрипт:
1. Загружает и очищает данные (через data_loader.py)
2. Делит выборку на train/val/test со стратификацией
3. Сравнивает 3 модели через кросс-валидацию (LogReg, RF, XGBoost)
4. Подбирает гиперпараметры для лучшей модели через GridSearchCV
5. Оценивает финальную модель на валидации (classification_report, ROC-AUC, confusion matrix)
6. Считает SHAP values для интерпретации
7. Сохраняет готовый пайплайн в models/churn_pipeline.joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.data_loader import load_dataset
from src.preprocessing import get_preprocessor


# ---------- Константы ----------
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.joblib"
RANDOM_STATE = 42


def split_data(X, y):
    """
    Делит данные на train (60%) / validation (20%) / test (20%).
    Стратификация сохраняет пропорцию классов в каждой части.
    """
    # Шаг 1: отделяем test (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Шаг 2: из оставшихся 80% отделяем validation (25% от 80% = 20% от всего)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    print(f"\nРазмеры выборок:")
    print(f"  Train:      {X_train.shape[0]} строк")
    print(f"  Validation: {X_val.shape[0]} строк")
    print(f"  Test:       {X_test.shape[0]} строк")

    return X_train, X_val, X_test, y_train, y_val, y_test


def compare_models(X_train, y_train, preprocessor, numeric_features, categorical_features):
    """
    Сравнивает 3 модели через кросс-валидацию по ROC-AUC.
    Возвращает имя лучшей модели.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ (5-fold CV, метрика ROC-AUC)")
    print("=" * 60)

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        results[name] = scores.mean()
        print(f"  {name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_model_name = max(results, key=results.get)
    print(f"\n>>> Лучшая модель: {best_model_name} "
          f"(ROC-AUC = {results[best_model_name]:.4f})")

    return best_model_name


def get_param_grid(model_name):
    """
    Возвращает сетку гиперпараметров и базовую модель для GridSearchCV.
    """
    if model_name == "XGBoost":
        base_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
        }
    elif model_name == "RandomForest":
        base_model = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
        }
    else:  # LogisticRegression
        base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
        }

    return base_model, param_grid


def tune_hyperparameters(X_train, y_train, preprocessor, best_model_name):
    """
    Подбирает гиперпараметры для лучшей модели через GridSearchCV.
    Возвращает обученный пайплайн с лучшими параметрами.
    """
    base_model, param_grid = get_param_grid(best_model_name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", base_model),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "=" * 60)
    print(f"ПОДБОР ГИПЕРПАРАМЕТРОВ ДЛЯ {best_model_name}")
    print("=" * 60)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"\nЛучшие параметры:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Лучший ROC-AUC на CV: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X, y, dataset_name="Validation"):
    """
    Считает метрики на указанной выборке и печатает отчёт.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("\n" + "=" * 60)
    print(f"МЕТРИКИ НА {dataset_name.upper()} SET")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["No Churn", "Churn"]))

    print(f"ROC-AUC:   {roc_auc_score(y, y_proba):.4f}")
    print(f"F1 (Churn): {f1_score(y, y_pred):.4f}")
    print(f"Precision (Churn): {precision_score(y, y_pred):.4f}")
    print(f"Recall (Churn):    {recall_score(y, y_pred):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"                 Predicted No   Predicted Yes")
    print(f"Actual No        {cm[0][0]:>10d}   {cm[0][1]:>10d}")
    print(f"Actual Yes       {cm[1][0]:>10d}   {cm[1][1]:>10d}")

    return {
        "roc_auc": roc_auc_score(y, y_proba),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
    }


def analyze_with_shap(model, X_sample, max_samples=200):
    """
    Считает SHAP values и печатает топ-10 самых важных признаков.
    Для полной визуализации нужен matplotlib + сохранение графика.
    """
    print("\n" + "=" * 60)
    print("ИНТЕРПРЕТАЦИЯ МОДЕЛИ (SHAP)")
    print("=" * 60)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Трансформируем признаки через препроцессор
    X_transformed = preprocessor.transform(X_sample[:max_samples])

    # Имена признаков после OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()

    # Для tree-based моделей используем TreeExplainer
    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)

        # Средняя абсолютная величина SHAP по всем样本 = важность признака
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).head(10)

        print("\nТоп-10 признаков по SHAP importance:")
        for i, row in importance_df.iterrows():
            print(f"  {row['feature']:50s} {row['mean_abs_shap']:.4f}")
    else:
        # Для линейных моделей — LinearExplainer
        explainer = shap.LinearExplainer(classifier, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).head(10)

        print("\nТоп-10 признаков по SHAP importance:")
        for i, row in importance_df.iterrows():
            print(f"  {row['feature']:50s} {row['mean_abs_shap']:.4f}")

    return importance_df


def train_pipeline():
    """
    Главная функция: полный цикл обучения.
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ ПАЙПЛАЙНА CHURN PREDICTION")
    print("=" * 60)

    # ---------- 1. Загрузка данных ----------
    print("\n[1/7] Загрузка данных...")
    X, y, numeric_features, categorical_features = load_dataset(DATA_PATH)

    # ---------- 2. Разделение на train/val/test ----------
    print("\n[2/7] Разделение на train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ---------- 3. Препроцессор ----------
    print("\n[3/7] Создание препроцессора...")
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    print(f"  Числовые признаки: {len(numeric_features)}")
    print(f"  Категориальные признаки: {len(categorical_features)}")

    # ---------- 4. Сравнение моделей ----------
    print("\n[4/7] Сравнение моделей...")
    best_model_name = compare_models(
        X_train, y_train, preprocessor, numeric_features, categorical_features
    )

    # ---------- 5. Подбор гиперпараметров ----------
    print("\n[5/7] Подбор гиперпараметров...")
    best_pipeline = tune_hyperparameters(
        X_train, y_train, preprocessor, best_model_name
    )

    # ---------- 6. Оценка на validation ----------
    print("\n[6/7] Оценка на validation set...")
    val_metrics = evaluate_model(best_pipeline, X_val, y_val, "Validation")

    # ---------- 7. Интерпретация через SHAP ----------
    print("\n[7/7] Интерпретация модели через SHAP...")
    try:
        analyze_with_shap(best_pipeline, X_val, max_samples=200)
    except Exception as e:
        print(f"  SHAP-анализ пропущен из-за ошибки: {e}")

    # ---------- Финальная оценка на TEST (не трогали до этого момента) ----------
    print("\n" + "#" * 60)
    print("# ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET (отложенная выборка)")
    print("#" * 60)
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, "Test")

    # ---------- Сохранение модели ----------
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nМодель сохранена: {MODEL_PATH}")

    # ---------- Итоговая сводка ----------
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 60)
    print(f"Лучшая модель: {best_model_name}")
    print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"Test ROC-AUC:       {test_metrics['roc_auc']:.4f}")
    print(f"Test F1 (Churn):    {test_metrics['f1']:.4f}")

    return best_pipeline, test_metrics


if __name__ == "__main__":
    train_pipeline()