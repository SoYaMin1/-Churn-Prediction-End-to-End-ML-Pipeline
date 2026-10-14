# src/data_loader.py
import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Загружает сырой датасет из CSV файла.
    
    Args:
        filepath: Путь к CSV файлу с данными.
    
    Returns:
        DataFrame с сырыми данными.
    
    Raises:
        FileNotFoundError: Если файл не найден.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Файл не найден: {filepath}\n"
            f"Скачайте датасет с Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"и положите его в папку data/"
        )
    
    df = pd.read_csv(filepath)
    print(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая очистка данных:
    - Удаление ID клиента (не несет предиктивной силы)
    - Преобразование целевой переменной Churn в бинарный формат
    - Приведение TotalCharges к числовому типу
    
    Args:
        df: Сырой DataFrame.
    
    Returns:
        Очищенный DataFrame.
    """
    df = df.copy()
    
    # 1. Удаляем customerID — уникальный идентификатор, бесполезен для модели
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # 2. Целевая переменная: Yes -> 1, No -> 0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 3. TotalCharges часто приходит как object с пробелами.
    #    Преобразуем в float, ошибки (пробелы) станут NaN.
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Для новых клиентов с tenure=0 логично заполнить 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 4. Удаляем дубликаты, если есть
    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        print(f"Удалено {n_before - len(df)} дубликатов")
    
    print(f"После очистки: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def get_feature_types(df: pd.DataFrame, target_col: str = 'Churn'):
    """
    Автоматически определяет, какие признаки числовые, а какие категориальные.
    Это удобнее, чем хардкодить списки — при изменении датасета код не сломается.
    
    Args:
        df: DataFrame с признаками и целевой переменной.
        target_col: Название целевой колонки (исключается из признаков).
    
    Returns:
        Кортеж (numeric_features, categorical_features).
    """
    feature_df = df.drop(columns=[target_col])
    
    numeric_features = feature_df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    
    categorical_features = feature_df.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist()
    
    return numeric_features, categorical_features


def load_dataset(filepath: str, target_col: str = 'Churn'):
    """
    Удобная обертка: загружает, очищает и разделяет признаки по типам.
    Именно эту функцию будет вызывать train.py.
    
    Returns:
        X: DataFrame с признаками
        y: Series с целевой переменной
        numeric_features: список числовых колонок
        categorical_features: список категориальных колонок
    """
    df = load_raw_data(filepath)
    df = clean_data(df)
    
    numeric_features, categorical_features = get_feature_types(df, target_col)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"\nЧисловые признаки ({len(numeric_features)}): {numeric_features}")
    print(f"Категориальные признаки ({len(categorical_features)}): {categorical_features}")
    print(f"\nРаспределение целевой переменной:\n{y.value_counts(normalize=True)}")
    
    return X, y, numeric_features, categorical_features


if __name__ == "__main__":
    # Быстрый тест: запускается только при прямом вызове файла
    X, y, num, cat = load_dataset('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nПример данных:\n{X.head()}")