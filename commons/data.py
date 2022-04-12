# third-party imports
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    data = pd.read_csv(filepath)

    # droping unused columns
    data.drop(columns=['ID', 'baseSeverity', 'publishedDate', 'exploitDate'], inplace=True)

    # replacing nan with none/0/True/False
    for column, dtype in zip(data.columns, data.dtypes):
        if dtype == object:
            data[column] = data[column].str.lower()
            data.loc[data[column].isnull(), column] = 'none'
        elif dtype == float:
            data.loc[data[column].isnull(), column] = 0
        elif dtype == bool:
            data[column] = data[column].astype(int)

    if 'label' in data.columns:
        data['label'].replace({'low': 0, 'medium': 1, 'high': 2, 'critical': 3}, inplace=True)

        # label encoding categorical columns
        categorical_features = data.columns[data.dtypes == object].tolist()
        for feature in categorical_features:
            data[feature] = LabelEncoder().fit_transform(data[feature])

        X = data.drop(columns='label').to_numpy()
        y = data['label'].to_numpy()

        return X, y

    # label encoding categorical columns
    categorical_features = data.columns[data.dtypes == object].tolist()
    for feature in categorical_features:
        data[feature] = LabelEncoder().fit_transform(data[feature])

    return data.to_numpy(), None
