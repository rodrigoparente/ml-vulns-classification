# third-party imports
from ast import literal_eval
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

# project imports
from constants import LABELS_DICT


def load_data(filepath):
    data = pd.read_csv(filepath)

    # droping unused columns
    data.drop(columns=[
        'cve_id', 'readable_cve_date', 'reference',
        'readable_exploit_date', 'audience_normalized'], inplace=True)

    # replacing space with underscore in part column
    data['part'].replace(' ', '_', regex=True, inplace=True)

    # casting dates to days
    columns = ['cve_published_date', 'exploit_published_date']
    for column in columns:
        data[column] =\
            (pd.to_datetime('today') - pd.to_datetime(data[column])).dt.days

        # replacing nan with 0
        data.loc[data[column].isnull(), column] = 0
        data[column] = data[column].astype(int)

    # replacing nan values in exploit
    # and audience columns to 0
    columns = ['exploit_count', 'audience']
    for column in columns:
        data.loc[data[column].isnull(), column] = 0
        data[column] = data[column].astype(int)

    # replacing nan values in google_trend column
    data.loc[data['google_trend'].isnull(), 'google_trend'] = 'none'

    # casting upper values to lower
    columns =\
        ['confidentiality_impact', 'integrity_impact', 'availability_impact',
         'google_trend', 'topology', 'asset_type', 'environment']
    for column in columns:
        data[column] = data[column].str.lower()

    # replacing attack_type space with underscore
    data['attack_type'].replace(', ', ',', regex=True, inplace=True)
    data['attack_type'].replace('-', '_', regex=True, inplace=True)
    data['attack_type'].replace(' ', '_', regex=True, inplace=True)

    # replacing attack_type nan value
    data.loc[data['attack_type'].isnull(), 'attack_type'] = "['none']"

    # casting attack_type string to array
    data['attack_type'] = data['attack_type'].apply(literal_eval)

    # manually encoding columns value
    data['topology'].replace({'local': 0, 'dmz': 1}, inplace=True)
    data['asset_type'].replace({'workstation': 0, 'server': 1}, inplace=True)
    data['environment'].replace({'development': 0, 'production': 1}, inplace=True)

    # one-hot-encoding data
    ohe = OneHotEncoder(sparse=False, dtype=int)

    columns = ['part', 'vendor', 'confidentiality_impact',
               'integrity_impact', 'availability_impact', 'google_trend']

    encoder_vars_array = ohe.fit_transform(data[columns])

    # create object for the feature names using the categorical variables
    encoder_feature_names = ohe.get_feature_names_out(columns)

    # create a dataframe to hold the one hot encoded variables
    encoder_vars_df = pd.DataFrame(encoder_vars_array, columns=encoder_feature_names)

    # concatenate the new dataframe back to the original input variables dataframe
    data = pd.concat([data.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis=1)

    # drop the original columns
    data.drop(columns, axis=1, inplace=True)

    # multi-hot-encoding
    mlb = MultiLabelBinarizer()
    mlb.fit(data['attack_type'])

    # creating new columns name
    new_col_names = [f'attack_type_{name}' for name in mlb.classes_]

    # create new dataFrame with transformed/one-hot encoded
    attacks = pd.DataFrame(mlb.fit_transform(data['attack_type']), columns=new_col_names)

    # concat encoded data with original dataframe
    data = pd.concat([data.reset_index(drop=True), attacks.reset_index(drop=True)], axis=1)

    # drop the original column
    data.drop('attack_type', axis=1, inplace=True)

    # adding possible missing attack types
    types_of_attack =\
        ['none', 'remote_code_execution', 'arbitrary_code_execution', 'tampering',
         'denial_of_service', 'spoofing', 'defense_in_depth', 'elevation_of_privilege',
         'security_feature_bypass', 'information_disclosure', 'xss', 'memory_leak',
         'sql_injection', 'zero_day', 'proof_of_concepts']

    # creating new columns
    missing_attack_types = list(set(types_of_attack).difference(mlb.classes_))
    missing_columns = [f'attack_type_{name}' for name in missing_attack_types]

    n_rows = data.shape[0]
    df_dict = dict()

    # prep data
    for column in missing_columns:
        df_dict.setdefault(column, np.zeros(n_rows, dtype=int))

    # concatenating to original dataset
    data = pd.concat(
        [data.reset_index(drop=True), pd.DataFrame(df_dict).reset_index(drop=True)], axis=1)

    # sorting columns
    data = data.reindex(sorted(data.columns), axis=1)

    if 'label' in data.columns:
        data['label'].replace(LABELS_DICT, inplace=True)

        X = data.drop(columns='label').to_numpy()
        y = data['label'].to_numpy()

        return X, y

    # if the dataset does not have labels
    return data.to_numpy(), None
