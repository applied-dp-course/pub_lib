import numpy as np
import pandas as pd

import anonypy


def generate_synthetic_data(features_dict, size, seed=None):
    rng = np.random.default_rng(seed)
    data = {}
    for feature, options in features_dict.items():
        data[feature] = np.array(rng.choice(options, size))
    data = pd.DataFrame(data)
    for col in data:
        data[col] = data[col].astype('category')
    return pd.DataFrame(data)


# Used by notebook 1
class Anonymizer:
    def __init__(self, k, sensitive_attribute, data):
        self.k = k
        self.sensitive_attribute = sensitive_attribute
        # Ensure categorical dtype for non-sensitive features to work with anonypy spans
        df = data.copy()
        for col in df.columns:
            if col != sensitive_attribute and df[col].dtype.name != 'category':
                df[col] = df[col].astype('category')
        self.__raw_data = df

    def build_anonymized_sub_datasets(self):
        sub_dfs = []
        features = [f for f in self.__raw_data.columns if f != self.sensitive_attribute]

        for i, a in enumerate(features):
            rest = features[:i] + features[i + 1 :]
            anonymized_sub_dataset = self.__anonymize(rest)
            anonymized_sub_dataset[features[i]] = self.__generalize_col(features[i]) * len(
                anonymized_sub_dataset
            )
            anonymized_sub_dataset = anonymized_sub_dataset[sorted(anonymized_sub_dataset.columns)]
            anonymized_sub_dataset = self.__repeat_instead_of_count(anonymized_sub_dataset)
            anonymized_sub_dataset = anonymized_sub_dataset.map(
                lambda x: tuple(x) if isinstance(x, list) else x
            )
            sub_dfs.append(anonymized_sub_dataset)
        return sub_dfs

    def __repeat_instead_of_count(self, df):
        repeats_df = df.loc[df.index.repeat(df['count'])]
        repeats_df = repeats_df.drop(columns=['count']).reset_index(drop=True)
        return repeats_df

    def __generalize_col(self, col_name):
        unique = self.__raw_data[col_name].unique().tolist()
        return [unique]

    def __expand_cells(self, cell):
        if isinstance(cell, (list, tuple)) and len(cell) > 0:
            if isinstance(cell[0], str) and ',' in cell[0]:
                return cell[0].split(',')
        return cell

    def __anonymize(self, cols_to_keep):
        p = anonypy.Preserver(self.__raw_data, cols_to_keep, self.sensitive_attribute)
        rows = pd.DataFrame(p.anonymize_k_anonymity(k=self.k))
        rows[self.sensitive_attribute] = rows[self.sensitive_attribute].apply(lambda x: [x])
        return rows.map(self.__expand_cells)
