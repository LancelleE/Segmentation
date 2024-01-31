from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


class RawData:
    def __init__(self, dataset):
        self.data = dataset
        self.data_standardized = pd.DataFrame()
        self.data_hot_encoded = pd.DataFrame()
        self.final_dataset = pd.DataFrame()


    def standardize_numerical_values(self):
        df_num = self.data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        if not df_num.empty:
            scaler = StandardScaler()
            self.data_standardized = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)

    def encode_qualitative(self):
        df_char = self.data.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        if not df_char.empty:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.data_hot_encoded = pd.DataFrame(encoder.fit_transform(df_char),
                                                 columns=encoder.get_feature_names_out(df_char.columns))


    def merge_datasets(self):
        self.final_dataset = pd.concat([self.data_standardized, self.data_hot_encoded], axis=1, sort=False)