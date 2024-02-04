from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


class RawData:
    def __init__(self, dataset):
        self.data = dataset
        self.data_standardized = pd.DataFrame(dataset.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']))
        self.data_hot_encoded = pd.DataFrame(dataset.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']))
        self.final_dataset = pd.DataFrame(dataset)

    def filter_data(self, exclude_columns):
        if exclude_columns:
            keep_cols = [col for col in self.data if col not in exclude_columns]
            self.data_filtered = self.data[keep_cols]
        else:
            self.data_filtered = self.data


    def standardize_numerical_values(self):
        df_num = self.data_filtered.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        if not df_num.empty:
            scaler = StandardScaler()
            self.data_standardized = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)

    def encode_qualitative(self, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = []
        df_char = self.data_filtered.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        
        # Exclude specified columns from one-hot encoding
        columns_to_encode = [col for col in df_char.columns if col not in exclude_columns]
        if len(columns_to_encode) > 0:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = encoder.fit_transform(df_char[columns_to_encode])

            # Create a DataFrame with the encoded features and concatenate it with the excluded columns
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))
            self.data_hot_encoded = pd.concat([encoded_df, df_char[exclude_columns]], axis=1, sort=False)

            



    def merge_datasets(self):
        self.final_dataset = pd.concat([self.data_standardized, self.data_hot_encoded], axis=1, sort=False)