import streamlit as st
import pandas as pd
from back.data import RawData

# Function to handle computation
def perform_computation(df, choix_colonnes, toggle_standardise, toggle_hot_encoding):
    if not choix_colonnes:
        # If no features selected, return the original dataframe
        return df

    filtered_df = df.filter(choix_colonnes)
    data_object = RawData(filtered_df)

    if toggle_standardise:
        data_object.standardize_numerical_values()

    if toggle_hot_encoding:
        data_object.encode_qualitative()

    # Check if neither standardization nor hot encoding is selected
    if not toggle_standardise and not toggle_hot_encoding:
        return filtered_df

    data_object.merge_datasets()
    return data_object.final_dataset


# Streamlit app
st.sidebar.title("Feature Analysis App")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'txt'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.markdown("**Select Features:**")
    options = list(df.columns)
    default = options
    choix_colonnes = st.sidebar.multiselect('Features to include in analysis', options=options, default=default)

    toggle_standardise = st.sidebar.checkbox('Standardise?', value=True)
    toggle_hot_encoding = st.sidebar.checkbox('Hot encoding?', value=True)

    compute_button = st.sidebar.button('Compute')

    if compute_button:
        with st.spinner("Computing..."):
            final_dataset = perform_computation(df, choix_colonnes, toggle_standardise, toggle_hot_encoding)
            st.success("Computation complete!")
            st.write(final_dataset)
