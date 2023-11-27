import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Muat dataset


@st.cache
def load_data():
    df = pd.read_csv('penguins_cleaned.csv')
    df = df.dropna()
    le_species = LabelEncoder()
    le_island = LabelEncoder()
    le_sex = LabelEncoder()
    df['species'] = le_species.fit_transform(df['species'])
    df['island'] = le_island.fit_transform(df['island'])
    df['sex'] = le_sex.fit_transform(df['sex'])
    return df, le_species, le_island, le_sex

# Membuat model


def create_model(df):
    X = df[['island', 'bill_length_mm', 'bill_depth_mm',
            'flipper_length_mm', 'body_mass_g', 'sex']]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Main app


def main():
    st.title('Aplikasi Streamlit untuk Dataset Palmer Penguins')
    df, le_species, le_island, le_sex = load_data()
    model = create_model(df)
    st.write('Model telah dilatih dan siap digunakan untuk prediksi.')

    st.subheader('Prediksi Spesies Penguin')
    island = st.selectbox('Pilih Pulau', le_island.classes_)
    bill_length = st.slider('Panjang Paruh (mm)', float(
        df['bill_length_mm'].min()), float(df['bill_length_mm'].max()))
    bill_depth = st.slider('Kedalaman Paruh (mm)', float(
        df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
    flipper_length = st.slider('Panjang Flipper (mm)', float(
        df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
    body_mass = st.slider('Massa Tubuh (g)', float(
        df['body_mass_g'].min()), float(df['body_mass_g'].max()))
    sex = st.selectbox('Jenis Kelamin', le_sex.classes_)

    inputs = [[le_island.transform([island])[
        0], bill_length, bill_depth, flipper_length, body_mass, le_sex.transform([sex])[0]]]
    species = model.predict(inputs)
    st.write(
        f'Spesies penguin diprediksi adalah: {le_species.classes_[species[0]]}')


if __name__ == "__main__":
    main()
