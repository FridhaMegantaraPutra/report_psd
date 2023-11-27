import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump
from sklearn.preprocessing import StandardScaler
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import skew, kurtosis, mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

import pyaudio
import wave
import numpy as np
import soundfile as sf


url = 'https://media.istockphoto.com/id/1326639660/vector/young-woman-choice-different-emotions-girl-play-role-change-emotion-control-reaction-vector.jpg?s=170667a&w=0&k=20&c=SOJ_Ci0xk1XeLzCrVxyKyb5_rlZqtK6sWAqoAFSpjtQ='
st.image(url, caption='', use_column_width=True, width=100)

# Judul aplikasi
st.title("Speech Emotion Recognition")

# Deskripsi
st.write("Aplikasi ini memungkinkan Anda mengunggah file audio dan melakukan klasifikasi menggunakan PCA dan K-Nearest Neighbors (KNN).")


def hitung_statistik(audio_file):
    zcr_data = pd.DataFrame(columns=[
                            'ZCR Mean', 'ZCR Median', 'ZCR Std Deviasi', 'ZCR Skewness', 'ZCR Kurtosis'])
    y, sr = librosa.load(audio_file)
    mean = np.mean(y)
    std_dev = np.std(y)
    max_value = np.max(y)
    min_value = np.min(y)
    median = np.median(y)
    skewness = skew(y)  # Calculate skewness
    kurt = kurtosis(y)  # Calculate kurtosis
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    mode_value, _ = mode(y)  # Calculate mode
    iqr = q3 - q1

    # Hitung ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    # Hitung rata-rata ZCR
    mean_zcr = zcr.mean()
    # Hitung nilai median ZCR
    median_zcr = np.median(zcr)
    # Hitung nilai std deviasa ZCR
    std_dev_zcr = np.std(zcr)
    # Hitung skewness ZCR
    skewness_zcr = stats.skew(zcr, axis=None)
    # Hitung kurtosis ZCR
    kurtosis_zcr = stats.kurtosis(zcr, axis=None)

    # Hitung RMS
    rms = librosa.feature.rms(y=y)
    # Hitung rata-rata RMS
    mean_rms = rms.mean()
    # Hitung nilai median RMS
    median_rms = np.median(rms)
    # Hitung nilai std deviasa RMS
    std_dev_rms = np.std(rms)
    # Hitung skewness RMS
    skewness_rms = stats.skew(rms, axis=None)
    # Hitung kurtosis RMS
    kurtosis_rms = stats.kurtosis(rms, axis=None)

    # Tambahkan data ke DataFrame
    # return[mean, std_dev, max_value, min_value, median, skewness, kurt, q1, q3, mode_value, iqr, mean_zcr, median_zcr, std_dev_zcr, skewness_zcr, kurtosis_zcr, mean_rms, median_rms,std_dev_rms, skewness_rms, kurtosis_rms]
    zcr_data = zcr_data._append({'ZCR Mean': mean_zcr, 'ZCR Median': median_zcr,
                                'ZCR Std Deviasi': std_dev_zcr, 'ZCR Skewness': skewness_zcr, 'ZCR Kurtosis': kurtosis_zcr, }, ignore_index=True)
    return zcr_data


dataknn = pd.read_csv('hasil_zcr_rms.csv')
# Ganti 'target_column' dengan nama kolom target
X = dataknn.drop(['Label', 'File'], axis=1)
y = dataknn['Label']
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size=0.2)

selected_option = st.selectbox('pilih dari', ['file'])


if selected_option == 'file':
    st.header("Upload an Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file...", type=["wav", "mp3"])
    # Memuat data audio (misalnya: fitur audio dari file WAV)
    # Di sini, Anda harus mengganti bagian ini dengan kode yang sesuai untuk membaca dan mengambil fitur-fitur audio.
    # Misalnya, jika Anda menggunakan pustaka librosa, Anda dapat menggunakannya untuk mengambil fitur-fitur audio.
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Deteksi Audio"):
        # Simpan file audio yang diunggah
        audio_path = "audio.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Hanya contoh data dummy (harap diganti dengan pengambilan data yang sesungguhnya)
        data_mentah = hitung_statistik(audio_path)
        data_mentah.to_csv('hasil_zcr_rms1.csv', index=False)
        # Standarisasi fitur (opsional, tapi dapat meningkatkan kinerja PCA dan KNN)
        kolom = ['ZCR Mean', 'ZCR Median', 'ZCR Std Deviasi',
                 'ZCR Kurtosis', 'ZCR Skewness']

        # --------- scaler -------------------------

        # # Inisialisasi StandardScaler
        # scaler = StandardScaler()
        # scaler.fit(X_train[kolom])
        # X_train_scaled = scaler.transform(X_train[kolom])
        # # Lakukan standarisasi pada kolom yang telah ditentukan
        # data_ternormalisasi = scaler.transform(data_mentah[kolom])

        with open('scaler5_fitur.pkl', 'rb') as standarisasi:
            loadscal = pickle.load(standarisasi)

        # Inisialisasi StandardScaler

        scaler = loadscal
        scaler.fit(X_train[kolom])
        X_train_scaled = scaler.transform(X_train[kolom])
        # Lakukan standarisasi pada kolom yang telah ditentukan
        data_ternormalisasi = scaler.transform(data_mentah[kolom])

        # with open("MinMaxScaler.pkl", "rb") as minmaxs:
        #     loadminmax = pickle.load(minmaxs)

        # loadminmax.fit(X_train[kolom])
        # data_MinimMaxim = loadminmax.transform(data_mentah[kolom])

        # Reduksi dimensi menggunakan PCA
        with open('sklearn_pca.pkl', 'rb') as reduk:
            loadpca = pickle.load(reduk)

        X_train_pca = loadpca.fit_transform(X_train_scaled)
        X_pca = loadpca.transform(data_ternormalisasi)

        # --------- INPORT PICKLE -------------------

        # untuk k 13 error di data set 212 dan jadi disgust

        with open('modelknn5_fitur.pkl', 'rb') as knn:
            loadknn = pickle.load(knn)
        loadknn.fit(X_train_scaled, y_train)

        # Inisialisasi KNN Classifier
        loadknn.fit(X_train_pca, y_train)
        y_prediksi = loadknn.predict(X_pca)

        # classifier = KNeighborsClassifier(n_neighbors=2)
        # classifier.fit(X_train_pca, y_train)
        # y_prediksi = classifier.predict(X_pca)

        # Menampilkan hasil klasifikasi
        st.write("Hasil Klasifikasi:")
        st.write(y_prediksi)
# ------------------------ jika suara live --------------------------------------------------
# ------------------------------------------------------------------------------------------
# elif selected_option == 'suara':
#     st.title("Rekam Audio dan Simpan ke File WAV (2 detik)")


#     if st.button("Mulai Rekaman"):
#             st.text("Sedang merekam...")

#             # Mulai merekam audio selama 2 detik ke dalam buffer
#             duration = 6  # Durasi rekaman dalam detik
#             sample_rate = 44100  # Frekuensi sampel
#             chunk = 1024  # Ukuran chunk untuk pembacaan audio
#             audio_data = []

#             p = pyaudio.PyAudio()

#             stream = p.open(format=pyaudio.paInt16,
#                             channels=1,
#                             rate=sample_rate,
#                             input=True,
#                             frames_per_buffer=chunk)

#             for i in range(0, int(sample_rate / chunk * duration)):
#                 audio_chunk = stream.read(chunk)
#                 audio_data.append(np.frombuffer(audio_chunk, dtype=np.int16))

#             stream.stop_stream()
#             stream.close()
#             p.terminate()

#             st.text("Rekaman Selesai")

#             # Simpan rekaman ke dalam file WAV
#             audio_filename = "rekaman_audio.wav"
#             wf = wave.open(audio_filename, 'wb')
#             wf.setnchannels(1)
#             wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
#             wf.setframerate(sample_rate)
#             wf.writeframes(b''.join(audio_data))
#             wf.close()


#     # if st.button("Mulai Rekaman"):
#     #     st.text("Sedang merekam...")

#     #     # Mulai merekam audio selama 2 detik ke dalam buffer
#     #     duration = 2  # Durasi rekaman dalam detik
#     #     sample_rate = 44100  # Frekuensi sampel
#     #     audio_data = sd.rec(int(duration * sample_rate),
#     #                         samplerate=sample_rate, channels=1, dtype='int16')
#     #     sd.wait()

#     #     st.text("Rekaman Selesai")

#     #     # Simpan rekaman ke dalam variabel Python
#     #     audio_buffer = audio_data.flatten()

#     #     # Simpan rekaman audio sebagai file WAV
#     #     audio_filename = "rekaman_audio.wav"
#     #     sf.write(audio_filename, audio_buffer, sample_rate)

#         # Hanya contoh data dummy (harap diganti dengan pengambilan data yang sesungguhnya)
#             data_mentah = hitung_statistik(audio_filename)
#             data_mentah.to_csv('hasil_zcr_rms2_live.csv', index=False)
#             # Standarisasi fitur (opsional, tapi dapat meningkatkan kinerja PCA dan KNN)
#             kolom = ['ZCR Mean', 'ZCR Median', 'ZCR Std Deviasi',
#                     'ZCR Kurtosis', 'ZCR Skewness']

#             # --------- scaler -------------------------

#             # # Inisialisasi StandardScaler
#             # scaler = StandardScaler()
#             # scaler.fit(X_train[kolom])
#             # X_train_scaled = scaler.transform(X_train[kolom])
#             # # Lakukan standarisasi pada kolom yang telah ditentukan
#             # data_ternormalisasi = scaler.transform(data_mentah[kolom])

#             with open('scaler.pkl', 'rb') as standarisasi:
#                 loadscal = pickle.load(standarisasi)

#             # Inisialisasi StandardScaler

#             scaler = loadscal
#             scaler.fit(X_train[kolom])
#             X_train_scaled = scaler.transform(X_train[kolom])
#             # Lakukan standarisasi pada kolom yang telah ditentukan
#             data_ternormalisasi = scaler.transform(data_mentah[kolom])

#             with open("MinMaxScaler.pkl", "rb") as minmaxs:
#                 loadminmax = pickle.load(minmaxs)
#                 loadminmax.fit(X_train[kolom])
#                 data_MinimMaxim = loadminmax.transform(data_mentah[kolom])

#             # Reduksi dimensi menggunakan PCA
#             with open('sklearn_pca.pkl', 'rb') as reduk:
#                     loadpca = pickle.load(reduk)

#             X_train_pca = loadpca.fit_transform(X_train_scaled)
#             X_pca = loadpca.transform(data_ternormalisasi)
#             MM_pca = loadpca.transform(data_MinimMaxim)


#             # --------- INPORT PICKLE -------------------

#             # untuk k 13 error di data set 212 dan jadi disgust

#             with open('modelknn_k1.sav', 'rb') as knn:
#                 loadknn = pickle.load(knn)
#             loadknn.fit(X_train_scaled, y_train)

#             # Inisialisasi KNN Classifier
#             loadknn.fit(X_train_pca, y_train)
#             y_prediksi = loadknn.predict(X_pca)

#             # classifier = KNeighborsClassifier(n_neighbors=2)
#             # classifier.fit(X_train_pca, y_train)
#             # y_prediksi = classifier.predict(X_pca)

#             # Menampilkan hasil klasifikasi
#             st.write("Hasil Klasifikasi:")
#             st.write(y_prediksi)
#             st.audio(audio_filename)
