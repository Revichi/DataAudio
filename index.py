import streamlit as st
import librosa
import joblib
import io

# Fungsi untuk melakukan ekstraksi fitur audio
def extract_features(audio_file):
    try:
        # Baca data file dari objek UploadedFile
        audio_data = io.BytesIO(audio_file.read())

        # Menggunakan librosa untuk ekstraksi fitur audio
        audio_data, _ = librosa.load(audio_data, sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=librosa.get_samplerate(audio_data), n_mfcc=13)
        return mfccs
    except Exception as e:
        st.error(f"Error during feature extraction: {str(e)}")
        return None

# Fungsi untuk melakukan klasifikasi dengan model pickle (PKL)
def classify_audio(features, model):
    try:
        # Melakukan klasifikasi
        prediction = model.predict(features.reshape(1, -1))
        return prediction[0]
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return None

def main():
    # Judul aplikasi
    st.title("Aplikasi Klasifikasi Audio")

    # Upload file audio
    audio_file = st.file_uploader("Unggah file audio (format: WAV)", type=["wav"])

    if audio_file is not None:
        # Tampilkan nama file dan informasi
        st.success(f"File audio yang diunggah: {audio_file.name}")
        st.audio(audio_file, format="audio/wav")

        # Ekstraksi fitur audio
        features = extract_features(audio_file)

        if features is not None:
            # Muat model PKL
            model_path = "knn/nn_model.pkl"  # Ganti dengan path model PKL Anda
            model = joblib.load(model_path)

            # Klasifikasi
            prediction = classify_audio(features, model)

            if prediction is not None:
                # Tampilkan hasil klasifikasi
                st.success(f"Hasil Klasifikasi: {prediction}")

if __name__ == "__main__":
    main()
