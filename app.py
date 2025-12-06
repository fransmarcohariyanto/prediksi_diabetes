
# Memuat model yang telah disimpan
model_xgb = joblib.load('xgboost_model.pkl')

# Fungsi untuk prediksi
def predict_diabetes(age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose):
    input_data = np.array([[age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]]).astype(float)
    prediction = model_xgb.predict(input_data)
    return prediction[0]

# Streamlit interface
st.title('Prediksi Diabetes dengan XGBoost')

# Input data
age = st.number_input('Usia', min_value=0, max_value=100, value=25)
hypertension = st.selectbox('Apakah Anda hipertensi?', ['Tidak', 'Ya'])
heart_disease = st.selectbox('Apakah Anda menderita penyakit jantung?', ['Tidak', 'Ya'])
smoking_history = st.selectbox('Riwayat merokok', ['Tidak', 'Pernah', 'Saat ini'])
bmi = st.number_input('BMI', min_value=10, max_value=50, value=25)
hba1c = st.number_input('Level HbA1c', min_value=4, max_value=10, value=6)
glucose = st.number_input('Level Glukosa Darah', min_value=50, max_value=300, value=100)

# Mengonversi input ke dalam bentuk numerik
hypertension = 1 if hypertension == 'Ya' else 0
heart_disease = 1 if heart_disease == 'Ya' else 0
smoking_history = 1 if smoking_history == 'Saat ini' else (0 if smoking_history == 'Tidak' else 2)  # 2 = 'Pernah'

# Tombol prediksi
if st.button('Prediksi Diabetes'):
    result = predict_diabetes(age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose)
    if result == 1:
        st.success('Anda berisiko diabetes')
    else:
        st.success('Anda tidak berisiko diabetes')
