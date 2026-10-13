import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Config Halaman
st.set_page_config(
    page_title="DSS Pemeliharaan & Seleksi Material Crew Boat",
    page_icon="🚢",
    layout="wide"
)

# Header Utama
st.title("🚢 Decision Support System (DSS) Pemeliharaan Prediktif & Seleksi Material")
st.subheader("Aplikasi Sistem Pendingin Air Laut Mesin Kapal Crew Boat")
st.markdown("---")

# 1. Load Data & Latih Model ML (Cache agar Cepat)
@st.cache_data
def load_and_train():
    df = pd.read_excel('Alloys_Dengan_PREN_dan_Laju_Korosi.xlsx')
    df = df.drop_duplicates(subset=['Komposisi_Cr_pct', 'Komposisi_Mo_pct', 'Komposisi_Ni_pct']).reset_index(drop=True)
    
    df['UTS_MPa'] = df['Kuat_Tarik_UTS_psi'] * 0.00689476
    df['Nilai_PREN'] = df['Komposisi_Cr_pct'] + 3.3 * (df['Komposisi_Mo_pct'] + 0.5 * df['Komposisi_W_pct']) + 16 * df['Komposisi_N_pct']
    
    np.random.seed(42)
    df['T_op_CrewBoat'] = np.random.uniform(40, 95, size=len(df))
    df['Seawater_Flow_Index'] = np.random.uniform(0.1, 1.0, size=len(df))
    
    base_degrad = (df['T_op_CrewBoat'] / df['Titik_Lebur_Liquidus_C']) * 0.02 + (df['Seawater_Flow_Index'] / (df['Nilai_PREN'] + 1)) * 0.15
    df['Degradation_Rate_per_1000h'] = base_degrad * (1000 / (df['UTS_MPa'] + 1))
    df['RUL_Hours_CrewBoat'] = ((0.40 * df['UTS_MPa']) / (df['Degradation_Rate_per_1000h'] / 1000)).clip(upper=100000)
    
    element_cols = [c for c in df.columns if c.startswith('Komposisi_')]
    feature_cols = element_cols + ['UTS_MPa', 'Nilai_PREN', 'T_op_CrewBoat', 'Seawater_Flow_Index']
    
    X = df[feature_cols].fillna(0)
    y = df['RUL_Hours_CrewBoat']
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    return df, feature_cols, model

df, feature_cols, model = load_and_train()

# Layout Sidebar Menu
st.sidebar.header("⚙️ Panel Navigasi Modul")
modul_pilihan = st.sidebar.radio("Pilih Modul DSS:", [
    "Modul 1: Kalkulator RUL & Early Warning",
    "Modul 2: Rekomendasi Material (TOPSIS)"
])

# ==========================================
# MODUL 1: KALKULATOR RUL & EARLY WARNING
# ==========================================
if modul_pilihan == "Modul 1: Kalkulator RUL & Early Warning":
    st.header("⏱️ Modul 1: Simulasi Prediksi Remaining Useful Life (RUL)")
    
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        st.subheader("1. Pilih Material Terpasang")
        selected_alloy = st.selectbox("Nama Paduan Logam Komponen:", df['Nama_Alloy'].unique())
        
        row_mat = df[df['Nama_Alloy'] == selected_alloy].iloc[0]
        st.info(f"**Properti Intrinsik Material:**\n"
                f"- **Kuat Tarik (UTS):** {row_mat['UTS_MPa']:.2f} MPa\n"
                f"- **Nilai PREN:** {row_mat['Nilai_PREN']:.2f}\n"
                f"- **Laju Korosi Base:** {row_mat['Laju_Korosi_Air_Laut_mm_thn']:.4f} mm/tahun")

    with col_in2:
        st.subheader("2. Parameter Operasional Fluida Air Laut")
        temp_input = st.slider("Suhu Air Laut Pendingin (°C):", min_value=40.0, max_value=95.0, value=65.0, step=1.0)
        flow_input = st.slider("Indeks Aliran & Turbulensi Fluida (0.1 - 1.0):", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        
    # Kalkulasi Prediksi ML
    input_data = row_mat[feature_cols].copy()
    input_data['T_op_CrewBoat'] = temp_input
    input_data['Seawater_Flow_Index'] = flow_input
    
    pred_rul = model.predict(pd.DataFrame([input_data]))[0]
    pred_bulan = pred_rul / (30 * 12) # Asumsi 12 jam/hari operasional
    
    st.markdown("---")
    st.subheader("📊 Hasil Output Prediksi RUL & Status Sistem Warning")
    
    col_out1, col_out2, col_out3 = st.columns(3)
    col_out1.metric("Prediksi RUL (Jam Kerja)", f"{pred_rul:,.0f} Jam")
    col_out2.metric("Estimasi Masa Operasi", f"{pred_bulan:.1f} Bulan")
    
    # Logic Alarm
    if pred_rul < 1500:
        col_out3.error("🚨 STATUS: KRITIS\n(Segera Jadwalkan Penggantian!)")
    elif pred_rul < 3500:
        col_out3.warning("⚠️ STATUS: WASPADA\n(Jadwalkan Inspeksi Berkala)")
    else:
        col_out3.success("✅ STATUS: AMAN\n(Kondisi Komponen Sangat Baik)")

    # Plot Degradasi
    fig, ax = plt.subplots(figsize=(10, 4))
    hours = np.linspace(0, pred_rul * 1.2, 100)
    uts_degrad = row_mat['UTS_MPa'] * (1 - 0.4 * (hours / pred_rul))
    
    ax.plot(hours, uts_degrad, color='navy', lw=2.5, label='Penurunan Kekuatan (UTS)')
    ax.axhline(row_mat['UTS_MPa'] * 0.6, color='red', linestyle='--', label='Batas Kritis Kegagalan (60% UTS)')
    ax.axvline(pred_rul, color='orange', linestyle=':', label=f'Titik RUL ({pred_rul:,.0f} Jam)')
    ax.set_title("Simulasi Laju Penurunan Kekuatan Material terhadap Jam Kerja Mesin", fontsize=11, fontweight='bold')
    ax.set_xlabel("Jam Kerja Operasional Mesin (Jam)")
    ax.set_ylabel("Kekuatan Tarik UTS (MPa)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

# ==========================================
# MODUL 2: REKOMENDASI MATERIAL (TOPSIS)
# ==========================================
else:
    st.header("🎯 Modul 2: Mesin Rekomendasi Material Pengganti Optimal (MCDM TOPSIS)")
    st.write("Tentukan bobot prioritas kriteria sesuai kebutuhan perencanaan perawatan kapal crew boat:")
    
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    w_uts = col_w1.slider("Bobot Kekuatan (UTS):", 0.0, 1.0, 0.2, 0.05)
    w_pren = col_w2.slider("Bobot Ketahanan Pitting (PREN):", 0.0, 1.0, 0.4, 0.05)
    w_laju = col_w3.slider("Bobot Laju Korosi (Cost):", 0.0, 1.0, 0.3, 0.05)
    w_biaya = col_w4.slider("Bobot Biaya Material (Cost):", 0.0, 1.0, 0.1, 0.05)
    
    # Hitung TOPSIS
    candidates = df[df['Nilai_PREN'] > 10].copy().reset_index(drop=True)
    candidates['Estimasi_Biaya'] = 10 + (1.5 * candidates['Komposisi_Cr_pct']) + (3.0 * candidates['Komposisi_Ni_pct']) + (5.0 * candidates['Komposisi_Mo_pct'])
    
    X_mat = candidates[['UTS_MPa', 'Nilai_PREN', 'Laju_Korosi_Air_Laut_mm_thn', 'Estimasi_Biaya']].values
    norm_mat = X_mat / np.sqrt((X_mat**2).sum(axis=0))
    weights = np.array([w_uts, w_pren, w_laju, w_biaya])
    v_mat = norm_mat * weights
    
    a_plus = np.array([v_mat[:,0].max(), v_mat[:,1].max(), v_mat[:,2].min(), v_mat[:,3].min()])
    a_minus = np.array([v_mat[:,0].min(), v_mat[:,1].min(), v_mat[:,2].max(), v_mat[:,3].max()])
    
    d_plus = np.sqrt(((v_mat - a_plus)**2).sum(axis=1))
    d_minus = np.sqrt(((v_mat - a_minus)**2).sum(axis=1))
    
    candidates['Skor_TOPSIS'] = d_minus / (d_plus + d_minus)
    ranked = candidates.sort_values(by='Skor_TOPSIS', ascending=False).reset_index(drop=True)
    
    top_df = ranked[['Nama_Alloy', 'UTS_MPa', 'Nilai_PREN', 'Laju_Korosi_Air_Laut_mm_thn', 'Estimasi_Biaya', 'Skor_TOPSIS']].head(10)
    
    st.markdown("---")
    st.subheader("🏆 Top 10 Rekomendasi Material Terbaik Hasil MCDM TOPSIS")
    st.dataframe(top_df.style.highlight_max(subset=['Skor_TOPSIS'], color='lightgreen'))