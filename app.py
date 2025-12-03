import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], 'GPU')

# ==================== CUSTOM LAYERS - ULTIMATE FIX ====================

class FixedInputLayer(tf.keras.layers.InputLayer):
    """Fix InputLayer batch_shape issue"""
    def __init__(self, **kwargs):
        # Convert batch_shape to batch_input_shape
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)
    
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return cls(**config)

class CustomMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """Fixed MultiHeadAttention"""
    def __init__(self, **kwargs):
        if 'key_dim' not in kwargs:
            kwargs['key_dim'] = 64
        
        remove_params = ['query_shape', 'key_shape', 'value_shape', 
                        '_use_query_shape', '_use_key_shape', '_use_value_shape',
                        'query_dim', 'value_dim']
        for param in remove_params:
            kwargs.pop(param, None)
        
        super().__init__(**kwargs)
    
    @classmethod
    def from_config(cls, config):
        if 'key_dim' not in config:
            config['key_dim'] = 64
        
        remove_params = ['query_shape', 'key_shape', 'value_shape',
                        '_use_query_shape', '_use_key_shape', '_use_value_shape',
                        'query_dim', 'value_dim']
        for param in remove_params:
            config.pop(param, None)
        
        return cls(**config)

class CustomLayerNormalization(tf.keras.layers.LayerNormalization):
    """Fixed LayerNormalization"""
    def __init__(self, **kwargs):
        kwargs.pop('rms_scaling', None)
        super().__init__(**kwargs)
    
    @classmethod
    def from_config(cls, config):
        config.pop('rms_scaling', None)
        return cls(**config)

# ==================== LOAD MODEL ====================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model dengan error handling lengkap"""
    model_path = "vit_compatible.h5"
    
    # Validasi file
    if not os.path.exists(model_path):
        st.error(f"❌ File tidak ditemukan: {model_path}")
        st.info("Pastikan vit_compatible.h5 ada di root repository")
        return None
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    
    if file_size < 5.0:
        st.error(f"❌ File terlalu kecil: {file_size:.2f} MB")
        st.info("Model seharusnya ~5.3 MB. File mungkin corrupt.")
        return None
    
    try:
        # Custom objects
        custom_objects = {
            'InputLayer': FixedInputLayer,
            'MultiHeadAttention': CustomMultiHeadAttention,
            'LayerNormalization': CustomLayerNormalization,
            'FixedInputLayer': FixedInputLayer,
            'CustomMultiHeadAttention': CustomMultiHeadAttention,
            'CustomLayerNormalization': CustomLayerNormalization,
        }
        
        # Load model
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False
            )
        
        return model
        
    except Exception as e:
        error_str = str(e)
        st.error("❌ Gagal load model!")
        
        if 'key_dim' in error_str:
            st.warning("⚠️ Error: Parameter 'key_dim' tidak ditemukan")
            with st.expander("🔧 Solusi"):
                st.code("""
# Jalankan di Google Colab dengan TF 2.15.0:

import tensorflow as tf
model = tf.keras.models.load_model('vit_compatible.h5')
model.save('vit_compatible_fixed.h5', save_format='h5')

# Upload vit_compatible_fixed.h5 ke Hugging Face
                """, language='python')
        else:
            st.code(error_str, language='text')
        
        return None

# ==================== CLASS NAMES ====================

CLASS_NAMES = [
    "Abraham_Ganda_Napitu", "Ahmad_Faqih_Hasani", "Abu_Bakar_Siddiq_Siregar",
    "Aldi_Sanjaya", "Alfajar", "Arkan_Hariz_Chandrawinata_Liem",
    "Alief_Fathur_Rahman", "Bayu_Ega_Ferdana", "Bayu_Prameswara_Haris",
    "Bezalel_Samuel_Manik", "Bintang_Fikri_Fauzan", "Boy_Sandro_Sigiro",
    "Desty_Ananta_Purba", "Dimas_Azi_Rajab_Aizar", "Dwi_Arthur_Revangga",
    "Dito_Rifki_Irawan", "Eden_Wijaya", "Dyo_Dwi_Carol_Bukit",
    "Eichal_Elphindo_Ginting", "Elsa_Elisa_Yohana_Sianturi",
    "Fajrul_Ramadhana_Aqsa", "Fathan_Andi_Kartagama", "Falih_Dzakwan_Zuhdi",
    "Femmy_Aprillia_Putri", "Fayyadh_Abdillah", "Ferdana_Al_Hakim",
    "Fiqri_Aldiansyah", "Festus_Mikhael", "Freddy_Harahap",
    "Gabriella_Natalya_Rumapea", "Garland_Wijaya", "hayyatul_fajri",
    "Havidz_Ridho_Pratama", "Ichsan_Kuntadi_Baskara", "Ikhsannudin_Lathief",
    "Intan_Permata_Sari", "Joy_Daniella_V", "Joyapul_Hanscalvin_Panjaitan",
    "Joshua_Palti_Sinaga", "Joshia_Fernandes_Sectio_Purba",
    "JP_Rafi_Radiktya_Arkan_R_AZ", "Kayla_Chika_Lathisya",
    "Kenneth_Austin_Wijaya", "Kevin_Naufal_Dany", "Lois_Novel_E_Gurning",
    "Martua_Kevin_AMHLubis", "Muhammad_Fasya_Atthoriq",
    "Muhammad_Nelwan_Fakhri", "Muhammad_Riveldo_Hermawan_Putra",
    "Muhammad_Zada_Rizki", "Nasya_Aulia_Efendi", "Raditya_Erza_Farandi",
    "Randy_Hendriyawan", "Rahmat_Aldi_Nasda", "Rayhan_Fadel_Irwanto",
    "Rayhan_Fatih_Gunawan", "Rizky_Abdillah", "Reynaldi_Cristian_Simamora",
    "Royfran_Roger_Valentino", "Rustian_Afencius_Marbun",
    "Shintya_Ayu_Wardani", "Sikah_Nubuahtul_Ilmi", "Zakhi_algifari",
    "Yohanna_Anzelika_Sitepu", "Zidan_Raihan", "Zaky_Ahmad_Makarim",
    "Zefanya_Danovanta_Tarigan", "William_Chan"
]

def preprocess_image(image):
    """Preprocessing sesuai training"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = image.resize((224, 224), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        return None

# ==================== MAIN APP ====================

def main():
    st.set_page_config(page_title="Presensi Mahasiswa", page_icon="🎓", layout="wide")
    
    st.markdown("""
    <style>
    .header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #1a2a6c, #b21f1f, #1a2a6c);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="header">
        <h1>🎓 SISTEM PRESENSI MAHASISWA</h1>
        <p>Deep Learning Face Recognition - Institut Teknologi Sumatera</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Info")
        st.metric("TensorFlow", tf.__version__)
        st.metric("Total Mahasiswa", len(CLASS_NAMES))
        
        st.markdown("---")
        st.markdown("### 📋 Tips")
        st.info("""
        • Foto frontal wajah
        • Pencahayaan baik
        • Tanpa objek menghalangi
        • Format: JPG/PNG
        """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is not None:
        st.success("✅ Model berhasil dimuat!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Layers", len(model.layers))
        with col2:
            st.metric("Parameters", f"{model.count_params():,}")
        with col3:
            st.metric("Input", str(model.input_shape[1:3]))
    
    st.markdown("---")
    st.markdown("### 📸 Upload Foto")
    
    uploaded = st.file_uploader("Pilih foto...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded and model is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded, caption="Input", use_container_width=True)
        
        with col2:
            with st.spinner("Analyzing..."):
                try:
                    img = Image.open(uploaded)
                    img_array = preprocess_image(img)
                    
                    if img_array is None:
                        return
                    
                    start = time.time()
                    pred = model.predict(img_array, verbose=0)
                    elapsed = time.time() - start
                    
                    idx = np.argmax(pred[0])
                    conf = pred[0][idx] * 100
                    
                    if conf > 15:
                        st.success(f"**{CLASS_NAMES[idx].replace('_', ' ')}**")
                        st.metric("Confidence", f"{conf:.2f}%")
                        st.caption(f"⏱️ {elapsed:.3f}s")
                        st.balloons()
                    else:
                        st.warning("⚠️ Confidence rendah")
                    
                    st.markdown("---")
                    st.markdown("### Top 3")
                    top = np.argsort(pred[0])[-3:][::-1]
                    for i, t in enumerate(top):
                        c = pred[0][t] * 100
                        st.write(f"{i+1}. **{CLASS_NAMES[t].replace('_', ' ')}** - {c:.1f}%")
                        st.progress(c/100)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif uploaded and model is None:
        st.error("Model belum dimuat!")

if __name__ == "__main__":
    main()