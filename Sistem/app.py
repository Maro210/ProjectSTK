import streamlit as st
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="ABSA - Analisis Sentimen",
    page_icon="📊",
    layout="wide"
)

# ========================================
# KONFIGURASI MODEL
# ========================================
MODEL_NAME = "indobenchmark/indobert-base-p1"
MODEL_PATH = "best_indobert_absa_augmented.pt"  # Ganti dengan nama file .pt Anda

# Judul aplikasi
st.title("🎯 Analisis Sentimen Berbasis Aspek (ABSA)")
st.markdown("### Aspek: Kemudahan, Pembayaran, dan Aplikasi")
st.markdown("---")

# Fungsi preprocessing teks (SAMA seperti di training)
def clean_text(text):
    """Preprocessing teks - HARUS SAMA dengan training!"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Load tokenizer dan model
@st.cache_resource
def load_model_and_tokenizer():
    """Load IndoBERT model dan tokenizer"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer dari Hugging Face (bukan dari file lokal!)
        st.info("📥 Loading tokenizer dari Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model architecture dari Hugging Face
        st.info("📥 Loading model architecture...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        
        # Load trained weights dari file lokal
        st.info(f"📥 Loading trained weights dari {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        return model, tokenizer, device
    
    except FileNotFoundError:
        st.error(f"❌ File model tidak ditemukan: {MODEL_PATH}")
        st.info("💡 Pastikan file .pt ada di folder yang sama dengan app.py")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

# Fungsi untuk menghapus kata kunci aspek dari konteks
def remove_aspect_keywords(text, aspect):
    """
    Hapus kata kunci aspek dari konteks untuk mengurangi bias
    Ini membantu model fokus pada sentimen, bukan keberadaan kata aspek
    """
    aspect_keywords = {
        'kemudahan': [
            'mudah', 'gampang', 'simple', 'praktis', 'susah', 'sulit', 
            'ribet', 'rumit', 'complicated', 'kemudahan', 'kesulitan'
        ],
        'pembayaran': [
            'bayar', 'pembayaran', 'payment', 'tagihan', 'invoice', 
            'biaya', 'harga', 'gratis', 'paid', 'pay', 'transaksi'
        ],
        'aplikasi': [
            'aplikasi', 'app', 'apk', 'platform', 'software', 
            'sistem', 'program', 'interface'
        ]
    }
    
    text_lower = text.lower()
    keywords = aspect_keywords.get(aspect.lower(), [])
    
    # Hapus setiap keyword dari konteks
    for keyword in keywords:
        # Gunakan regex untuk menghapus kata utuh (bukan bagian dari kata lain)
        import re
        pattern = r'\b' + re.escape(keyword) + r'\b'
        text_lower = re.sub(pattern, '', text_lower)
    
    # Bersihkan spasi ganda
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    
    return text_lower

# Fungsi untuk deteksi kata positif/negatif
def detect_sentiment_words(text):
    """Deteksi kata-kata sentimen dalam teks"""
    positive_words = [
        'bagus', 'baik', 'mantap', 'oke', 'keren', 'sempurna', 'excellent',
        'cepat', 'mudah', 'gampang', 'praktis', 'efisien', 'lancar', 'smooth',
        'recommended', 'puas', 'suka', 'senang', 'worth', 'terbaik', 'top',
        'amazing', 'awesome', 'great', 'nice', 'good', 'love', 'perfect'
    ]
    
    negative_words = [
        'jelek', 'buruk', 'parah', 'error', 'bug', 'crash', 'lambat', 'lemot',
        'lelet', 'susah', 'ribet', 'rumit', 'gagal', 'tidak', 'nggak', 'ga',
        'bisa', 'eror', 'loading', 'lag', 'hang', 'force close', 'rusak',
        'bermasalah', 'kecewa', 'bad', 'worst', 'terrible', 'horrible', 'lama',
        'gak', 'gk', 'ngga', 'payah', 'ancur', 'kacau', 'berantakan', 'mengecewakan'
    ]
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    return pos_count, neg_count

# Fungsi prediksi dengan bias correction
def predict_sentiment(text, aspect, model, tokenizer, device):
    """
    Prediksi sentimen untuk satu aspek dengan BIAS CORRECTION
    """
    # Clean text
    aspect_clean = clean_text(aspect)
    context_clean = clean_text(text)
    
    # Hapus kata kunci aspek dari konteks
    context_clean_no_aspect = remove_aspect_keywords(context_clean, aspect)
    
    # Deteksi kata sentimen dalam konteks ASLI (sebelum hapus aspek)
    pos_count, neg_count = detect_sentiment_words(context_clean)
    
    # Format input model
    input_text = f"{aspect_clean} [SEP] {context_clean_no_aspect}"
    
    # Tokenize
    encoding = tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Prediksi model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    # BIAS CORRECTION: Adjust probabilities berdasarkan kata sentimen
    # Strategi baru: lebih konservatif, hanya adjust jika perbedaan signifikan
    adjusted_probs = probabilities.copy()
    
    # Hitung selisih kata sentimen
    sentiment_diff = pos_count - neg_count
    
    # Hanya adjust jika ada perbedaan jelas (minimal 2 kata)
    if abs(sentiment_diff) >= 2:
        if sentiment_diff > 0:
            # Lebih banyak kata positif (minimal 2 selisih)
            adjusted_probs[2] *= 1.3  # Boost positif lebih moderat
            adjusted_probs[0] *= 0.8   # Turunkan negatif sedikit
        else:
            # Lebih banyak kata negatif (minimal 2 selisih)
            adjusted_probs[0] *= 1.2   # Boost negatif moderat
            adjusted_probs[2] *= 0.8   # Turunkan positif sedikit
    
    # Jika perbedaan kecil (0-1 kata) atau seimbang, boost netral
    elif abs(sentiment_diff) <= 1:
        adjusted_probs[1] *= 1.4   # Boost netral
    
    # Normalize kembali ke total = 1
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    # Prediksi final
    prediction = int(adjusted_probs.argmax())
    
    return prediction, adjusted_probs, input_text, pos_count, neg_count

# Mapping label
label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

# Fungsi warna sentimen
def get_sentiment_color(sentiment):
    colors = {
        "Positif": "#28a745",
        "Netral": "#ffc107",
        "Negatif": "#dc3545"
    }
    return colors.get(sentiment, "#6c757d")

# Fungsi chart
def create_sentiment_chart(results):
    aspects = list(results.keys())
    sentiments = list(results.values())
    colors = [get_sentiment_color(s) for s in sentiments]
    
    fig = go.Figure(data=[
        go.Bar(
            x=aspects,
            y=[1, 1, 1],
            marker_color=colors,
            text=sentiments,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Sentimen: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Hasil Analisis Sentimen per Aspek",
        xaxis_title="Aspek",
        yaxis_title="",
        showlegend=False,
        height=400,
        yaxis=dict(showticklabels=False)
    )
    
    return fig

# Load model
with st.spinner("Loading IndoBERT model..."):
    model, tokenizer, device = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.stop()

st.success(f"✅ Model berhasil dimuat!")
st.info(f"🖥️ Device: {device} | Model: {MODEL_NAME}")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Tunggal", "📁 Prediksi Batch (CSV)", "📊 Visualisasi Data"])

# TAB 1: Prediksi Tunggal
with tab1:
    st.header("Prediksi Sentimen Teks Tunggal")
    
    user_input = st.text_area(
        "Masukkan teks untuk dianalisis:",
        placeholder="Contoh: Aplikasinya mudah digunakan, pembayarannya juga cepat dan aman.",
        height=150
    )
    
    st.markdown("### Pilih Aspek yang Ingin Dianalisis:")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        analyze_kemudahan = st.checkbox("✅ Kemudahan", value=True)
    with col_b:
        analyze_pembayaran = st.checkbox("💳 Pembayaran", value=True)
    with col_c:
        analyze_aplikasi = st.checkbox("📱 Aplikasi", value=True)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🚀 Analisis Sentimen", type="primary")
    with col2:
        if st.button("🗑️ Clear"):
            st.rerun()
    
    if predict_button and user_input:
        # Cek apakah minimal 1 aspek dipilih
        selected_aspects = []
        if analyze_kemudahan:
            selected_aspects.append("Kemudahan")
        if analyze_pembayaran:
            selected_aspects.append("Pembayaran")
        if analyze_aplikasi:
            selected_aspects.append("Aplikasi")
        
        if not selected_aspects:
            st.warning("⚠️ Pilih minimal 1 aspek untuk dianalisis!")
        else:
            with st.spinner(f"Menganalisis sentimen untuk {len(selected_aspects)} aspek..."):
                try:
                    # Prediksi untuk aspek yang dipilih
                    results = {}
                    probabilities_dict = {}
                    
                    for aspect in selected_aspects:
                        pred, probs, processed_input, pos_cnt, neg_cnt = predict_sentiment(user_input, aspect, model, tokenizer, device)
                        results[aspect] = label_map[pred]
                        probabilities_dict[aspect] = {
                            'probs': probs,
                            'pos_words': pos_cnt,
                            'neg_words': neg_cnt
                        }
                    
                    # Tampilkan hasil
                    st.markdown("### 📊 Hasil Analisis:")
                    
                    # Buat kolom dinamis berdasarkan jumlah aspek yang dipilih
                    cols = st.columns(len(selected_aspects))
                    
                    for idx, aspect in enumerate(selected_aspects):
                        with cols[idx]:
                            sentiment = results[aspect]
                            color = get_sentiment_color(sentiment)
                            probs = probabilities_dict[aspect]['probs']
                            confidence = probs.max() * 100
                            
                            # Icon per aspek
                            icon_map = {
                                "Kemudahan": "✅",
                                "Pembayaran": "💳",
                                "Aplikasi": "📱"
                            }
                            icon = icon_map.get(aspect, "🎯")
                            
                            st.markdown(f"""
                            <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
                                <h3 style='color: white; margin: 0;'>{icon} {aspect}</h3>
                                <h2 style='color: white; margin: 10px 0;'>{sentiment}</h2>
                                <p style='color: white; margin: 0; font-size: 14px;'>Confidence: {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Chart (hanya jika lebih dari 1 aspek)
                    if len(selected_aspects) > 1:
                        fig = create_sentiment_chart(results)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detail probabilities
                    with st.expander("📈 Detail Analisis per Aspek"):
                        for aspect in selected_aspects:
                            st.markdown(f"### {aspect}")
                            
                            data = probabilities_dict[aspect]
                            probs = data['probs']
                            pos_words = data['pos_words']
                            neg_words = data['neg_words']
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**Probabilitas:**")
                                prob_df = pd.DataFrame({
                                    'Sentimen': ['Negatif', 'Netral', 'Positif'],
                                    'Probabilitas': [f"{p*100:.2f}%" for p in probs]
                                })
                                st.dataframe(prob_df, hide_index=True)
                            
                            with col_b:
                                st.markdown("**Analisis Kata:**")
                                st.metric("Kata Positif", pos_words)
                                st.metric("Kata Negatif", neg_words)
                                
                                if pos_words > neg_words:
                                    st.success("✅ Lebih banyak kata positif")
                                elif neg_words > pos_words:
                                    st.error("❌ Lebih banyak kata negatif")
                                else:
                                    st.info("ℹ️ Kata sentimen seimbang/netral")
                            
                            st.markdown("---")
                    
                    # Teks setelah preprocessing untuk setiap aspek
                    with st.expander("🔍 Lihat Preprocessing per Aspek"):
                        st.markdown("**Teks Original:**")
                        st.code(user_input)
                        st.markdown("---")
                        for aspect in selected_aspects:
                            pred, probs, processed_input, pos_cnt, neg_cnt = predict_sentiment(user_input, aspect, model, tokenizer, device)
                            st.markdown(f"**Input untuk Aspek {aspect}:**")
                            st.code(processed_input)
                            st.caption(f"↳ Kata kunci '{aspect.lower()}' dihapus | Kata positif: {pos_cnt} | Kata negatif: {neg_cnt}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    elif predict_button and not user_input:
        st.warning("⚠️ Mohon masukkan teks terlebih dahulu!")

# TAB 2: Prediksi Batch
with tab2:
    st.header("Prediksi Sentimen Batch (CSV)")
    
    st.info("📁 Upload file CSV dengan kolom 'context' (teks ulasan)")
    
    uploaded_csv = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("**Preview Data:**")
        st.dataframe(df.head())
        
        text_column = st.selectbox("Pilih kolom yang berisi teks:", df.columns)
        
        if st.button("🚀 Analisis Semua Data"):
            with st.spinner("Memproses data..."):
                try:
                    results_list = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        text = str(row[text_column])
                        
                        # Prediksi untuk 3 aspek
                        result_row = {'Text': text}
                        
                        for aspect in ["Kemudahan", "Pembayaran", "Aplikasi"]:
                            pred, _, _, _, _ = predict_sentiment(text, aspect, model, tokenizer, device)
                            result_row[f'Sentimen_{aspect}'] = label_map[pred]
                        
                        results_list.append(result_row)
                        progress_bar.progress((idx + 1) / len(df))
                    
                    results_df = pd.DataFrame(results_list)
                    st.session_state['results_df'] = results_df
                    
                    st.success(f"✅ Berhasil menganalisis {len(results_df)} data!")
                    st.dataframe(results_df)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Hasil (CSV)",
                        data=csv,
                        file_name="hasil_analisis_sentimen.csv",
                        mime="text/csv"
                    )
                    
                    # Statistik
                    st.markdown("### 📈 Statistik Hasil")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Kemudahan:**")
                        st.write(results_df['Sentimen_Kemudahan'].value_counts())
                    
                    with col2:
                        st.markdown("**Pembayaran:**")
                        st.write(results_df['Sentimen_Pembayaran'].value_counts())
                    
                    with col3:
                        st.markdown("**Aplikasi:**")
                        st.write(results_df['Sentimen_Aplikasi'].value_counts())
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# TAB 3: Visualisasi
with tab3:
    st.header("Visualisasi Data Hasil Analisis")
    
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.pie(
                results_df, 
                names='Sentimen_Kemudahan',
                title='Distribusi Sentimen - Kemudahan',
                color='Sentimen_Kemudahan',
                color_discrete_map={
                    'Positif': '#28a745',
                    'Netral': '#ffc107',
                    'Negatif': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                results_df, 
                names='Sentimen_Pembayaran',
                title='Distribusi Sentimen - Pembayaran',
                color='Sentimen_Pembayaran',
                color_discrete_map={
                    'Positif': '#28a745',
                    'Netral': '#ffc107',
                    'Negatif': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.pie(
                results_df, 
                names='Sentimen_Aplikasi',
                title='Distribusi Sentimen - Aplikasi',
                color='Sentimen_Aplikasi',
                color_discrete_map={
                    'Positif': '#28a745',
                    'Netral': '#ffc107',
                    'Negatif': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Perbandingan Sentimen Antar Aspek")
        
        sentiment_summary = pd.DataFrame({
            'Kemudahan': results_df['Sentimen_Kemudahan'].value_counts(),
            'Pembayaran': results_df['Sentimen_Pembayaran'].value_counts(),
            'Aplikasi': results_df['Sentimen_Aplikasi'].value_counts()
        }).fillna(0)
        
        fig = px.bar(
            sentiment_summary,
            barmode='group',
            title='Perbandingan Distribusi Sentimen',
            labels={'value': 'Jumlah', 'variable': 'Aspek'},
            color_discrete_map={
                'Kemudahan': '#007bff',
                'Pembayaran': '#17a2b8',
                'Aplikasi': '#6f42c1'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📊 Lakukan prediksi batch terlebih dahulu untuk melihat visualisasi")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🎯 Analisis Sentimen Berbasis Aspek (ABSA) | IndoBERT Model</p>
    </div>
    """, 
    unsafe_allow_html=True
)