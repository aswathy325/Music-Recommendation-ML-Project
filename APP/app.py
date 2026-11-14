
import streamlit as st
import pandas as pd
import joblib
import os


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Spotify AI Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. CUSTOM DASHBOARD THEME
# =====================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #100c24 0%,
        #1b1235 50%,
        #241044 100%
    );
    color: #f5f3ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* TEXT AND HEADINGS */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 750 !important;
}

p, label, .stMarkdown {
    color: #e4ddf7;
}

/* TEXT INPUTS */
.stTextInput input,
.stNumberInput input {
    background-color: #251b3d !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #8055b8 !important;
    border-radius: 11px !important;
    min-height: 48px;
}

.stTextInput input::placeholder {
    color: #b9a8d5 !important;
    opacity: 1 !important;
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] {
    background-color: #251b3d !important;
    border-radius: 11px !important;
    border: 1px solid #8055b8 !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #ffffff !important;
}

/* DROPDOWN MENU */
div[data-baseweb="popover"] {
    background-color: #251b3d !important;
}

div[data-baseweb="menu"] {
    background-color: #251b3d !important;
}

div[data-baseweb="menu"] li {
    color: #ffffff !important;
}

/* SLIDERS */
.stSlider [data-testid="stThumbValue"] {
    color: #e9d5ff !important;
}

.stSlider [data-testid="stTickBar"] {
    background: #8055b8 !important;
}

/* FORM CONTAINER */
[data-testid="stForm"] {
    background: rgba(40, 27, 69, 0.78) !important;
    border: 1px solid #8055b8 !important;
    border-radius: 20px !important;
    padding: 26px !important;
}

/* BUTTONS */
.stFormSubmitButton button,
.stButton button {
    background: linear-gradient(
        90deg,
        #7628d7,
        #a855f7
    ) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 48px;
    transition: all 0.25s ease;
}

.stFormSubmitButton button:hover,
.stButton button:hover {
    background: linear-gradient(
        90deg,
        #9333ea,
        #c084fc
    ) !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(168, 85, 247, 0.3);
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #281b45,
        #38215c
    );
    border: 1px solid #8055b8;
    border-radius: 16px;
    padding: 24px;
}

[data-testid="stMetricLabel"] {
    color: #d8c5f5 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* DARK PURPLE NOTIFICATIONS */
div[data-testid="stAlert"] {
    background: #251538 !important;
    color: #ffffff !important;
    border: 1px solid #8055b8 !important;
    border-radius: 12px !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
    color: #e9d5ff !important;
}

/* DARK TEXT SELECTION HIGHLIGHT */
::selection {
    background: #6b21a8 !important;
    color: #ffffff !important;
}

/* PROGRESS BAR */
.stProgress > div > div > div {
    background: linear-gradient(
        90deg,
        #7628d7,
        #c084fc
    ) !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 1px solid #8055b8;
    border-radius: 12px;
    overflow: hidden;
}

/* DIVIDERS */
hr {
    border-color: #594078 !important;
}

/* RESULT CARD */
.result-card {
    background: linear-gradient(
        135deg,
        #281b45,
        #38215c
    );
    border: 1px solid #8055b8;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
}

.result-score {
    font-size: 42px;
    font-weight: 800;
    color: #d8b4fe;
}

.result-label {
    font-size: 15px;
    color: #c4b5fd;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #b9a8d5;
    font-size: 13px;
    padding-top: 35px;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. APPLICATION HEADER
# =====================================================

st.markdown("""
<h1 style="font-size:42px;">
🎵 Spotify Popularity Predictor
</h1>

<p style="font-size:17px; color:#c4b5fd;">
Discover the predicted popularity of a song using
Machine Learning and Spotify audio features.
</p>
""", unsafe_allow_html=True)

st.divider()


# =====================================================
# 4. LOAD TRAINED MODEL AND ARTIFACTS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "MODEL")


@st.cache_resource
def load_model_artifacts():

    loaded_model = joblib.load(
        os.path.join(MODEL_DIR, "spotify_random_forest.pkl")
    )

    loaded_preprocessor = joblib.load(
        os.path.join(MODEL_DIR, "spotify_preprocessor.pkl")
    )

    loaded_mappings = joblib.load(
        os.path.join(MODEL_DIR, "spotify_feature_mappings.pkl")
    )

    return loaded_model, loaded_preprocessor, loaded_mappings


try:

    model, preprocessor, feature_mappings = load_model_artifacts()

    artist_counts = feature_mappings["artist_counts"]
    album_counts = feature_mappings["album_counts"]

    genres = sorted(
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .categories_[0]
    )

    st.success("Machine Learning model loaded successfully!")

except Exception as e:

    st.error(f"Unable to load model files: {e}")
    st.stop()


# =====================================================
# 5. PREDICTION HISTORY
# =====================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =====================================================
# 6. SONG INFORMATION FORM
# =====================================================

st.header("🎧 Song Information")

st.write(
    "Enter song details and audio characteristics "
    "to estimate its popularity."
)

with st.form("song_prediction_form"):

    # BASIC INFORMATION

    st.subheader("🎼 Basic Information")

    col1, col2 = st.columns(2)

    with col1:

        artist_name = st.text_input(
            "Artist Name",
            placeholder="Example: Jason Mraz"
        )

        album_name = st.text_input(
            "Album Name",
            placeholder="Example: We Sing. We Dance. We Steal Things."
        )

    with col2:

        genre = st.selectbox(
            "Music Genre",
            genres
        )

        explicit = st.selectbox(
            "Explicit Content",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    st.divider()

    # AUDIO FEATURES

    st.subheader("🎚️ Audio Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        duration_ms = st.number_input(
            "Duration (milliseconds)",
            min_value=1000,
            max_value=1800000,
            value=210000
        )

        danceability = st.slider(
            "Danceability",
            0.0, 1.0, 0.5,
            step=0.0001
        )

        energy = st.slider(
            "Energy",
            0.0, 1.0, 0.5,
            step=0.0001
        )

        loudness = st.slider(
            "Loudness (dB)",
            -60.0, 5.0, -8.0,
            step=0.001
        )

        key = st.selectbox(
            "Key",
            list(range(12))
        )

    with col2:

        mode = st.selectbox(
            "Mode",
            [0, 1]
        )

        speechiness = st.slider(
            "Speechiness",
            0.0, 1.0, 0.05,
            step=0.0001
        )

        acousticness = st.slider(
            "Acousticness",
            0.0, 1.0, 0.5,
            step=0.0001
        )

        instrumentalness = st.slider(
            "Instrumentalness",
            0.0, 1.0, 0.0,
            step=0.0001
        )

        liveness = st.slider(
            "Liveness",
            0.0, 1.0, 0.15,
            step=0.0001
        )

    with col3:

        valence = st.slider(
            "Valence",
            0.0, 1.0, 0.5,
            step=0.0001
        )

        tempo = st.number_input(
            "Tempo (BPM)",
            min_value=0.0,
            max_value=250.0,
            value=120.0,
            step=0.001,
            format="%.3f"
        )

        time_signature = st.selectbox(
            "Time Signature",
            [0, 1, 3, 4, 5],
            index=3
        )

    st.divider()

    predict_button = st.form_submit_button(
        "✨ Predict Song Popularity",
        use_container_width=True
    )


# =====================================================
# 7. FEATURE ENGINEERING AND PREDICTION
# =====================================================

if predict_button:

    if not artist_name.strip() or not album_name.strip():

        st.warning(
            "Please enter both artist name and album name."
        )

    else:

        try:

            artist_count = len([
                artist.strip()
                for artist in artist_name.split(";")
                if artist.strip()
            ])

            artist_track_count = artist_counts.get(
                artist_name.strip(), 0
            )

            album_track_count = album_counts.get(
                album_name.strip(), 0
            )

            duration_minutes = duration_ms / 60000

            input_data = pd.DataFrame([{

                "duration_ms": duration_ms,
                "explicit": explicit,
                "danceability": danceability,
                "energy": energy,
                "key": key,
                "loudness": loudness,
                "mode": mode,
                "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "valence": valence,
                "tempo": tempo,
                "time_signature": time_signature,
                "artist_count": artist_count,
                "artist_track_count": artist_track_count,
                "album_track_count": album_track_count,
                "duration_minutes": duration_minutes,
                "track_genre": genre

            }])

            processed_input = preprocessor.transform(input_data)

            prediction = model.predict(processed_input)[0]

            prediction = max(0, min(100, prediction))

            if prediction < 40:
                popularity_category = "LOW"

            elif prediction < 70:
                popularity_category = "MEDIUM"

            else:
                popularity_category = "HIGH"

            # SAVE PREDICTION HISTORY

            st.session_state.prediction_history.append({

                "Artist": artist_name,
                "Album": album_name,
                "Genre": genre,
                "Predicted Popularity": round(float(prediction), 1),
                "Category": popularity_category

            })

            # DISPLAY RESULTS

            st.divider()

            st.header("✨ Prediction Results")

            result_col1, result_col2 = st.columns([2, 1])

            with result_col1:

                st.metric(
                    label="Predicted Popularity",
                    value=f"{prediction:.1f} / 100"
                )

                st.progress(prediction / 100)

            with result_col2:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            POPULARITY CATEGORY
                        </div>
                        <div class="result-score">
                            {popularity_category}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.success("Prediction generated successfully!")

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# =====================================================
# 8. PREDICTION HISTORY
# =====================================================

if st.session_state.prediction_history:

    st.divider()

    st.header("📊 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []
        st.rerun()


# =====================================================
# 9. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    Spotify AI Popularity Predictor
    <br>
    Machine Learning | Random Forest Regression
</div>
""", unsafe_allow_html=True)
