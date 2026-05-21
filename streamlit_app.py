import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
    
# ── 설정 ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
WEIGHTS_PATH = "leather_model.weights.h5"   # 실행 디렉토리 기준 경로
THRESHOLD = 0.5

# ── 페이지 설정 ────────────────────────────────────────
st.set_page_config(
    page_title="가죽 불량 검사",
    page_icon="🔍",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Pretendard:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
    background-color: #0f0f0f;
    color: #e8e2d9;
}

.stApp { background-color: #0f0f0f; }

h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    letter-spacing: 0.12em;
    color: #c8b99a;
    border-bottom: 1px solid #2e2a24;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

.result-box {
    border: 1px solid #2e2a24;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-top: 1.2rem;
    background: #161410;
}

.label-defect {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    color: #e05a4e;
    letter-spacing: 0.08em;
}

.label-normal {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    color: #6ab187;
    letter-spacing: 0.08em;
}

.prob-text {
    font-size: 0.85rem;
    color: #7a7068;
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
}

.stFileUploader > div {
    background: #161410 !important;
    border: 1px dashed #3a342c !important;
    border-radius: 4px !important;
}

.stButton button {
    background-color: #c8b99a;
    color: #0f0f0f;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    border: none;
    border-radius: 3px;
    padding: 0.5rem 1.5rem;
    width: 100%;
}
.stButton button:hover {
    background-color: #e8d8b8;
}
</style>
""", unsafe_allow_html=True)

# ── 모델 로드 (캐싱) ───────────────────────────────────
@st.cache_resource(show_spinner="모델 가중치 로드 중…")
def load_model(weights_path: str):
    base = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    model = keras.Sequential([
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid", name="predictions"),
    ])
    model.load_weights(weights_path)
    return model


def predict(model, pil_image: Image.Image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "불량" if prob > THRESHOLD else "정상"
    return prob, label


# ── UI ────────────────────────────────────────────────
st.markdown("<h1>LEATHER INSPECTION</h1>", unsafe_allow_html=True)

# 가중치 경로 확인
if not os.path.exists(WEIGHTS_PATH):
    st.error(f"가중치 파일을 찾을 수 없습니다: `{WEIGHTS_PATH}`\n\n"
             f"스크립트와 같은 디렉토리에 `leather_model.weights.h5`를 놓아주세요.")
    st.stop()

model = load_model(WEIGHTS_PATH)

uploaded = st.file_uploader(
    "이미지를 업로드하세요 (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    pil_img = Image.open(uploaded)
    st.image(pil_img, use_container_width=True)

    if st.button("검사 실행"):
        with st.spinner("분석 중…"):
            prob, label = predict(model, pil_img)

        defect_pct = prob * 100
        normal_pct = (1 - prob) * 100
        label_class = "label-defect" if label == "불량" else "label-normal"

        st.markdown(f"""
        <div class="result-box">
            <div class="{label_class}">{label}</div>
            <div class="prob-text">
                불량 확률 {defect_pct:.1f}% &nbsp;·&nbsp; 정상 확률 {normal_pct:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(prob, text=f"불량확률: {defect_pct:.1f}%")