import streamlit as st
import numpy as np
from PIL import Image
import os
from openvino.runtime import Core

# ── 설정 ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_XML = "weights/leather_model.xml"
MODEL_BIN = "weights/leather_model.bin"
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
    color: #1a1a1a;
}

h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    letter-spacing: 0.12em;
    color: #1a6fd4;
    border-bottom: 1px solid #d0d0d0;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

.result-box {
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-top: 1.2rem;
    background: #f7f7f7;
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

.stButton button {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    border-radius: 3px;
    padding: 0.5rem 1.5rem;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── 모델 로드 (캐싱) ───────────────────────────────────
@st.cache_resource(show_spinner="모델 가중치 로드 중…")
def load_model(xml_path: str):
    ie = Core()
    model = ie.read_model(model=xml_path)
    compiled = ie.compile_model(model, device_name="CPU")
    return compiled


def predict(compiled_model, pil_image: Image.Image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)

    arr = arr[:, :, ::-1]  # RGB → BGR
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    arr -= mean

    arr = np.expand_dims(arr, axis=0)

    # 모델 입력이 NCHW라면 아래 주석 해제:
    # arr = arr.transpose(0, 3, 1, 2)

    output_layer = compiled_model.output(0)
    result = compiled_model([arr])[output_layer]
    prob = float(result.flatten()[0])

    label = "불량" if prob > THRESHOLD else "정상"
    return prob, label


# ── UI ────────────────────────────────────────────────
st.markdown("<h1>ParkJunSeo</h1>", unsafe_allow_html=True)

if not os.path.exists(MODEL_XML):
    st.error(f"모델 파일을 찾을 수 없습니다: `{MODEL_XML}`\n\n"
             "`weights/` 폴더에 `leather_model.xml` 과 `leather_model.bin` 을 놓아주세요.")
    st.stop()

compiled_model = load_model(MODEL_XML)

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
            prob, label = predict(compiled_model, pil_img)

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