# # import streamlit as st
# # import tempfile
# # import os
# # from inference import load_model, predict_image, predict_video

# # st.set_page_config(page_title="Deepfake Detection System", layout="centered")
# # st.title("AI Deepfake Detection System")
# # st.write("Upload an image or video to verify whether the media is authentic or AI-manipulated.")

# # @st.cache_resource
# # def get_model():
# #     return load_model("artifacts/best_deepfake_model.pth")

# # model = get_model()

# # uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# # if uploaded_file is not None:
# #     suffix = os.path.splitext(uploaded_file.name)[1].lower()
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
# #         tmp.write(uploaded_file.read())
# #         temp_path = tmp.name

# #     if suffix in [".jpg", ".jpeg", "png"]:
# #         st.image(temp_path, caption="Uploaded Image", use_container_width=True)
# #         with st.spinner("Analyzing facial artifacts..."):
# #             label, conf, fake_prob = predict_image(model, temp_path)
# #     else:
# #         st.video(temp_path)
# #         with st.spinner("Sampling video frames and analyzing..."):
# #             label, conf, fake_prob = predict_video(model, temp_path, num_frames=12)

# #     os.remove(temp_path)

# #     st.subheader("Detection Result")
# #     if label == "FAKE":
# #         st.error(f"**Prediction: FAKE (AI-Manipulated)** — Confidence: {conf:.2f}%")
# #     else:
# #         st.success(f"**Prediction: REAL (Authentic)** — Confidence: {conf:.2f}%")
# #     st.progress(fake_prob, text=f"Manipulation Probability: {fake_prob*100:.1f}%")


# import streamlit as st
# import tempfile
# import os
# from inference import load_model, predict_image, predict_video

# st.set_page_config(page_title="Deepfake Detection System", layout="centered")
# st.title("AI Deepfake Detection System")
# st.write("Upload a photorealistic image or video containing a human face to verify authenticity.")

# @st.cache_resource
# def get_model():
#     return load_model("artifacts/best_deepfake_model.pth")

# model = get_model()

# uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# if uploaded_file is not None:
#     suffix = os.path.splitext(uploaded_file.name)[1].lower()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(uploaded_file.read())
#         temp_path = tmp.name

#     if suffix in [".jpg", ".jpeg", "png"]:
#         st.image(temp_path, caption="Uploaded Image", use_container_width=True)
#         with st.spinner("Detecting face and analyzing artifacts..."):
#             label, conf, fake_prob = predict_image(model, temp_path)
#     else:
#         st.video(temp_path)
#         with st.spinner("Sampling video frames and analyzing..."):
#             label, conf, fake_prob = predict_video(model, temp_path, num_frames=12)

#     os.remove(temp_path)

#     st.subheader("Detection Result")
#     if label == "NO_FACE":
#         st.warning("**No Human Face Detected:** Please upload a clear, front-facing photorealistic image or video. Stylized illustrations and non-face media are outside the facial deepfake scope.")
#     elif label == "FAKE":
#         st.error(f"**Prediction: FAKE (AI-Manipulated)** — Confidence: {conf:.2f}%")
#         st.progress(fake_prob, text=f"Manipulation Probability: {fake_prob*100:.1f}%")
#     else:
#         st.success(f"**Prediction: REAL (Authentic)** — Confidence: {conf:.2f}%")
#         st.progress(fake_prob, text=f"Manipulation Probability: {fake_prob*100:.1f}%")






import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from inference import load_model, predict_image, predict_video

app = FastAPI(title="AI Deepfake Detection System")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = load_model("artifacts/best_deepfake_model.pth")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
async def analyze_media(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    allowed_img = {".jpg", ".jpeg", ".png", ".webp"}
    allowed_vid = {".mp4", ".avi", ".mov", ".mkv"}

    if suffix not in allowed_img and suffix not in allowed_vid:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        if suffix in allowed_img:
            result = predict_image(model, temp_path)
        else:
            result = predict_video(model, temp_path, num_frames=12)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result