import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
import io
import os
import gdown
import uuid

# ---------- FUNGSI MODEL GAN ----------
@st.cache_resource(show_spinner="Memuat model GAN...")
def load_gan_model_from_drive(model_url: str, output_path: str):
    if not os.path.exists(output_path):
        gdown.download(model_url, output_path, quiet=False)
    model = tf.keras.models.load_model(output_path)
    return model


def get_gan_model(model_option, lr_option):
    if model_option == "Model Epoch 40":
        if lr_option == "0.001":
            model_url = 'https://drive.google.com/uc?id=1eDxJUTxbYn_pbI2sZGdr6rHB11SarBUP'
            model_path = 'generator_epoch40_lr001.h5'
        elif lr_option == "0.0001":
            model_url = 'https://drive.google.com/uc?id=1qjtFGfrboU8g2cP2bIttK5rqT8w_jTdF'
            model_path = 'generator_epoch40_lr0001.h5'
        elif lr_option == "0.00001":
            model_url = 'https://drive.google.com/uc?id=1Ho_6zuZJrc9ynMxUk-8L7t8_kHi4t4JJ'
            model_path = 'generator_epoch40_lr00001.h5'
    elif model_option == "Model Epoch 100":
        if lr_option == "0.001":
            model_url = 'https://drive.google.com/uc?id=1-FYpJElB_n0PuD8m143FabnvcNKi72ov'
            model_path = 'generator_epoch100_lr001.h5'
        elif lr_option == "0.0001":
            model_url = 'https://drive.google.com/uc?id=1iQVQ4yltLh9hSyTjfAMFitI1Wj4GGIHm'
            model_path = 'generator_epoch100_lr0001.h5'
        elif lr_option == "0.00001":
            model_url = 'https://drive.google.com/uc?id=154-uO01t-ixxlAIV7bHwuxD5Br7C49nz'
            model_path = 'generator_epoch100_lr00001.h5'

    generator = load_gan_model_from_drive(model_url, model_path)
    return generator

def predict_image(generator, grayscale_image):
    grayscale_image = grayscale_image.resize((256, 256))
    grayscale_array = np.array(grayscale_image).astype('float32') / 255.0
    if grayscale_array.shape != (256, 256):
        raise ValueError(f"Expected image size (256, 256), but got {grayscale_array.shape}")
    grayscale_array = np.repeat(grayscale_array[..., np.newaxis], 3, axis=-1)
    if grayscale_array.shape != (256, 256, 3):
        raise ValueError(f"Expected image shape (256, 256, 3), but got {grayscale_array.shape}")
    input_tensor = np.expand_dims(grayscale_array, axis=0)
    prediction = generator.predict(input_tensor)
    prediction = np.clip(prediction[0], 0, 1)
    return prediction

# ---------- FUNGSI MODEL CNN ----------
@st.cache_resource
def load_caffe_model():
    caffe_dir = 'caffe_model'
    os.makedirs(caffe_dir, exist_ok=True)

    prototxt_path = os.path.join(caffe_dir, 'colorization_deploy_v2.prototxt')
    points_path = os.path.join(caffe_dir, 'pts_in_hull.npy')
    model_path = os.path.join(caffe_dir, 'colorization_release_v2.caffemodel')

    # Ganti ID di bawah ini dengan ID file-mu sendiri
    if not os.path.exists(prototxt_path):
        gdown.download('https://drive.google.com/uc?id=1DZ4cFBYC3_KjOn2ayrhnk2XKHt6E54EJ', prototxt_path, quiet=False)
    if not os.path.exists(points_path):
        gdown.download('https://drive.google.com/uc?id=1Qh54l1Jhh5psiytgsv9WmJVByjpHdF8o', points_path, quiet=False)
    if not os.path.exists(model_path):
        gdown.download('https://drive.google.com/uc?id=1RCb6SJN2T5tdrpPUXEx0L4GBaTtc2OcL', model_path, quiet=False)

    net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts = np.load(points_path)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

def colorize_with_caffe(net, img_pil):
    image = np.array(img_pil.convert("RGB"))
    image_resized = cv.resize(image, (256, 256))
    scaled = image_resized.astype("float32") / 255.0
    lab = cv.cvtColor(scaled, cv.COLOR_RGB2LAB)
    resized = cv.resize(lab, (224, 224))
    L = cv.split(resized)[0]
    L -= 50
    net.setInput(cv.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv.resize(ab, (256, 256))
    L_full = cv.split(lab)[0]
    colorized = np.concatenate((L_full[:, :, np.newaxis], ab), axis=2)
    colorized = cv.cvtColor(colorized, cv.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return Image.fromarray(colorized)

# ---------- STREAMLIT LAYOUT ----------
st.set_page_config(page_title="GAN & CNN Image Colorization", layout="wide")

uploaded_files = st.sidebar.file_uploader("📤 Unggah gambar grayscale", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

model_type = st.sidebar.radio("🧠 Pilih Tipe Model", ["GAN", "CNN"])

if model_type == "GAN":
    model_option = st.sidebar.selectbox("🧪 Pilih model GAN (Epoch)", ["Model Epoch 40", "Model Epoch 100"])
    lr_option = st.sidebar.selectbox("⚙️ Pilih Learning Rate", ["0.001", "0.0001", "0.00001"])

if not uploaded_files:
    st.title("🎉 Pewarnaan Otomatis Batik Menggunakan Model GAN & CNN (Pretrained Caffe)")
    st.markdown("""
        ### Selamat datang di aplikasi Pewarnaan Batik! 🖼️
        Pilih model pewarnaan di sidebar: **GAN** (Pix2Pix) atau **CNN** (Pretrained Caffe Model).  
        Unggah gambar batik dan dapatkan hasil berwarna secara otomatis! 🎨✨
    """)

else:
    if model_type == "GAN":
        st.title("🧠 Hasil Pewarnaan Gambar dengan Model GAN")
    elif model_type == "CNN":
        st.title("📊 Hasil Pewarnaan Gambar dengan Model CNN")



if uploaded_files:
    if model_type == "GAN":
        generator = get_gan_model(model_option, lr_option)
    elif model_type == "CNN":
        caffe_net = load_caffe_model()

    for uploaded_file in uploaded_files:
        grayscale_image = Image.open(uploaded_file).convert("L")
        input_image = Image.open(uploaded_file).resize((256, 256))

        if model_type == "GAN":
            prediction_image = predict_image(generator, grayscale_image)
            prediction_image = Image.fromarray((prediction_image * 255).astype(np.uint8))
        elif model_type == "CNN":
            prediction_image = colorize_with_caffe(caffe_net, grayscale_image)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(input_image, caption="Gambar Asli (Ground Truth))", use_container_width=True)
        with col2:
            st.image(grayscale_image.resize((256, 256)), caption="Gambar Grayscale (Input)", use_container_width=True)
        with col3:
            st.image(prediction_image, caption="Gambar Hasil Generate", use_container_width=True)

        img_bytes = io.BytesIO()
        prediction_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        st.download_button(
            label="Download Gambar Hasil Prediksi",
            data=img_bytes,
            file_name=f"predicted_{uploaded_file.name}",
            mime="image/png",
            key=f"download_{str(uuid.uuid4())}"
        )
