import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from openai import OpenAI
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_v2_preprocess_input,
)

st.set_page_config(
    page_title="My Website AI",
    page_icon=":robot_face:",
    layout="wide",  # Atur tata letak halaman menjadi wide
    initial_sidebar_state="expanded",  # Sidebar akan diperluas secara default
)

with st.sidebar:
    st.sidebar.image(
        "icon.png",
        width=25,
        use_column_width=True,
    )
    pages_dict = option_menu(
        menu_title="Main Menu",
        options=["Home", "Image Classification", "ChatBot AI"],
        icons=["house", "image", "chat"],
        default_index=0,
        styles={
            "icon": {"color": "white", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#666",
                "font-family": "'Arial', sans-serif",
            },
            "menu-title": {
                "font-size": "16px",
            },
        },
    )
    st.markdown("")
    st.caption("Copyright (c) - Created by mfajarjati - 2024")

if pages_dict == "Home":
    st.write(
        "<h1 style='text-align: center;'>Selamat Datang di Program Image Classification dan ChatBot AI 🚀 </h1>",
        unsafe_allow_html=True,
    )
    st.header("", divider="rainbow")
    st.markdown(
        """
    Program ini menyediakan dua fitur utama: **Image Classification** dan **ChatBot AI**. Dengan kedua fitur ini, Anda dapat menjelajahi dan mengalami kecerdasan buatan dalam dua konteks yang berbeda.

    ## Image Classification 🖼️ 

    Image Classification, Anda dapat mengunggah gambar hewan dan melihat bagaimana sistem kami mengklasifikasikan gambar tersebut. Program ini dilatih dengan dataset berbagai hewan, seperti anjing, kuda, gajah, dan banyak lagi. Saksikan kemampuan model dalam mengenali dan mengklasifikasikan objek berdasarkan gambar yang Anda unggah.

    ## ChatBot AI 🤖

    ChatBot AI memungkinkan Anda berinteraksi dengan ChatBot cerdas berbasis model OpenAI. Tanyakan pertanyaan atau ajak ChatBot untuk memulai percakapan. Nikmati respons real-time yang dihasilkan oleh model kecerdasan buatan terkini.

    ### How To Use
    1. Pilih opsi pada Sidebar untuk beralih antara Image Classification dan ChatBot AI.
    2. Ikuti petunjuk yang disediakan di setiap bagian untuk penggunaan yang optimal.
    3. Nikmati pengalaman berinteraksi dengan teknologi kecerdasan buatan!

    Jelajahi fitur-fitur menarik, dan mari kita mulai petualangan ke dunia AI bersama!
    """
    )


elif pages_dict == "Image Classification":
    st.write(
        "<h1 style='text-align: center;'>Image Classification 🖼️ </h1>",
        unsafe_allow_html=True,
    )

    model = tf.keras.models.load_model("saved_model/mdl_wts.hdf5")
    ### load file
    st.header("", divider="rainbow")
    import streamlit as st

    # Tentang Image Classification
    st.markdown(
        """
    ## About Image Classification
    Image Classification merupakan sebuah teknik dalam bidang kecerdasan buatan yang memungkinkan komputer untuk mengidentifikasi dan mengklasifikasikan objek atau pola dalam gambar. Tujuan utamanya adalah mengajarkan mesin untuk mengenali dan memahami berbagai kategori atau label yang dapat muncul dalam gambar.
    """
    )

    # Data Hewan yang Dilatih
    st.markdown(
        """
    ### Data Hewan yang Dilatih
    Pada halaman ini, kami menyediakan model Image Classification yang telah dilatih dengan dataset berisi berbagai gambar hewan. Dataset ini mencakup beragam spesies, seperti anjing, kuda, gajah, kupu-kupu, ayam, kucing, sapi, domba, laba-laba, dan tupai. Model ini dapat membantu Anda mengidentifikasi dan mengklasifikasikan gambar-gambar hewan tersebut.
    """
    )

    # Cara Menggunakan
    st.markdown(
        """
    ### How To Use
    1. Gunakan fitur unggah file dibawah ini untuk memilih gambar yang ingin Anda identifikasi. Pastikan gambar tersebut merupakan hewan serta jelas dan sesuai dengan kategori yang telah dilatih.
    2. Klik tombol "Generate Prediction" untuk memulai proses klasifikasi. Model akan memberikan prediksi tentang kategori atau jenis hewan yang ada dalam gambar.

    Selamat mencoba dan nikmati pengalaman interaktif dalam mengklasifikasikan gambar hewan!
    """
    )

    st.markdown("")
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    map_dict = {
        0: "dog (Anjing)",
        1: "horse (Kuda)",
        2: "elephant (Gajah)",
        3: "butterfly (Kupu-kupu)",
        4: "chicken (Ayam)",
        5: "cat (Kucing)",
        6: "cow (Sapi)",
        7: "sheep (Domba)",
        8: "spider (Laba-laba)",
        9: "squirrel (Tupai)",
    }

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image, (224, 224))
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="RGB")

        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis, ...]

        Genrate_pred = st.button("Generate Prediction")
        if Genrate_pred:
            prediction = model.predict(img_reshape).argmax()
            st.title(
                "Label yang Diprediksi untuk gambar ini adalah {}".format(
                    map_dict[prediction]
                )
            )


elif pages_dict == "ChatBot AI":
    st.write(
        "<h1 style='text-align: center;'>ChatBot AI 🤖 </h1>",
        unsafe_allow_html=True,
    )
    st.header("", divider="rainbow")

    # Inisialisasi OpenAI
    st.markdown(
        """
            ### About ChatBot AI
            ChatBot AI menggunakan teknologi OpenAI GPT-3.5 Turbo untuk memberikan respon yang cerdas berdasarkan teks atau prompt yang Anda berikan. 
            Silakan masukkan pertanyaan atau percakapan Anda ke dalam kotak input, dan ChatBot AI akan memberikan respons sesuai dengan konteks.
        """
    )

    st.markdown(
        """
            ### How To Use
            1. Gunakan kotak input "What is up?" untuk memasukkan pertanyaan atau percakapan Anda.
            2. Setelah Anda memasukkan teks atau prompt, ChatBot AI akan memberikan respons secara real-time.
            3. Anda dapat terus berinteraksi dengan ChatBot AI dengan menambahkan lebih banyak pertanyaan atau percakapan. 
            4. Jika Anda ingin membersihkan percakapan dan memulai percakapan baru, cukup segarkan pada halaman web.
            
            Nikmati percakapan cerdas dan interaktif dengan ChatBot AI!
        """
    )

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Mengecek dan menginisialisasi state untuk model dan pesan
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Menampilkan pesan-pesan sebelumnya
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Menerima input dari pengguna
    if prompt := st.chat_input("What is up?"):
        # Menyimpan pesan pengguna ke dalam sesi
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Menampilkan pesan pengguna
        with st.chat_message("user"):
            st.markdown(prompt)

        # Menampilkan respons dari ChatBot AI secara real-time
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")

            # Setelah respons selesai, menampilkan respons lengkap tanpa tanda '▌'
            message_placeholder.markdown(full_response)

        # Membersihkan pesan-pesan pengguna setelah respons selesai
        st.session_state.messages = []
