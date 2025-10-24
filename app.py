import streamlit as st
from utils import init_state, classify_dog_breed

# page config
st.set_page_config(
    page_title="WoofTalk • Home",
    layout="centered",
    page_icon="🐾",
    initial_sidebar_state="collapsed",
)

# Init session state
init_state()

# Page title & caption
st.title("Let’s meet your pup 🐶")
st.caption("Upload a photo to start a chat about your dog's breed and care tips. We don't store photos beyond this session. 💖")

st.divider()

# Invokes file selector to upload a dog photo
uploaded = st.file_uploader("Upload a photo of your beloved dog to begin. 🐾", type=["jpg", "jpeg", "png"])

# If a photo is provided, save & jump to chat
if uploaded is not None:
    image_bytes = uploaded.read()

    # Only calssify when image changed
    if st.session_state["uploaded_image"] != image_bytes:
        st.session_state["uploaded_image"] = image_bytes
        with st.spinner("Classifying breed..."):
            result = classify_dog_breed(image_bytes)

        predictions = (result or {}).get("predictions", [])

        if not predictions:
            st.error("We couldn't get a response by this time. Please try again later.")

        else:
            top_prediction = predictions[0]
            breed = top_prediction.get("tagName", "Unknown")
            confidence = top_prediction.get("probability", 0)

            if confidence >= 0.7:
                st.session_state["dog_profile"] = {
                    "breed": breed,
                    "confidence": confidence
                }
                st.success(f"Identified breed: **{breed}** with confidence {confidence:.2%}.")
            else:
                st.warning(f"Sorry, the confidence {confidence:.2%} is too low. Please try another photo of your dog for better results.")

    # If breed identified, show button to jump to chat page
    if st.session_state.get("dog_profile"):
        if st.button("Start Chatting!"):
            st.switch_page("pages/chat.py")

# Hide sidebar and header completely
st.markdown(
    """
    <style>
        /* hide sidebar */
        section[data-testid="stSidebar"]{ display: none;}

        /* hide header */
        header[data-testid="stHeader"] { display: none !important; }
        div[data-testid="stHeader"] { display: none !important; }
    </style>
    """, 
    unsafe_allow_html=True
)