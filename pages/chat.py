import streamlit as st
from utils import (
    init_state,
    generate_basic_breed_info,
    reset_session,
    generate_general_response,
    generate_quick_action_response,
    transcribe_audio,
    synthesize_speech,
    _mount_audio_iframe,
    _unmount_audio_iframe,
    _to_data_url_wav,
    detect_quick_action
)

# page config
st.set_page_config(
    page_title="WoofTalk • Chat",
    layout="centered",
    page_icon="🐾",
    initial_sidebar_state="collapsed",
)

# Init session state
init_state()

# Function to render chat area with initial breed info message
def render_chat_area(content: str, role: str = "user"):
    st.session_state["messages"].append({"role": role, "content": content})
    st.session_state["ctx"]["msg_count"] += 1

    # If the message is from user, set flags to trigger assistant reply generation
    if role == "user":
        st.session_state["pending_chat_info"]["need_reply"] = True
        st.session_state["pending_chat_info"]["pending_user_idx"] = len(st.session_state["messages"]) - 1

    # If the message is from assistant and TTS is on, generate and play audio
    if role == "assistant":
        if st.session_state.get("tts_on", False):
            audio_bytes = synthesize_speech(content)
            st.session_state["current_audio_bytes"] = audio_bytes if audio_bytes else None
        else:
            st.session_state["current_audio_bytes"] = None

    st.rerun()

# Function to trigger response generation from assistant
def trigger_response_generation():
    msgs = st.session_state["messages"]
    pending_chat_info = st.session_state.get("pending_chat_info", {})

    if (
        pending_chat_info.get("need_reply", False)
        and msgs
        and pending_chat_info.get("pending_user_idx", -1) == len(msgs) - 1
        and msgs[-1]["role"] == "user"
    ):
        last_user_text = msgs[-1]["content"]
        quick_action = pending_chat_info.get("quick_action")  

        # # Detect quick action if not already set
        if not quick_action:
            detected = detect_quick_action(last_user_text)
            if detected:
                # print(f"Detected quick action: {detected}")
                quick_action = detected

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("WoofTalk is thinking..."):
                if quick_action:
                    reply = generate_quick_action_response(quick_action)
                else: 
                    reply = generate_general_response(last_user_text)

            placeholder.markdown(reply)

        pending_chat_info["need_reply"] = False
        pending_chat_info["pending_user_idx"] = -1
        pending_chat_info["quick_action"] = ""

        render_chat_area(reply, role="assistant")


# Generate the first message for user to begin chat with basic info about the recognized dog breed
if st.session_state["dog_profile"] and not st.session_state["messages"]:
    with st.spinner("Loading your dog's breed info..."):
        breed_info = generate_basic_breed_info()

    if breed_info != "":
        render_chat_area(breed_info, role="assistant")
    else:
        render_chat_area("Sorry, we encountered an error while generating your dog's breed info. Please try again later.", role="assistant")
        
# Nav bar area with buttons and avatar
with st.container():
    # Add a hook for custom CSS styling
    st.markdown('<span data-hook="topbar-scope"></span>', unsafe_allow_html=True)

    # Divide the nav bar into three columns
    left, center, right = st.columns([3, 2, 3], gap="small", vertical_alignment="center")

    # Left: TTS toggle
    with left:
        st.toggle("🔊 Audio", key="tts_on")

    # Center: Circular avatar
    with center:
        st.markdown(
            """
            <div style="display:flex;justify-content:center;align-items:center;" data-hook="agent-avatar-scope">
              <img src="https://i.postimg.cc/KzSqTzwT/image.png"
                   alt="avatar"
                   style="width:72px;height:72px;border-radius:50%;object-fit:cover;"/>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Right: New chat button
    with right:
        if st.button("💬 New Chat"):
            reset_session() 

# Chat area
chat_box = st.container(height=450, border=True)
with chat_box:
    # Add a hook for custom CSS styling
    st.markdown('<span data-hook="chat-box-scope"></span>', unsafe_allow_html=True)

    # Display existing messages
    for m in st.session_state["messages"]: 
        with st.chat_message(m["role"]): 
            st.write(m["content"])

    trigger_response_generation()

# Quick actions area
with st.expander("⚡ Quick Actions", expanded=False):
    b1, b2, b3, b4 = st.columns(4, gap="small")
    with b1:
        if st.button("🐕 Training", use_container_width=True):
            st.session_state["pending_chat_info"]["quick_action"] = "training_guidance"
            render_chat_area("Can you provide effective training guidance for my dog?")
    with b2:
        if st.button("🏥 Health", use_container_width=True):
            st.session_state["pending_chat_info"]["quick_action"] = "health_assessment"
            render_chat_area("Can you provide a health assessment for my dog?")
    with b3:
        if st.button("🥏 Exercise", use_container_width=True):
            st.session_state["pending_chat_info"]["quick_action"] = "exercise_routine"
            render_chat_area("Can you provide some exercise routines for my dog?")
    with b4:
        if st.button("🥣 Nutrition", use_container_width=True):
            st.session_state["pending_chat_info"]["quick_action"] = "nutrition_recommendation"
            render_chat_area("Can you provide some nutrition recommendations for my dog?")

# Speaker area
# check if have pending audio to transcribe
if st.session_state["audio_input_info"]["pending_audio"] is not None:
    data = st.session_state["audio_input_info"]["pending_audio"]
    st.session_state["audio_input_info"]["pending_audio"] = None
    # transcribe audio to text and render chat area
    with st.spinner("Transcribing..."):
        text = transcribe_audio(data["bytes"], mime=data.get("mime"))
    render_chat_area(text)
# audio recorder placeholder
recorder_ph = st.empty()
# function to render audio recorder
def render_recorder():
    with recorder_ph.container():
        return st.audio_input(
            " ",
            key=f"voice_input_{st.session_state['audio_input_info']['voice_key']}",
            label_visibility="collapsed",
        )
# render audio recorder and handle new audio input
audio_file = render_recorder()
if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    mime = (getattr(audio_file, "type", "") or "").lower()
    # store pending audio to session state to transcribe in next rerun
    st.session_state["audio_input_info"]["pending_audio"] = {"bytes": audio_bytes, "mime": mime}
    st.session_state["audio_input_info"]["voice_key"] += 1
    _ = render_recorder()
    st.rerun()

# Chat input area
user_text = st.chat_input("Please enter your message here...")
# user_text = st.chat_input("Please enter your message here...", accept_file=True, file_type=["jpg", "jpeg", "png"])
if user_text:
    render_chat_area(user_text)

# Audio slot to control audio bytes for current message
player_holder = st.empty()
if not st.session_state.get("tts_on", False):
    # turn off TTS -> clear current audio bytes and unmount iframe
    st.session_state["current_audio_bytes"] = None
    _unmount_audio_iframe(player_holder)
else:
    b = st.session_state.get("current_audio_bytes") or b""
    # turn on TTS -> mount iframe if have audio bytes
    if b:
        _unmount_audio_iframe(player_holder)
        _mount_audio_iframe(player_holder, _to_data_url_wav(b)) 
    else:
        _unmount_audio_iframe(player_holder)










# Additional CSS styling
st.markdown(
    """
    <style>
        /* hide sidebar */
        section[data-testid="stSidebar"]{ display: none;}

        /* hide header */
        header[data-testid="stHeader"] { display: none !important; }
        div[data-testid="stHeader"] { display: none !important; }

        
        /* adjust whole page layout */
        section[data-testid="stAppScrollToBottomContainer"] > *:nth-child(2) {
            display: none;
        }

        
        /* redefine the main container padding to better utilize space */
        div[data-testid="stMainBlockContainer"] {
            padding: 8px 16px 0px 16px;
        }

        
        /* style nav bar */
        div[data-testid="stLayoutWrapper"]:has([data-hook="topbar-scope"]) > div[data-testid="stVerticalBlock"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
            justify-content: center;
            align-items: center;
        }

        div[data-testid="stLayoutWrapper"]:has([data-hook="topbar-scope"]) {
            padding-bottom: 16px;
        }

        
        /* style chat box */
        div[data-testid="stLayoutWrapper"]:has([data-hook="chat-box-scope"]) {
            padding-bottom: 0px
        }

        
        /* style speaker area */
        div[data-testid="stAudioInput"] > div {
            height: 32px;
            overflow: hidden;
        }

        
        /* responsive setting */
        @media (max-width: 640px) {
            /* hide agent avatar on small screens */
            div[data-testid="stColumn"]:has([data-hook="agent-avatar-scope"]) {
                display: none;
            }

            
            /* adjust padding for nav bar */
            div[data-testid="stLayoutWrapper"]:has([data-hook="topbar-scope"]) {
                padding-bottom: 0px;
            }

            
            /* adjust padding for main container */
            div[data-testid="stMainBlockContainer"] {
                padding-top: 0px;
            }

            
            /* style chat box */
            div[data-testid="stLayoutWrapper"]:has([data-hook="chat-box-scope"]) {
                padding-bottom: 0px
            }

            
            /* adjust speaker area */
            div[data-testid="stAudioInput"] > div {
                height: 28px;
                overflow: hidden;
            }
        }
    </style>
    """, 
    unsafe_allow_html=True
)