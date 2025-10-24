import os, base64, tempfile, re, joblib, requests
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz, distance
import pandas as pd
import scipy.sparse as sp

# Window size for context summary update
CONTEXT_SUMMARY_WINDOW_SIZE = 6

# --- env ---
load_dotenv()
# Azure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT") or st.secrets.get("OPENAI_API_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION") or st.secrets.get("OPENAI_API_VERSION")
OPENAI_API_DEPLOYMENT = os.getenv("OPENAI_API_DEPLOYMENT") or st.secrets.get("OPENAI_API_DEPLOYMENT")
# Azure Custom Vision
CUSTOM_VISION_PREDICTION_ENDPOINT = os.getenv("CUSTOM_VISION_PREDICTION_ENDPOINT") or st.secrets.get("CUSTOM_VISION_PREDICTION_ENDPOINT")
CUSTOM_VISION_PREDICTION_KEY = os.getenv("CUSTOM_VISION_PREDICTION_KEY") or st.secrets.get("CUSTOM_VISION_PREDICTION_KEY")
# Azure Speech Service
SPEECH_SERVICE_KEY = os.getenv("SPEECH_SERVICE_KEY") or st.secrets.get("SPEECH_SERVICE_KEY")
SPEECH_SERVICE_REGION = os.getenv("SPEECH_SERVICE_REGION") or st.secrets.get("SPEECH_SERVICE_REGION")

# --- session state ---
def init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "dog_profile" not in st.session_state:
        st.session_state["dog_profile"] = {}
    if "tts_on" not in st.session_state:
        st.session_state["tts_on"] = False
    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None
    if "ctx" not in st.session_state:
        st.session_state["ctx"] = {
            "summary": "",
            "msg_count": 0,
            "last_summary_index": 0,
        }
    if "pending_chat_info" not in st.session_state:
        st.session_state["pending_chat_info"] = {
            "need_reply": False,
            "pending_user_idx": -1,
            "quick_action": "",
        }
    if "current_audio_bytes" not in st.session_state:
        st.session_state["current_audio_bytes"] = None
    if "audio_input_info" not in st.session_state:
        st.session_state["audio_input_info"] = {
            "voice_key": 0,
            "pending_audio": None,
        }

# --- reset session ---
def reset_session():
    st.session_state.clear()
    st.switch_page("app.py")
    # init_state()

def detect_quick_action(user_msg: str, threshold: float = 0.30) -> Optional[str]:
    if not (user_msg or "").strip():
        return None

    # normalize typos
    user_msg = _normalize_typos(user_msg)

    # compute cosine similarity
    X_user = _vectorizer.transform([user_msg.lower()])
    sims = cosine_similarity(X_user, _X_intents)[0]

    order = sims.argsort()[::-1]
    best_idx = int(order[0])
    second_idx = int(order[1])
    best, second = float(sims[best_idx]), float(sims[second_idx])

    # require passing the threshold AND a small lead over runner-up
    if best >= threshold and (best - second) >= 0.02:
        return _intent_keys[best_idx]
    return None

# --- Azure Custom Vision ---
# Classify dog breed
def classify_dog_breed(image_bytes: bytes) -> Optional[Dict]:
    headers = {
        "Content-Type": "application/octet-stream",
        "Prediction-Key": CUSTOM_VISION_PREDICTION_KEY,
    }
    try:
        response = requests.post(
            CUSTOM_VISION_PREDICTION_ENDPOINT,
            headers=headers,
            data=image_bytes,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error classifying dog breed: {e}")
        return None
    except ValueError as ve:
        print(f"Error parsing classification response: {ve}")
        return None

# --- Azure Speech Service ---
# Transcribe audio to text
def transcribe_audio(audio_bytes: bytes, mime: str = "") -> str | None:
    if not audio_bytes or len(audio_bytes) < 800:
        return None

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_SERVICE_KEY,
        region=SPEECH_SERVICE_REGION
    )

    try:
        if (mime in ("audio/wav", "audio/x-wav")) or _is_wav_bytes(audio_bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            try:
                audio_config = speechsdk.AudioConfig(filename=path)
                rec = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
                res = rec.recognize_once()
            finally:
                try: os.remove(path)
                except: pass
        else:
            cmap = {
                "audio/webm": speechsdk.audio.AudioStreamContainerFormat.WEBM_OPUS,
                "audio/ogg":  speechsdk.audio.AudioStreamContainerFormat.OGG_OPUS,
                "audio/mpeg": speechsdk.audio.AudioStreamContainerFormat.MP3,
                "audio/mp3":  speechsdk.audio.AudioStreamContainerFormat.MP3,
                "audio/flac": speechsdk.audio.AudioStreamContainerFormat.FLAC,
            }
            fmt = cmap.get((mime or "").lower(), speechsdk.audio.AudioStreamContainerFormat.MP3)
            sformat = speechsdk.audio.AudioStreamFormat(compressed_stream_format=fmt)
            pstream = speechsdk.audio.PushAudioInputStream(sformat)
            pstream.write(audio_bytes); pstream.close()
            audio_config = speechsdk.AudioConfig(stream=pstream)
            rec = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            res = rec.recognize_once()

        if res.reason == speechsdk.ResultReason.RecognizedSpeech:
            return res.text
        else:
            print("ASR failed:", res.reason, getattr(res, "error_details", ""))
            return None
    except Exception as e:
        print("ASR exception:", e)
        return None
    
# Synthesize speech from text
def synthesize_speech(text: str) -> str:
    try:
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_SERVICE_KEY, region=SPEECH_SERVICE_REGION)

        # audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        audio_config = speechsdk.audio.PullAudioOutputStream()

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        else:
            print(f"Speech synthesis failed: {result.reason}")
            return ""
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return ""

# --- Azure OpenAI ---
# Get OpenAI client
def get_openai_client() -> AzureOpenAI:
    client = AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_ENDPOINT,
        api_key=OPENAI_API_KEY
    )
    return client

# Send request to OpenAI and get response
def send_openai_request(messages: List[Dict], max_tokens: int, temperature: float) -> str:
    # check if context summary needs update
    try:
        if st.session_state["ctx"]["msg_count"] - st.session_state["ctx"]["last_summary_index"] >= CONTEXT_SUMMARY_WINDOW_SIZE:
            update_summary_context()
    except Exception as e:
        print(f"Error updating summary context: {e}")

    # Update system prompt with context summary and recent unsummarized messages
    ctx_sum = st.session_state["ctx"]["summary"]
    unsummarized_msgs = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state["messages"][st.session_state["ctx"]["last_summary_index"] :]]
    )

    messages[0]["content"] = (
                f"{messages[0]['content']}\n\n"
                f"Current Conversation Summary: . If there is no summary available, you may ignore it.\n{ctx_sum}\n\n"
                f"Here are unsummarized recent messages for context:\n"
                f"{unsummarized_msgs} . if there are no unsummarized messages, you may ignore this.\n"
                "Incorporate relevant information from the these messages into your responses to user queries."
            )

    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=OPENAI_API_DEPLOYMENT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        print(f"Error sending OpenAI request: {e}")
        return ""
    return response.choices[0].message.content

# Update conversation summary context
def update_summary_context():
    window_size_threshold = CONTEXT_SUMMARY_WINDOW_SIZE
    ctx = st.session_state["ctx"]
    window_size = ctx["msg_count"] - ctx["last_summary_index"]
    if window_size < window_size_threshold:
        return
    
    recent_msgs = st.session_state["messages"][-window_size:]
    recent_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])

    old_summary = ctx.get("summary", "")
    
    system = (
        "You are WoofTalk, a friendly and knowledgeable dog expert. "
        "Merge the new chat into the existing context summary for future turns. "
        "Keep key facts, dog details (like breed, age, symptoms, characteristics, and more), user concerns, open questions, and next steps. "
        "Be concise (~200 words), deduplicate, and correct contradictions."
    )

    user = (
        f"Please summarize the following conversation:\n\n{recent_text}"
        f"\n\nExisting Summary:\n{old_summary}\n\n"
        "Provide an updated summary that incorporates any new information from the recent messages and existing summary."
    )

    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=OPENAI_API_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=450,
            temperature=0.3,
        )
    except Exception as e:
        print(f"Error updating summary context: {e}")
        return

    if response:
        new_summary = response.choices[0].message.content
        ctx["summary"] = new_summary
        ctx["last_summary_index"] = ctx["msg_count"]
        # print(f"Conversation summary updated: {ctx['summary']}")

# Generate recognized dog's basic breed info and care tips
def generate_basic_breed_info() -> str:
    dog_breed = st.session_state["dog_profile"].get("breed", "Unknown")
    dog_confidence = st.session_state["dog_profile"].get("confidence", 0)

    system = (
        "You are WoofTalk, a friendly and knowledgeable dog expert. "
        "Provide concise and informative responses about dog breeds and care tips. "
        "You can use bullet points for this response. Keep your answers under 150-250 words."
    )
    user = (
        f"Please give a brief information about the recognized dog breed '{dog_breed}'. You can refer to the following format as sample response:\n\n"
        f"Your dog is {dog_confidence:.2%} likely to be a {dog_breed}.\n\n"
        f"Here is some information about {dog_breed}:\n"
        f"- Characteristics: ...\n"
        f"- Temperament: ...\n\n"
        f"Daily Care Tips for {dog_breed}:\n"
        f"- Tips 1: ...\n"
        f"- Tips 2: ...\n"
        f"- Tips 3: ...\n"
        f"You should change the Tips 1, 2, 3 headings to be more descriptive.\n"
        f"You can add a sentence to warn that these tips are general guidelines and may not apply to every individual dog. And also remind them to consult with a veterinarian for specific advice. You can have more or less tips as needed. Those tips should focus on daily care, common health issues, and training advice.\n"
        "Please make sure the response is easy to understand and engaging for dog owners. Also Please ensure your words count is between 150-250 words and no grammatical errors."
    )

    response = send_openai_request(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=500,
        temperature=0.7,
    )

    return response

# Generate response for user messages
def generate_general_response(user_msg: str) -> str:
    dog_breed = st.session_state["dog_profile"].get("breed", "Unknown")
    activity_level = st.session_state["dog_profile"].get("activity_level", "Unknown")
    diet = st.session_state["dog_profile"].get("diet", "Unknown")
    play_time = st.session_state["dog_profile"].get("play_time_hrs", "Unknown")
    health_status = st.session_state["dog_profile"].get("healthy_txt", "Unknown")

    system = (
        "You are WoofTalk, a friendly and knowledgeable dog expert. "
        "Provide concise and informative responses about dog breeds, care tips, and user inquiries. "
        f"User have questions related to their dog, the user's dog's breed is '{dog_breed}', daily activity level is '{activity_level}', diet is '{diet}', play time is '{play_time}', and health status is '{health_status}'. "
        "Please try to keep the responses under 80-100 words. "
        "If you don't know the answer or need more information, politely inform the user or kindly ask for clarification. "
        "Please use appropriate grammar and formatting for better readability. "
        "Please remind the user that these are general guidelines and may not apply to every individual dog. "
    )
    user = (
        f"Please provide a relevant response to this question: {user_msg}."
    )

    response = send_openai_request(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=400,
        temperature=0.7,
    )

    return response

# Generate response for quick actions
def generate_quick_action_response(action: str) -> str:
    # read the summary context
    sum_ctx = st.session_state["ctx"].get("summary", "")
    # read unsummarized recent messages
    unsum_msgs = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state["messages"][st.session_state["ctx"]["last_summary_index"] :]]
    )

    # combine context
    context = f"{sum_ctx}\n{unsum_msgs}"

    # predicted context to get dog's daily activity level, diet, play time (hrs), and healthy status
    preds = predict_all_from_text(context)

    # Update dog profile with predictions, using more descriptive keys and fallback values
    profile = st.session_state["dog_profile"]
    profile["activity_level"] = preds.get("Daily Activity Level", "Unknown")
    profile["diet"] = preds.get("Diet", "Unknown")
    profile["play_time_hrs"] = round(preds.get("Play Time (hrs)", 0), 2) if isinstance(preds.get("Play Time (hrs)"), (int, float)) else "Unknown"
    profile["healthy_txt"] = preds.get("Healthy", "Unknown")
    profile["healthy_prob"] = preds.get("Healthy_Prob", None)

    dog_breed = profile.get("breed", "Unknown")
    activity_level = profile.get("activity_level", "Unknown")
    diet = profile.get("diet", "Unknown")
    play_time = profile.get("play_time_hrs", "Unknown")
    health_status = profile.get("healthy_txt", "Unknown")

    # rag for knowledge base
    kb_snips = _retrieve_kb_snippets(context, action, topk=4, min_sim=0.3)
    kb_context = _format_snippets_to_prompt(kb_snips, max_chars=1200)
    if kb_context:
        context_intro = (
            "Here are some relevant information from the WoofTalk knowledge base to help you answer the user's question:\n"
            f"{kb_context}\n\n"
            "Incorporate relevant information from the above knowledge base snippets into your responses to user queries."
        )
    else:
        context_intro = ""

    system = (
        "You are WoofTalk, a friendly and knowledgeable dog expert. "
        "Provide concise and informative responses for the user's quick action request. "
        f"The user's dog's breed is '{dog_breed}', daily activity level is '{activity_level}', diet is '{diet}', play time is '{play_time}', and health status is '{health_status}'. {context_intro}"
        "Keep your answer under 100-150 words, use clear grammar and formatting, and remind the user these are general guidelines that may not apply to every individual dog."
    )

    if action == "training_guidance":
        user = (
            f"Please provide effective training methods, common challenges, and solutions. If you need more information, please ask for clarification."
        )
    elif action == "health_assessment":
        user = (
            f"Please provide a general assessment of this dog's health, and provide any common health issues and preventive care tips for these specific symptoms if user mentions them. If you need more information, please ask for clarification."
        )
    elif action == "exercise_routine":
        user = (
            f"Please provide suitable exercise routines, frequency, and intensity levels for this breed. If you need more information, please ask for clarification."
        )
    elif action == "nutrition_recommendation":
        user = (
            f"Please provide dietary recommendations, portion sizes, and any breed-specific nutritional needs. If you need more information, please ask for clarification."
        )

    response = send_openai_request(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=400,
        temperature=0.7,
    )

    return response

# --- Audio utils ---
# check if bytes represent WAV audio
def _is_wav_bytes(b: bytes) -> bool:
    return len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WAVE"

# convert bytes to data URL for wav audio
def _to_data_url_wav(b: bytes) -> str:
    if not b: return ""
    return "data:audio/wav;base64," + base64.b64encode(b).decode("utf-8")

# Mount audio iframe
def _mount_audio_iframe(holder, data_url: str):
    if not data_url:
        holder.empty()
        return
    html = f"""
    <html><head><meta charset="utf-8">
      <style>html,body{{margin:0;padding:0;overflow:hidden;}}</style>
    </head><body>
      <audio id="tts" autoplay playsinline>
        <source src="{data_url}" type="audio/wav">
      </audio>
      <script>
        (function(){{
          const a = document.getElementById('tts');
          if (a) {{
            a.loop = false;
            a.play && a.play().catch(()=>{{}});
          }}

          // collapse iframe and parent container after playback
          const fr = window.frameElement;
          if (fr) {{
            // collapse iframe itself
            fr.style.width='0px';
            fr.style.height='0px';
            fr.style.border='0';
            fr.style.position='absolute';
            fr.style.left='-99999px';
            fr.style.top='0';
            fr.style.opacity='0';
            fr.style.pointerEvents='none';

            // collapse parent container
            const parent = fr.parentElement;
            if (parent && parent.style) {{
              parent.style.width='0px';
              parent.style.height='0px';
              parent.style.margin='0';
              parent.style.padding='0';
              parent.style.border='0';
              parent.style.overflow='hidden';
              parent.style.position='absolute';
              parent.style.left='-99999px';
              parent.style.top='0';
              parent.style.opacity='0';
              parent.style.pointerEvents='none';
            }}
          }}
        }})();
      </script>
    </body></html>
    """
    with holder:
        components.html(html, height=0, scrolling=False)

# Unmount audio iframe
def _unmount_audio_iframe(holder):
    holder.empty()

# --- Detect quick action from user message ---
# st.cache_resource.clear()
INTENT_DOCS = {
    "training_guidance": """
        train training obedience command commands cue clicker behavior behaviour discipline correct fix stop
        barking bark bite biting jump chew leash pulling recall sit stay down heel crate potty housebreaking
        teach how to teach how do i stop how can i stop how to stop stop my dog from
        socialization reactivity counterconditioning impulse control marker reward shaping
    """,
    "health_assessment": """
        health symptom symptoms health condition health assessment assess assessment analysis analyse analyze evaluate evaluation
        check vet clinic examination checkup is this normal
        injury pain wound fever cough diarrhea diarrhoea vomit vomiting nausea rash itch itchy hotspot limp parasite
        tick flea ear infection eye discharge dehydration gums vaccine vaccination deworm wellness emergency
        analyze my dog's condition analyse my dog's condition assess my dog's condition evaluate my dog's condition
        analyze dog condition assess dog condition evaluate dog condition dog's condition dog condition
    """,
    "exercise_routine": """
        exercise workout activity walk run jog hike play fetch frisbee tug treadmill swim
        schedule routine frequency intensity duration minutes minute hours hour daily weekly per day per week
        energy enrichment bored step steps km kilometer kilometers mile miles distance
        how many minutes how long how far how often
        how often walk dog how often to walk my dog how often should i walk my dog walks per day walk per day
        walking schedule walking frequency daily walks puppy walk frequency
    """,
    "nutrition_recommendation": """
        nutrition diet food feed feeding meal meals portion portions ingredients ingredient kibble wet raw treats
        calories calorie kcal protein fat carb carbohydrate allergy intolerance sensitive formula ratio
        how much should i feed what should i feed how many grams cup cups per day per meal
        prepare preparing prepared cook cooking recipe recipes meal prep mealplan meal plan homecooked home-cooked homemade
        balanced complete ration menu bowl topper add-ins rotation
        dog food recipe dog recipes homemade dog food recipe recipes
        share recipe prepare a great meal prepare a meal cook a meal make a meal
    """,
}
_intent_keys = list(INTENT_DOCS.keys())
_intent_texts = list(INTENT_DOCS.values())

# Fit vectorizer and transform intent texts. Use char_wb to be robust to typos.
try:
    @st.cache_resource
    def _get_intent_space():
        vec = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(4, 6), 
            min_df=1,
        )
        X = vec.fit_transform(_intent_texts) 
        return vec, X

    _vectorizer, _X_intents = _get_intent_space()
except Exception:
    # fallback if not in Streamlit context; keep settings identical
    _vectorizer = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(4, 6), min_df=1)
    _X_intents = _vectorizer.fit_transform(_intent_texts)

# For typo normalization
@st.cache_resource
def _get_intent_vocab():
    vocab = set()
    for doc in _intent_texts:
        for w in re.findall(r"[A-Za-z']+", (doc or "").lower()):
            if len(w) >= 3:
                vocab.add(w)
    return sorted(vocab)

_INTENT_VOCAB = _get_intent_vocab()

# rapidfuzz for typo correction
def _normalize_typos(text: str) -> str:
    s = (text or "").lower()
    if not _INTENT_VOCAB:
        return s
    pat = re.compile(r"[A-Za-z']{4,}")
    def repl(m):
        tok = m.group(0)
        cand, score, _ = process.extractOne(tok, _INTENT_VOCAB, scorer=fuzz.WRatio)
        if score >= 88 or (score >= 80 and distance.Levenshtein.distance(tok, cand) <= 2):
            return cand
        return tok
    return pat.sub(repl, s)

# --- Model loading ---
# Load task models
@st.cache_resource
def _load_task_models():
    env_path = os.getenv("MODEL_PATH", "").strip() or st.secrets.get("MODEL_PATH", "").strip()
    # print(f"Loading models from MODEL_PATH={env_path}")
    if env_path:
        try:
            return joblib.load(env_path)
        except Exception as e:
            print(f"Failed to load model from MODEL_PATH={env_path}: {e}")
    return None
_MODELS = _load_task_models()

# feature extraction from user text
def _extract_numbers_from_text(text: str):
    if not text: 
        return {}
    t = text.lower()
    out = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(year|years|yr|yrs)", t)
    if m: out["Age"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(month|months|mo)", t)
    if m and "Age" not in out: out["Age"] = float(m.group(1))/12.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lb|lbs|pound|pounds)", t)
    if m: out["Weight (lbs)"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilogram|kilograms)", t)
    if m and "Weight (lbs)" not in out: out["Weight (lbs)"] = float(m.group(1))*2.20462
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mile|miles)", t)
    if m: out["Daily Walk Distance (miles)"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(km|kilometer|kilometers)", t)
    if m and "Daily Walk Distance (miles)" not in out: out["Daily Walk Distance (miles)"] = float(m.group(1))*0.621371
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\s*(?:of)?\s*sleep", t)
    if m: out["Hours of Sleep"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\s*(?:of)?\s*play", t)
    if m: out["Play Time (hrs)"] = float(m.group(1))
    return out

# Build feature DataFrame from user text and context
def build_feature_frame_from_nl(user_text: str, ctx: dict) -> pd.DataFrame:
    profile = ctx.get("dog_profile", {}) if ctx else {}
    row = {
        "Breed": profile.get("breed"),
        "Breed Size": profile.get("size"),
        "Sex": profile.get("sex"),
        "Spay/Neuter Status": profile.get("spay_neuter"),
        "Age": profile.get("age_years"),
        "Weight (lbs)": profile.get("weight_lbs"),
        "Daily Walk Distance (miles)": profile.get("daily_walk_miles"),
        "Hours of Sleep": profile.get("sleep_hours"),
        "Annual Vet Visits": profile.get("annual_vet_visits"),
        "Average Temperature (F)": profile.get("avg_temp_f"),
        "notes": profile.get("notes", ""),
        "user_text": user_text or "",
        # the following will be overwritten if found in text
        "Diet": profile.get("diet"),
        "Daily Activity Level": profile.get("activity_level"),
        "Play Time (hrs)": profile.get("play_time_hrs"),
        "Healthy": profile.get("healthy_txt"),
    }
    row.update(_extract_numbers_from_text(user_text or ""))
    row["__joined_text__"] = (str(row.get("notes") or "") + " " + str(row.get("user_text") or "")).strip()
    return pd.DataFrame([row])

# TF-IDF transform
def _tfidf_transform(model_dict, X):
    tf = model_dict["tfidf"]
    return tf.transform(X)

# Predict all attributes from user text
def predict_all_from_text(user_text: str) -> dict:
    X = build_feature_frame_from_nl(user_text, st.session_state)
    out = {}

    # Activity
    m = _MODELS["activity"]
    Xa = m["ct"].transform(X); Ta = _tfidf_transform(m, X)
    Pa = sp.hstack([Xa, Ta])
    pred_act = m["label_encoder"].inverse_transform(m["clf"].predict(Pa))
    out["Daily Activity Level"] = pred_act[0]

    # Diet
    m = _MODELS["diet"]
    Xd = m["ct"].transform(X); Td = _tfidf_transform(m, X)
    Pd = sp.hstack([Xd, Td])
    pred_diet = m["label_encoder"].inverse_transform(m["clf"].predict(Pd))
    out["Diet"] = pred_diet[0]

    # Play Time (hrs)
    m = _MODELS["play"]
    Xp = m["ct"].transform(X)
    Tp = _tfidf_transform(m, X)
    Pp = sp.hstack([Xp, Tp])

    # convert sparse to dense if needed
    if sp.issparse(Pp):
        Pp = Pp.toarray()

    out["Play Time (hrs)"] = float(m["reg"].predict(Pp)[0])

    # Healthy
    m = _MODELS["healthy"]
    Xh = m["ct"].transform(X); Th = _tfidf_transform(m, X)
    Ph = sp.hstack([Xh, Th])
    if hasattr(m["clf"], "predict_proba"):
        p = float(m["clf"].predict_proba(Ph)[:,1][0])
        out["Healthy_Prob"] = round(p, 3)
        out["Healthy"] = "Yes" if p >= 0.5 else "No"
    else:
        yhat = int(m["clf"].predict(Ph)[0])
        out["Healthy"] = "Yes" if yhat==1 else "No"

    # print(f"Predicted from text: {out}")    
    return out

# --- Simple RAG ---
# Knowledge base directory and file map
KB_DIR = os.getenv("KB_DIR") or st.secrets.get("KB_DIR")
_KB_FILE_MAP = {
    "health_assessment":       "healthcare.md",
    "exercise_routine":        "exercise.md",
    "training_guidance":       "training.md",
    "nutrition_recommendation":"nutrition.md",
}

# read and chunk markdown file
def _chunk_markdown(text: str, max_chars: int = 800) -> list[dict]:
    text = text or ""
    lines = text.splitlines()
    chunks = []
    cur_title = ""
    buf = []
    # flush buffer into chunks
    def flush_buf():
        nonlocal chunks, buf, cur_title
        if not buf: 
            return
        block = "\n".join(buf).strip()
        if not block:
            buf = []
            return
        # split into chunks
        start = 0
        while start < len(block):
            piece = block[start:start+max_chars]
            chunks.append({
                "title": cur_title,
                "text": piece.strip()
            })
            start += max_chars
        buf = []

    for ln in lines:
        if ln.strip().startswith("#"):
            # new section
            flush_buf()
            cur_title = ln.strip().lstrip("#").strip()
        elif ln.strip() == "":
            # paragraph separator
            flush_buf()
        else:
            buf.append(ln)
    flush_buf()
    return chunks

# load kb file and chunk
def _load_kb_file(filepath: str) -> list[dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return []
    return _chunk_markdown(text)

# build kb index with TF-IDF
@st.cache_resource
def _build_kb_index():
    index = {}
    for action, fname in _KB_FILE_MAP.items():
        path = os.path.join(KB_DIR, fname)
        if not os.path.isfile(path):
            index[action] = {"chunks": [], "vec": None, "X": None, "path": path}
            continue
        chunks = _load_kb_file(path)
        # use TF-IDF vectorizer to index chunks
        vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1,2),
            max_features=50000,
            analyzer="word",
        )
        docs = [c["text"] for c in chunks]
        if len(docs) == 0:
            index[action] = {"chunks": [], "vec": None, "X": None, "path": path}
            continue
        X = vec.fit_transform(docs)
        index[action] = {"chunks": chunks, "vec": vec, "X": X, "path": path}
    return index

_KB_INDEX = _build_kb_index()

# retrieve kb snippets for query and action
def _retrieve_kb_snippets(query: str, action: str, topk: int = 4, min_sim: float = 0.08) -> list[dict]:
    entry = _KB_INDEX.get(action) or {}
    vec, X, chunks = entry.get("vec"), entry.get("X"), entry.get("chunks")
    if not vec or X is None or not chunks:
        return []
    # normalize typos in query
    q = _normalize_typos((query or "").strip().lower())
    qv = vec.transform([q])
    sims = cosine_similarity(qv, X)[0]
    order = sims.argsort()[::-1]
    out = []
    for idx in order[:topk]:
        sc = float(sims[idx])
        if sc < min_sim:
            continue
        ch = chunks[idx]
        out.append({"title": ch.get("title") or "", "text": ch.get("text") or "", "score": sc})
    return out

# format snippets into prompt string
def _format_snippets_to_prompt(snips: list[dict], max_chars: int = 1200) -> str:
    if not snips:
        return ""
    buf = []
    total = 0
    for i, s in enumerate(snips, 1):
        head = f"• {s['title']}: " if s.get("title") else "• "
        body = s["text"].strip().replace("\n", " ")
        piece = (head + body).strip()
        if total + len(piece) > max_chars:
            break
        buf.append(piece)
        total += len(piece)
    return "\n".join(buf)