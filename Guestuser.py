import streamlit as st
import json
import os
import time
import requests
from langdetect import detect

# ------------------- CONFIG -------------------
COUNTRIES = ["India", "US", "UAE", "China"]
SECTORS = ["Banking", "Finance", "Environment", "Healthcare"]
MAX_QUESTIONS = 15
COOLDOWN_SECONDS = 12 * 60 * 60  # 12 hours
USAGE_FILE = "usage.json"


# ------------------- USAGE FUNCTIONS -------------------
def load_usage():
    if not os.path.exists(USAGE_FILE):
        return {}
    try:
        with open(USAGE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_usage(data):
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)


def check_limit(user_id):
    usage_data = load_usage()
    user_data = usage_data.get(user_id, {"usage_count": 0, "first_use_timestamp": None})
    if user_data["first_use_timestamp"]:
        elapsed = time.time() - user_data["first_use_timestamp"]
        if elapsed > COOLDOWN_SECONDS:
            user_data = {"usage_count": 0, "first_use_timestamp": None}
    usage_data[user_id] = user_data
    save_usage(usage_data)
    return user_data


def update_usage(user_id):
    usage_data = load_usage()
    user_data = usage_data.get(user_id, {"usage_count": 0, "first_use_timestamp": None})
    if user_data["first_use_timestamp"] is None:
        user_data["first_use_timestamp"] = time.time()
    user_data["usage_count"] += 1
    usage_data[user_id] = user_data
    save_usage(usage_data)


# ------------------- LLM RESPONSE -------------------
def stream_answer(question, country, sector):
    url = "http://localhost:11434/api/generate"
    system_prompt = (
        
        f"You are a Knowledge Assistant specializing ONLY in {sector.upper()} for {country.upper()}.\n"
        "Your rules:\n"
        f"- Only answer if the question is directly related to {sector.upper()} in {country.upper()}.\n"
        f"- If the question is unrelated, say: 'I can only answer about {sector.upper()} in {country.upper()}.'\n"
        "- Be concise, factual, and structured in your answers."
        
    )
    data = {
        "model": "mistral:latest",
        "prompt": f"{system_prompt}\n\nUser Question: {question}",
        "stream": True,
        "options": {"num_predict": 300},
    }

    try:
        response = requests.post(url, json=data, stream=True, timeout=180)
        for line in response.iter_lines():
            if line:
                try:
                    part = json.loads(line.decode("utf-8"))
                    chunk = part.get("response", "")
                    if chunk:
                        yield chunk
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to LLM API: {e}"
    yield ""


# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Guest user", layout="wide", page_icon="images/AI_1.png")

with st.sidebar:
    user_id = st.text_input("User ID / Email:")

    if 'email_entered' not in st.session_state:
        st.session_state.email_entered = False
    if 'prev_user_id' not in st.session_state:
        st.session_state.prev_user_id = ""

    if user_id and (user_id != st.session_state.prev_user_id):
        st.toast("✅ Email entered successfully!", duration="short")
        st.session_state.email_entered = True
        st.session_state.prev_user_id = user_id

    if not user_id:
        st.info("ℹ️ Please enter your User ID / Email to start chatting.")
        st.session_state.email_entered = False
        st.session_state.prev_user_id = ""

    with st.sidebar.expander("Settings", expanded=True):
        with st.form("settings_form"):
            country = st.selectbox("Select Country", COUNTRIES)
            sector = st.selectbox("Select Sector", SECTORS)
            save_button = st.form_submit_button("Save")

    if save_button:
        if user_id:
            st.toast("✅ Settings saved successfully!", duration="short")
            st.session_state.settings_saved = True
        else:
            st.warning("Please enter your User ID / Email before saving settings.")
            st.session_state.settings_saved = False

    if user_id:
        user_data = check_limit(user_id)
        st.progress(min(user_data["usage_count"] / MAX_QUESTIONS, 1.0))
        st.caption(f"Limit: {user_data['usage_count']} / {MAX_QUESTIONS} questions")
    else:
        st.session_state.settings_saved = False


st.markdown(
    """
    <style>
    .dark-green-text {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #00325b;
    }
    </style>
    <h1 class="dark-green-text">Fanam Guard</h1>
    """,
    unsafe_allow_html=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg, unsafe_allow_html=True)

# Chat input logic with validations
if user_id:
    if user_data["usage_count"] >= MAX_QUESTIONS:
        remaining = COOLDOWN_SECONDS - (time.time() - user_data["first_use_timestamp"])
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        seconds = int(remaining % 60)
        st.warning(f"❌ Limit reached ({MAX_QUESTIONS} questions). Try again in {hours}h {minutes}m {seconds}s.")
    else:
        if 'settings_saved' not in st.session_state:
            st.session_state.settings_saved = False

        if save_button:
            st.session_state.settings_saved = True

        if st.session_state.settings_saved:
            if user_input := st.chat_input("Ask anything"):
                st.session_state.chat_history.append(("user", user_input))
                with st.chat_message("user"):
                    st.markdown(user_input, unsafe_allow_html=True)
                try:
                    lang = detect(user_input)
                except:
                    lang = "en"
                with st.chat_message("ai"):
                    placeholder = st.empty()
                    full_reply = ""
                    for chunk in stream_answer(user_input, country, sector):
                        full_reply += chunk
                        placeholder.markdown(full_reply + "▌")
                        time.sleep(0.02)
                    placeholder.markdown(full_reply)
                st.session_state.chat_history.append(("ai", full_reply))
                update_usage(user_id)
        else:
            st.warning("⚙️ Please select your Country and Sector in Settings and click 'Save' to start chatting.")
else:
    st.stop()
