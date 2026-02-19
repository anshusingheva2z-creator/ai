from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from deep_translator import GoogleTranslator
from langdetect import detect
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime

# ---------------- FILE CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_FILENAME = os.path.join(BASE_DIR, "faq.txt")

vectorstore = None

# ----------------- TRANSLATION -----------------

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def translate_from_english(text, target_lang):
    try:
        if target_lang == "en":
            return text
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text
def is_roman_hindi(text):
    hindi_markers = [
        "kya", "hai", "kaise", "kyun", "kya", "nahi",
        "yeh", "isko", "kaun", "kitna", "karna",
        "mera", "aap", "tum", "sakta", "hoga",
        "kar sakta", "chahiye"
    ]

    text_lower = text.lower()
    match_count = sum(1 for word in hindi_markers if word in text_lower)

    return match_count >= 1

# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(" Starting up EVA2Z Assistant...")
    setup_qa_system()
    yield
    print(" Shutting down...")

app = FastAPI(lifespan=lifespan)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

class QuestionRequest(BaseModel):
    question: str

# ----------------- SETUP SYSTEM (TXT VERSION) -----------------

def setup_qa_system():
    global vectorstore
    try:
        if os.path.exists(os.path.join(BASE_DIR, "faiss_index")):
            vectorstore = FAISS.load_local(
                os.path.join(BASE_DIR, "faiss_index"),
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(" FAISS index loaded")
            return

        if not os.path.exists(TXT_FILENAME):
            print(f" faq.txt not found inside app folder")
            return

        print(" Loading faq.txt...")

        with open(TXT_FILENAME, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split Q&A blocks safely
        blocks = re.split(r'\n(?=Q\d+\.)', content)
        chunks = [b.strip() for b in blocks if len(b.strip()) > 20]

        if not chunks:
            chunks = [content]

        vectorstore = FAISS.from_documents(
            [Document(page_content=c) for c in chunks],
            embeddings
        )

        vectorstore.save_local(os.path.join(BASE_DIR, "faiss_index"))

        print(f" FAISS index created with {len(chunks)} chunks")

    except Exception as e:
        print(f" Error: {e}")
        vectorstore = None


# ---------------- ANSWER EXTRACTION ----------------

def find_answer_in_text(question, context):
    ql = question.lower()
    if not any(w in context.lower() for w in ql.split() if len(w) > 3):
        return None

    for pat in [
        r'A\.\s*(.*?)(?=\nQ\d+\.|$)',
        r'Answer:\s*(.*?)(?=\nQ\d+\.|$)',
        r'A:\s*(.*?)(?=\nQ\d+\.|$)'
    ]:
        m = re.search(pat, context, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

    return context


# ---------------- ASK ENDPOINT ----------------

@app.post("/ask")
async def ask_question(req: QuestionRequest):

    if vectorstore is None:
        return {"answer": "System not ready. Check 'faq.txt'."}

    original_question = req.question.strip()

    detected_lang = detect_language(original_question)

# If detected as English but looks like Roman Hindi
    if detected_lang == "en" and is_roman_hindi(original_question):
       detected_lang = "hi"

    translated_question = translate_to_english(original_question)
    q = translated_question.lower()

    # GREETINGS
    greetings = ["hi", "hii", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if q in greetings:
        response = "👋 Thank you for visiting EVA2Z! How can I help you today?"
        return {"answer": translate_from_english(response, detected_lang)}

    # THANKS
    thanks = ["thanks", "thank you", "thank you very much"]
    if q in thanks:
        response = "You're welcome! 😊 Let me know if you need anything else."
        return {"answer": translate_from_english(response, detected_lang)}

    # STATUS
    if q in ["status", "system status"]:
        response = "System ready. Click a question or type your own."
        return {"answer": translate_from_english(response, detected_lang)}

    # SPECIAL COMMANDS
    if q in ["clear chat", "clear history", "reset chat"]:
        return {"answer": "CHAT_CLEAR_REQUEST"}

    if q in ["save chat", "export chat"]:
        return {"answer": "CHAT_SAVE_REQUEST"}

    # GPS INSTALLATION VIDEO
    install_keywords = [
        "how to install gps", "gps installation", "install gps",
        "installation guide", "installation video",
        "install the gps", "gps install",
        "can i install the gps device myself",
        "self install", "install myself"
    ]

    if any(k in q for k in install_keywords):
        response = (
            "Here's our GPS installation guide video: "
            "https://youtu.be/ZamBx94F0-4?si=YZbohc8WTqQ9-Sgj\n\n"
            "(Click the link to watch. For written instructions, please refer to our FAQ.)"
        )
        return {"answer": translate_from_english(response, detected_lang)}

    # NORMAL QA FROM TXT
    try:
        results = vectorstore.similarity_search(translated_question, k=1)

        if not results:
            response = "I don't know."
            return {"answer": translate_from_english(response, detected_lang)}

        best = results[0].page_content.strip()

        ans = find_answer_in_text(translated_question, best)

        if ans and len(ans) > 10:
            clean_answer = re.sub(r'^Q\d+\.\s*', '', ans)
            return {"answer": translate_from_english(clean_answer, detected_lang)}

        return {"answer": translate_from_english(best, detected_lang)}

    except Exception as e:
        print("Error during search:", e)
        response = "I don't know."
        return {"answer": translate_from_english(response, detected_lang)}

# ----------------- HTML UI (UNCHANGED) -----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
<html>
<head>
<title>EVA2Z Assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body { margin:0; font-family:Segoe UI; background:#f4f6f9; }
#evaIcon {
position:fixed; bottom:25px; right:25px; width:65px; height:65px; border-radius:50%;
background:linear-gradient(135deg,#0066ff,#00c6ff); display:flex; align-items:center;
justify-content:center; color:white; font-size:26px; cursor:pointer;
box-shadow:0 8px 20px rgba(0,0,0,0.25); transition:0.3s; z-index:1000;
}
#evaIcon:hover { transform:scale(1.1); }
#chatBox {
position:fixed; bottom:100px; right:25px; width:360px; height:550px;
background:white; border-radius:18px; box-shadow:0 15px 35px rgba(0,0,0,0.2);
display:none; flex-direction:column; overflow:hidden; z-index:1000;
}
#chatHeader {
background:linear-gradient(135deg,#0066ff,#00c6ff); color:white; padding:15px;
font-weight:600; font-size:16px; display:flex; justify-content:space-between; align-items:center;
}
.header-actions { display:flex; gap:8px; }
.header-btn {
background:rgba(255,255,255,0.2); color:white; border:none; padding:6px 10px;
border-radius:6px; cursor:pointer; font-size:12px; display:flex; align-items:center; gap:4px;
transition:all 0.2s;
}
.header-btn:hover { background:rgba(255,255,255,0.3); transform:translateY(-1px); }
#chatMessages {
flex:1; padding:15px; overflow-y:auto; background:#f9fafc;
}
.message {
margin-bottom:12px; padding:10px 14px; border-radius:12px; max-width:75%;
font-size:14px; line-height:1.4; white-space:pre-wrap; word-wrap:break-word;
animation:fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }
.user { background:#0066ff; color:white; margin-left:auto; }
.bot { background:#e9eef6; color:#333; }
/* NEW: style for links inside bot messages */
.bot a { color: #0066ff; text-decoration: underline; word-break: break-all; }
.bot a:hover { text-decoration: none; }
#faqButtons {
padding:12px; border-top:1px solid #eee; border-bottom:1px solid #eee;
background:#f8f9fa;
}
#faqButtons p {
margin:0 0 8px 0; font-size:13px; color:#666; font-weight:600;
}
.faq-grid {
display:flex; flex-wrap:wrap; gap:6px; justify-content:flex-start;
}
.faq-btn {
background:#e6f0ff; color:#0066ff; border:1px solid #b3d1ff;
padding:6px 12px; border-radius:20px; font-size:12px; cursor:pointer;
transition:all 0.2s; white-space:nowrap;
}
.faq-btn:hover {
background:#d1e3ff; transform:translateY(-1px); box-shadow:0 2px 5px rgba(0,102,255,0.2);
}
#chatInput {
display:flex; border-top:1px solid #eee; background:white; padding:10px;
}
#messageInput {
flex:1; padding:10px; border:1px solid #ddd; border-radius:20px;
outline:none; font-size:14px; transition:border 0.2s;
}
#messageInput:focus { border-color:#0066ff; }
#sendButton {
background:#0066ff; color:white; border:none; border-radius:20px;
padding:10px 16px; margin-left:8px; cursor:pointer; font-size:14px;
transition:all 0.2s;
}
#sendButton:hover { background:#004ecc; transform:scale(0.98); }
.status-message { padding:10px; margin:5px; border-radius:5px; font-size:12px; }
.success { background:#e7f7e7; color:#2d662d; }
.error { background:#fde8e8; color:#c53030; }
.info { background:#e3f2fd; color:#1565c0; }
#infoPanel {
position:fixed; top:20px; left:20px; background:white; padding:15px;
border-radius:10px; box-shadow:0 5px 15px rgba(0,0,0,0.1); max-width:300px; z-index:1000;
}
#historyIndicator {
position:absolute; top:-8px; right:-8px; background:#ff4757; color:white;
font-size:10px; width:16px; height:16px; border-radius:50%; display:none;
align-items:center; justify-content:center;
}
.confirmation-modal {
display:none; position:fixed; top:0; left:0; right:0; bottom:0;
background:rgba(0,0,0,0.5); z-index:2000; align-items:center; justify-content:center;
}
.confirmation-content {
background:white; padding:20px; border-radius:12px; max-width:300px; text-align:center;
}
.confirmation-buttons { display:flex; gap:10px; margin-top:15px; justify-content:center; }
.confirm-btn { padding:8px 16px; border:none; border-radius:6px; cursor:pointer; font-weight:500; }
.confirm-btn.yes { background:#ff4757; color:white; }
.confirm-btn.no { background:#f1f2f6; color:#333; }
</style>
</head>
<body>

<div id="infoPanel">
    <h4 style="margin-top:0;">EVA2Z Assistant</h4>
    <p>Loaded: <strong>FA&Q.pdf</strong></p>
    <p>Click a quick question or type your own.</p>
    <div id="systemStatus" class="status-message info">Loading...</div>
</div>

<div id="confirmationModal" class="confirmation-modal">
    <div class="confirmation-content">
        <h4 style="margin-top:0;">Clear Chat History</h4>
        <p>Are you sure you want to clear all messages?</p>
        <div class="confirmation-buttons">
            <button class="confirm-btn yes" onclick="confirmClear()">Yes, Clear</button>
            <button class="confirm-btn no" onclick="cancelClear()">Cancel</button>
        </div>
    </div>
</div>

<div id="evaIcon">🤖<div id="historyIndicator">!</div></div>

<div id="chatBox">
    <div id="chatHeader">
        <span>EVA2Z Assistant</span>
        <div class="header-actions">
            <button class="header-btn" onclick="saveChatHistory()"> Save</button>
            <button class="header-btn" onclick="showClearConfirmation()"> Clear</button>
            <span id="chatStatus">Loading...</span>
        </div>
    </div>
    <div id="chatMessages"></div>
    
    <div id="faqButtons">
        <p>⚡ Quick questions – click to ask:</p>
        <div class="faq-grid">
            <button class="faq-btn" onclick="askQuick('Is this GPS device safe for my vehicle warranty?')">🔹 Warranty safe?</button>
            <button class="faq-btn" onclick="askQuick('Do you provide free installation?')">🔹 Free install?</button>
            <button class="faq-btn" onclick="askQuick('How long does the GPS work if the vehicle battery is disconnected?')">🔹 Battery backup?</button>
            <button class="faq-btn" onclick="askQuick('Will the GPS device be visible after installation?')">🔹 Visible?</button>
            <button class="faq-btn" onclick="askQuick('Can I install the GPS device myself?')">🔹 Self install?</button>
            <button class="faq-btn" onclick="askQuick('How long does the installation process take?')">🔹 Install time?</button>
            <button class="faq-btn" onclick="askQuick('Is drilling required during installation?')">🔹 Drilling?</button>
        </div>
    </div>
    
    <div id="chatInput">
        <input type="text" id="messageInput" placeholder="Type your question here..." autocomplete="off">
        <button id="sendButton" onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
const evaIcon = document.getElementById('evaIcon');
const chatBox = document.getElementById('chatBox');
const chatMessages = document.getElementById('chatMessages');
const chatStatus = document.getElementById('chatStatus');
const systemStatus = document.getElementById('systemStatus');
const messageInput = document.getElementById('messageInput');
const historyIndicator = document.getElementById('historyIndicator');
const confirmationModal = document.getElementById('confirmationModal');

let chatHistory = [];

evaIcon.onclick = () => {
    chatBox.style.display = chatBox.style.display === 'flex' ? 'none' : 'flex';
    if (chatBox.style.display === 'flex' && chatMessages.children.length === 0) {
        showGreeting();
        setTimeout(() => messageInput.focus(), 300);
    }
};

function getGreeting() {
    const h = new Date().getHours();
    if (h < 12) return 'Good Morning ☀️';
    if (h < 17) return 'Good Afternoon 🌤️';
    return 'Good Evening 🌙';
}

function showGreeting() {
    addMessage(getGreeting() + "! I'm EVA2Z Assistant 🤖\\n\\nClick a quick question above or type your own below.", 'bot');
}

// NEW: convert URLs to clickable links
function linkify(text) {
    const urlPattern = /\\b(?:https?:\\/\\/|www\\.)[^\\s]+/gi;
    return text.replace(urlPattern, function(url) {
        let fullUrl = url.startsWith('http') ? url : 'http://' + url;
        return `<a href="${fullUrl}" target="_blank" rel="noopener noreferrer">${url}</a>`;
    });
}

function addMessage(text, type) {
    const msg = document.createElement('div');
    msg.className = 'message ' + type;
    if (type === 'bot') {
        // Render bot messages with clickable links
        msg.innerHTML = linkify(text.replace(/\\n/g, '<br>'));
    } else {
        // User messages – plain text only
        msg.innerText = text;
    }
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    chatHistory.push({ text, type, timestamp: new Date().toISOString() });
    updateHistoryIndicator();
}

async function askQuick(question) {
    if (!question) return;
    addMessage(question, 'user');
    await fetchAnswer(question);
}

async function sendMessage() {
    const question = messageInput.value.trim();
    if (!question) return;
    addMessage(question, 'user');
    messageInput.value = '';
    await fetchAnswer(question);
}

async function fetchAnswer(question) {
    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        const data = await res.json();
        if (data.answer === 'CHAT_CLEAR_REQUEST') showClearConfirmation();
        else if (data.answer === 'CHAT_SAVE_REQUEST') saveChatHistory();
        else addMessage(data.answer, 'bot');
    } catch {
        addMessage('Sorry, there was an error.', 'bot');
    }
}

function showClearConfirmation() {
    if (chatMessages.children.length === 0) { alert('Chat is already empty!'); return; }
    confirmationModal.style.display = 'flex';
}

function confirmClear() {
    while (chatMessages.firstChild) chatMessages.removeChild(chatMessages.firstChild);
    chatHistory = [];
    updateHistoryIndicator();
    showGreeting();
    confirmationModal.style.display = 'none';
}

function cancelClear() { confirmationModal.style.display = 'none'; }

function updateHistoryIndicator() {
    if (chatHistory.length > 0) {
        historyIndicator.style.display = 'flex';
        historyIndicator.textContent = chatHistory.length > 9 ? '9+' : chatHistory.length;
    } else historyIndicator.style.display = 'none';
}

function saveChatHistory() {
    if (chatHistory.length === 0) { alert('No chat history to save!'); return; }
    let txt = "EVA2Z Assistant Chat History\\nGenerated: " + new Date().toLocaleString() + "\\n" + "=".repeat(50) + "\\n\\n";
    chatHistory.forEach((m, i) => {
        const time = new Date(m.timestamp).toLocaleTimeString();
        const role = m.type === 'user' ? 'You' : 'EVA2Z Assistant';
        txt += `${i+1}. [${time}] ${role}:\\n${m.text}\\n${"-".repeat(40)}\\n`;
    });
    const blob = new Blob([txt], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `eva2z_chat_${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
    addMessage(' Chat history saved to downloads.', 'bot');
}

messageInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

window.onload = () => {
    setTimeout(() => {
        fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: 'status' })
        })
        .then(r => r.json())
        .then(d => {
            if (d.answer?.includes('ready')) {
                chatStatus.textContent = ' Ready';
                systemStatus.textContent = ' System ready – FA&Q.pdf loaded';
                systemStatus.className = 'status-message success';
            } else {
                chatStatus.textContent = ' Error';
                systemStatus.textContent = d.answer || 'System unavailable';
                systemStatus.className = 'status-message error';
            }
        })
        .catch(() => {
            chatStatus.textContent = ' Error';
            systemStatus.textContent = 'Connection error';
            systemStatus.className = 'status-message error';
        });
    }, 1500);
};
</script>

</body>
</html>"""