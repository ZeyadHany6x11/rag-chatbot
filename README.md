# 🤖 RAG-Powered Customer Support Chatbot

This is a minimal yet powerful **RAG (Retrieval-Augmented Generation)** chatbot built using **Streamlit**, **FAISS**, **sentence-transformers**, and **OpenAI GPT** models. It helps answer customer support questions by grounding responses in actual company documents.

### 🔍 Example Use Cases
- Customer support FAQs
- Product troubleshooting
- Internal helpdesk chatbot
- Retrieval-based GenAI demos
---

## 🚀 Live Demo

🟢 [Launch the App](https://rag-chatbot-nrkd3tb8gx3vjceqtxa2xb.streamlit.app)  

---

## 🧠 How It Works

1. Loads and cleans a customer support dataset (CSV)
2. Chunks the instruction–response pairs into small text segments
3. Embeds the chunks using `all-MiniLM-L6-v2` (via `sentence-transformers`)
4. Stores embeddings in a FAISS index for fast similarity search
5. Accepts a user query and retrieves top matching chunks
6. Sends a prompt with those chunks to OpenAI’s `gpt-4o-mini` model
7. Returns a concise, grounded answer with source citations

---

## 📂 Project Structure

rag-chatbot/
├── app.py                          # Main Streamlit app
├── Customer_Support_Training_Dataset.csv  # CSV data
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

---

## 📦 Installation & Run (Locally)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
echo "your-openai-api-key" > OPENAI_API_KEY.txt

# 5. Run the app
streamlit run app.py


⸻

🔐 Handling Secrets (for Streamlit Cloud)

Create a secrets.toml entry in Streamlit Cloud like:

OPENAI_API_KEY = "your-api-key"

Update app.py to use:

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


⸻

📚 Dataset

This app uses a cleaned and chunked version of a customer support Q&A dataset formatted as:

instruction	response
How to cancel my order?	You can cancel by logging in.
Can I change my address?	Yes, go to profile settings.


⸻

🔧 Requirements

streamlit
pandas
faiss-cpu
sentence-transformers
openai


⸻

🧪 Example Questions

Try asking:
	•	“How do I cancel an order?”
	•	“Can I change my delivery address?”
	•	“What’s your return policy?”

⸻

✨ Features

✅ RAG pipeline using FAISS
✅ Streamlit interface with memory
✅ Inline source chunk display
✅ OpenAI GPT-4o-mini integration
✅ Lightweight & fast (under 100MB)


