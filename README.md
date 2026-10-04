# 🩺 Medical RAG Chatbot

A chatbot that answers questions using only the content of medical documents, built with Retrieval-Augmented Generation (RAG). Instead of guessing, it first searches the documents for relevant passages, then uses an LLM to write an answer from them, and shows the sources it used.

🔗 **Live demo:** [https://medical-rag-chatbot-at47frmdujyujewgqsgtqp.streamlit.app/]

> ⚠️ The app may take a few seconds to wake up if it has been idle.

## How it works

1. **Load:** PDFs are read page by page.
2. **Split:** Text is cut into small chunks (500 characters, 20 overlap).
3. **Embed:** Each chunk is turned into a vector with the `all-MiniLM-L6-v2` model from Hugging Face.
4. **Store:** The vectors are saved in a Pinecone vector database.
5. **Retrieve:** When you ask a question, the 3 most similar chunks are found.
6. **Generate:** The chunks and your question go to an LLM on Groq, which writes the answer.
7. **Show sources:** The app displays the pages the answer came from.

## Tech stack

| Part | Tool |
|---|---|
| Framework | LangChain |
| Embeddings | Hugging Face (`all-MiniLM-L6-v2`) |
| Vector database | Pinecone |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Frontend | Streamlit |
| Backend notebook | Google Colab |

## Project structure

```
├── app.py            # Streamlit chat app
├── requirements.txt  # Python dependencies
├── notebook.ipynb    # Colab notebook: PDF loading, chunking, indexing
└── README.md
```

## Run it yourself

1. Clone the repo:
```
   git clone https://github.com/[your-username]/medical-rag-chatbot.git
   cd medical-rag-chatbot
```
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. Create a file `.streamlit/secrets.toml` and add your own keys:
```
   GROQ_API_KEY = "your-groq-key"
   PINECONE_API_KEY = "your-pinecone-key"
```
4. Build the index by running the notebook with your own PDFs. It creates a Pinecone index named `medical-chatbot`.
5. Start the app:
```
   streamlit run app.py
```

## Data

The knowledge base is built from [name of the medical book] and [name of your group's PDF].

## Limitations

- Answers are only as good as the source documents.
- If the answer isn't in the documents, the bot is told to say it doesn't know.
- **Educational use only. This is not medical advice.**

## Team

Built by Aryan Yadav as part of KML Society

## Future improvements

- Support uploading PDFs directly in the app
- Add chat memory for follow-up questions
- Show the source file name for each answer
