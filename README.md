# End-to-End Working of our Streamlit-based RAG Educational Research Tool
#Educational_research_bot

This script is a Streamlit-based AI-powered research tool called "Nanditha's Bot", designed to extract and analyze educational content from URLs using OpenAI's language models and FAISS-based vector search.




How It Works
------------
User Input URLs:
Users input up to three educational article URLs via the Streamlit sidebar.
The URLs are stored and processed when the "Process URLs" button is clicked.
Data Extraction & Processing:

The script uses UnstructuredURLLoader to extract content from the URLs.
If successful, the content is split into smaller text chunks using RecursiveCharacterTextSplitter for better processing.
Embedding Generation & Vector Storage:

The text chunks are converted into embeddings (numerical representations) using OpenAIEmbeddings.
A FAISS (Facebook AI Similarity Search) index is built and stored as a faiss_store_openai.pkl file for fast retrieval.
Question-Answering System:

Users can enter a query related to the processed content.
The RetrievalQAWithSourcesChain retrieves the most relevant document chunks from FAISS.
OpenAI’s language model (GPT) generates a well-structured answer with cited sources.
Displaying the Results:

The answer is displayed along with sources if available.

<img width="431" alt="image" src="https://github.com/user-attachments/assets/2ef239c3-f56e-4e54-a755-208b8e42643c" />


End-to-End Flow:
----------------
User inputs URLs → Data is extracted → Text is split → Embeddings are created → FAISS index is stored.
User asks a question → FAISS retrieves relevant chunks → GPT answers using the retrieved content.
