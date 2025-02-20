# import os
# import streamlit as st
# import pickle
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env (especially openai api key)

# st.title("NanduBot: Education Research Tool 📈")
# st.sidebar.title("Education Article URLs")

# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")
# file_path = "faiss_store_openai.pkl"

# main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)

# if process_url_clicked:
#     # load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load()
#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitter...Started...✅✅✅")
#     docs = text_splitter.split_documents(data)
#     # create embeddings and save it to FAISS index
#     embeddings = OpenAIEmbeddings()
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore_openai, f)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)
#             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#             st.header("Answer")
#             st.write(result["answer"])

#             # Display sources, if available
#             sources = result.get("sources", "")
#             if sources:
#                 st.subheader("Sources:")
#                 sources_list = sources.split("\n")  # Split the sources by newline
#                 for source in sources_list:
#                     st.write(source)

import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables (especially OpenAI API key)
load_dotenv()

st.title("Nanditha's Bot:Educational Research Tool📈")
st.sidebar.title("Educational Article URLs")

# Capture URLs from user input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url.strip())  # Strip spaces

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    urls = [url for url in urls if url]  # Remove empty URLs
    if not urls:
        st.error("Please provide at least one valid URL.")
    else:
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()

            # Check if data is successfully loaded
            if not data:
                st.error("Failed to load data. Check if the URLs are accessible.")
            else:
                # Split data into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                main_placeholder.text("Text Splitter...Started...✅✅✅")
                docs = text_splitter.split_documents(data)

                if not docs:
                    st.error("No documents were extracted. Check if the URLs contain text data.")
                else:
                    # Create embeddings and save to FAISS index
                    embeddings = OpenAIEmbeddings()
                    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                    vectorstore_openai = FAISS.from_documents(docs, embeddings)

                    # Save FAISS index
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore_openai, f)

                    st.success("Processing completed! You can now ask questions.")

        except Exception as e:
            st.error(f"Error: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            try:
                result = chain({"question": query}, return_only_outputs=True)
                st.header("Answer")
                st.write(result.get("answer", "No answer found."))

                # Display sources, if available
                sources = result.get("sources", "").strip()
                if sources:
                    st.subheader("Sources:")
                    for source in sources.split("\n"):
                        st.write(source)
                else:
                    st.write("No sources available.")
            except Exception as e:
                st.error(f"Error while retrieving answer: {e}")
    else:
        st.error("No FAISS index found. Please process URLs first.")
