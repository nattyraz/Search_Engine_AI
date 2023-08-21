import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pinecone

# Set up Pinecone API key and environment
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Define the index name and create it if it doesn't exist
INDEX_NAME = "semantic-search-index"
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(index_name=INDEX_NAME, dimension=768)

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Define the function to encode text
def encode_text(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.numpy().squeeze()

# Define the function to extract documents from a row
def extract_document_from_row(row):
    return " ".join(row.dropna().astype(str).tolist())

# Set up the Streamlit app
st.title("Moteur de recherche sémantique")

# Upload the Excel file
uploaded_file = st.file_uploader("Téléchargez votre fichier Excel", type=["xlsx"])

# Process the uploaded file
if uploaded_file:
    with st.spinner('Traitement\n\n# Process the uploaded file\nif uploaded_file:\n with st.spinner(&#x27;Traitement du fichier...&#x27;):\n # Read the Excel file into a Pandas dataframe\ndf = pd.read_excel(uploaded_file)\n # Extract the documents from the dataframe\ndocuments = df.apply(extract_document_from_row, axis=1).tolist()\n # Convert the documents to embeddings\ndocument_vectors = [encode_text(doc) for doc in documents]\n\n# Display the results\nst.write(&quot;Documents: &quot;)\nfor doc in document_vectors:\n st.write(f"{doc}\n", reset=True)\n\n# Allow the user to search for documents\nsearch_query = st.text_input(&quot;Rechercher une document: &quot;)\nif search_query:\n search_query = search_query.lower()\n filtered_documents = [doc for doc in document_vectors if search_query in doc]\n st.write(&quot;Documents correspondants à la recherche : &quot;)\n for doc in filtered_documents:\n st.write(f"{doc}\n", reset=True)\n\n# Clean up\nst.stop()\n\n# 
    End of the program\n```
