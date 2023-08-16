import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pinecone
import os

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENVIRONMENT"]

# Initialisation de Pinecone juste après avoir récupéré la clé API
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)


INDEX_NAME = "semantic-search-index"
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(index_name=INDEX_NAME, dimension=768)


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

def encode_text(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.numpy().squeeze()

def extract_document_from_row(row):
    return " ".join(row.dropna().astype(str).tolist())

# Interface Streamlit
st.title("Moteur de recherche sémantique")

uploaded_file = st.file_uploader("Téléchargez votre fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    documents = df.apply(extract_document_from_row, axis=1).tolist()
    document_vectors = {str(i): encode_text(doc) for i, doc in enumerate(documents)}
    
    # Insérer les vecteurs dans Pinecone
    pinecone.upsert(vectors=document_vectors, index_name=INDEX_NAME)
    st.success("Fichier téléchargé et indexé avec succès!")

    query = st.text_input("Entrez votre requête de recherche :")
    if query:
        query_vector = encode_text(query)
        # Utiliser Pinecone pour la recherche
        results = pinecone.query(queries=[query_vector], top_k=5, index_name=INDEX_NAME)
        st.write("Résultats :", results)
