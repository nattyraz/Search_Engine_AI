import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pinecone

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENVIRONMENT"]

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

INDEX_NAME = "semantic-search-index"
index = pinecone.Index(INDEX_NAME)

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(index_name=INDEX_NAME, dimension=768)

with st.spinner('Chargement du modèle et du tokenizer...'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

def encode_text(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.numpy().squeeze()

def extract_document_from_row(row):
    return " ".join(row.dropna().astype(str).tolist())

st.title("Moteur de recherche sémantique")

uploaded_file = st.file_uploader("Téléchargez votre fichier Excel", type=["xlsx"])

if uploaded_file:
    with st.spinner('Traitement du fichier...'):
        df = pd.read_excel(uploaded_file)
        documents = df.apply(extract_document_from_row, axis=1).tolist()
        document_vectors = [{'id': str(i), 'values': encode_text(doc).tolist()} for i, doc in enumerate(documents)]
        
        upsert_response = index.upsert(vectors=document_vectors)
    st.success("Fichier téléchargé et indexé avec succès!")

    query = st.text_input("Entrez votre requête de recherche :")
    if query:
        query_vector = encode_text(query)
        results = index.query(queries=[query_vector.tolist()], top_k=5)
        st.write("Résultats :", results)
