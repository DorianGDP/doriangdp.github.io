# backend/chatbot.py
import faiss
import numpy as np
import json
from openai import OpenAI
import os

class ChatBot:
    def __init__(self, api_key):
        """
        Initialise le chatbot avec la base de données d'embeddings
        """
        self.client = OpenAI(api_key=api_key)
        
        # Charger l'index FAISS
        self.index = faiss.read_index('embeddings_db/faiss_index.idx')
        
        # Charger les metadata
        with open('embeddings_db/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def get_query_embedding(self, question):
        """
        Crée l'embedding pour la question
        """
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        return np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)

    def recherche_documents_pertinents(self, question, k=3):
        """
        Recherche les documents les plus pertinents
        """
        # Obtenir l'embedding de la question
        question_embedding = self.get_query_embedding(question)
        
        # Rechercher les plus proches voisins
        D, I = self.index.search(question_embedding, k)
        
        # Récupérer les documents pertinents
        docs_pertinents = []
        for idx in I[0]:
            docs_pertinents.append(self.metadata[idx])
            
        return docs_pertinents

    def generer_reponse(self, question, documents_pertinents):
        """
        Génère une réponse avec GPT-4
        """
        # Préparer le contexte
        context = "\n\n".join([
            f"Titre: {doc['title']}\nContenu: {doc['content']}\nURL: {doc['url']}" 
            for doc in documents_pertinents
        ])
        
        system_prompt = """Tu es un assistant virtuel expert chargé d'aider les utilisateurs à naviguer sur notre site web. 
        Tu dois:
        1. Fournir des réponses précises basées uniquement sur le contenu fourni
        2. Inclure systématiquement les URLs pertinentes dans ta réponse
        3. Indiquer clairement si tu ne trouves pas l'information dans le contexte
        4. Formuler des réponses naturelles et engageantes"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\n\nContexte:\n{context}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Erreur lors de la génération de la réponse: {str(e)}"

    def repondre_question(self, question):
        """
        Point d'entrée principal du chatbot
        """
        # Rechercher les documents pertinents
        docs_pertinents = self.recherche_documents_pertinents(question)
        
        # Générer la réponse
        reponse = self.generer_reponse(question, docs_pertinents)
        
        return {
            'reponse': reponse,
            'documents': docs_pertinents
        }
