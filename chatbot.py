import faiss
import numpy as np
import json
from openai import OpenAI
import os
import time
from supabase import create_client

class ChatBot:
    def __init__(self, api_key):
        """
        Initialise le chatbot avec la base de données d'embeddings
        """
        self.client = OpenAI(api_key=api_key)

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)
        
        # Obtenir le chemin du répertoire courant
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construire les chemins absolus
        index_path = os.path.join(current_dir, 'embeddings_db', 'faiss_index.idx')
        metadata_path = os.path.join(current_dir, 'embeddings_db', 'metadata.json')
        
        # Charger l'index FAISS
        self.index = faiss.read_index(index_path)
        
        # Charger les metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        # Initialiser l'historique des conversations
        self.conversations = {}

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

    def update_lead_data(self, conversation_id, lead_data):
        """Mise à jour des données du lead dans Supabase"""
        try:
            # Vérifier si le lead existe
            existing_lead = self.supabase.table('leads')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()

            if not existing_lead.data:
                # Créer un nouveau lead
                self.supabase.table('leads').insert({
                    'conversation_id': conversation_id,
                    **lead_data
                }).execute()
            else:
                # Mettre à jour le lead existant
                self.supabase.table('leads')\
                    .update(lead_data)\
                    .eq('conversation_id', conversation_id)\
                    .execute()
                
            return True
        except Exception as e:
            print(f"Erreur lors de la mise à jour Supabase: {str(e)}")
            return False

    def track_lead_info(self, conversation_id, response_text):
        """Analyse et stocke les informations du lead"""
        lead_info = self.lead_data.get(conversation_id, {
            'name': None,
            'profession': None,
            'patrimoine': None,
            'objectifs': None,
            'contact': None,
            'status': 'new'
        })
        
        # Mettre à jour le statut si on a toutes les infos nécessaires
        if all([lead_info.get(k) for k in ['name', 'contact', 'patrimoine']]):
            lead_info['status'] = 'qualified'
            
        # Sauvegarder dans Supabase
        self.update_lead_data(conversation_id, lead_info)
        
        return lead_info

    def extract_lead_info(self, text):
        """Extraire les informations du texte avec GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": """Analyse le texte et extrait les informations suivantes au format JSON:
                    - name: nom complet si mentionné
                    - profession: profession ou situation professionnelle
                    - patrimoine: montant ou fourchette de patrimoine
                    - contact: email ou téléphone
                    - objectifs: objectifs patrimoniaux mentionnés"""
                }, {
                    "role": "user",
                    "content": text
                }],
                temperature=0.3,
                max_tokens=200
            )
            
            extracted_info = json.loads(response.choices[0].message.content)
            return extracted_info
        except:
            return {}
            
    def generer_reponse(self, question, documents_pertinents, conversation_id):
        """
        Génère une réponse avec GPT-4 en tenant compte de l'historique
        """
        try:
            # Extraire les infos de la question
            new_info = self.extract_lead_info(question)
            
            # Mettre à jour les infos du lead
            lead_info = self.track_lead_info(conversation_id, new_info)
            
            context = "\n\n".join([
                f"""Source {idx+1}:
                Titre: {doc['title']}
                Contenu: {doc['content']}
                URL: {doc['url']}""" 
                for idx, doc in enumerate(documents_pertinents)
            ])
            
            # Récupérer l'historique de la conversation
            conversation_history = self.conversations.get(conversation_id, [])
            
            # Construire les messages avec l'historique
            messages = [
                {"role": "system", "content": """Tu es l'assistant virtuel officiel de gestiondepatrimoine.com, expert en gestion de patrimoine. Tu incarnes l'expertise et les valeurs de notre site, qui est une référence en conseil patrimonial.
            
                Ta mission principale est d'accompagner les visiteurs sur notre plateforme gestiondepatrimoine.com en :
                1. Fournissant des réponses précises basées exclusivement sur le contenu de notre site
                2. Incluant systématiquement les URLs de nos pages dans tes réponses pour permettre aux utilisateurs d'approfondir les sujets
                3. Indiquant clairement si une information n'est pas disponible dans notre base de connaissances
                4. Formulant des réponses engageantes qui reflètent notre expertise en gestion de patrimoine
                5. Tenant compte de l'historique de la conversation pour des échanges cohérents et personnalisés
                6. Répondant avec professionnalisme et empathie aux remerciements des utilisateurs
            
                Tu dois toujours te présenter comme faisant partie intégrante de gestiondepatrimoine.com et orienter les utilisateurs vers nos contenus et services. Si un utilisateur pose une question hors sujet ou non liée à la gestion de patrimoine, rappelle-lui poliment que tu es spécialisé dans le conseil patrimonial et recentre la conversation sur ce domaine."""}
            ]
            
            # Ajouter l'historique des messages
            messages.extend(conversation_history)
            
            # Ajouter la question actuelle avec son contexte
            messages.append({"role": "user", "content": f"Question: {question}\n\nContexte:\n{context}"})
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stop=None
            )
            contenu = response.choices[0].message.content
            if len(contenu) >= 900:  # Seuil augmenté significativement
                contenu = contenu.rsplit('.', 1)[0] + ".\n\nNote : La réponse est longue, n'hésitez pas à me poser des questions spécifiques pour plus de détails."
            # Sauvegarder la question et la réponse dans l'historique
            self.conversations[conversation_id] = conversation_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": contenu}
            ]
            
            return contenu
                
        except Exception as e:
            return f"Erreur lors de la génération de la réponse: {str(e)}"

    def repondre_question(self, question, conversation_id=None):
        """
        Point d'entrée principal du chatbot
        """
        if conversation_id is None:
            conversation_id = str(time.time())  # Générer un ID si non fourni
            
        docs_pertinents = self.recherche_documents_pertinents(question)
        reponse = self.generer_reponse(question, docs_pertinents, conversation_id)
        
        return {
            'reponse': reponse,
            'documents': docs_pertinents,
            'conversation_id': conversation_id
        }
