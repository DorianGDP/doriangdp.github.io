import faiss
import numpy as np
import json
from openai import OpenAI
import os
import time
from supabase import create_client

class ChatBot:
    # Questions de qualification pour différentes étapes
    QUALIFICATION_QUESTIONS = {
        'name': [
            "Je vois que vous avez des questions intéressantes sur la gestion de patrimoine. Pour mieux suivre notre échange, puis-je connaître votre nom ?",
            "Pour personnaliser notre conversation et garder une trace de nos conseils, comment puis-je vous appeler ?"
        ],
        'contact': [
            "Je peux vous envoyer dès maintenant un premier diagnostic gratuit de votre situation. Sur quelle adresse email puis-je vous l'envoyer ?",
            "Pour que vous puissiez retrouver nos échanges et mes recommandations initiales, quelle est votre adresse email ?"
        ],
        'profession': [
            "Votre situation professionnelle va beaucoup influencer les stratégies possibles. Quelle est votre activité actuelle ?",
            "Pour identifier les meilleures opportunités fiscales, quelle est votre profession ?"
        ],
        'patrimoine': [
            "Pour vous orienter vers les solutions les plus adaptées, dans quelle fourchette se situe votre patrimoine global ?",
            "Afin de vous conseiller les meilleurs investissements, quel est approximativement votre niveau de patrimoine ?"
        ]
    }
    SYSTEM_PROMPT = """Tu es Emma, l'assistante virtuelle de gestiondepatrimoine.com.
    
    OBJECTIF PRINCIPAL : 
    Collecter rapidement le nom et l'email du visiteur, quelle que soit sa question initiale.
    
    STRATÉGIE DE RÉPONSE :
    1. TOUJOURS apporter une micro-réponse à la question du visiteur (1-2 phrases max)
    2. PUIS enchaîner naturellement sur la collecte du nom ou email
    
    SÉQUENCE OBLIGATOIRE :
    1. Si pas de nom → Demander le nom en priorité absolue
    2. Si nom mais pas d'email → Proposer l'envoi d'une analyse gratuite par email
    3. Une fois nom + email obtenus → Passer aux autres informations
    
    FORMULATION TYPE :
    - Si première interaction : "Votre question sur [sujet] est intéressante. Pour pouvoir vous répondre de manière personnalisée, puis-je d'abord connaître votre nom ?"
    - Après avoir le nom : "Merci [nom]. Je peux vous envoyer une première analyse gratuite de votre situation. Sur quelle adresse email puis-je vous l'envoyer ?"
    
    IMPORTANT :
    - NE JAMAIS donner de réponse détaillée avant d'avoir nom + email
    - TOUJOURS justifier la demande d'email par l'envoi d'une analyse gratuite
    - Insister sur : gratuit, sans engagement, personnalisé"""

    def __init__(self, api_key):
        """Initialise le chatbot avec la base de données d'embeddings"""
        self.client = OpenAI(api_key=api_key)
        
        # Configuration Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)

        # Initialisation des données des leads
        self.lead_data = {}
        
        # Configuration des chemins
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.index = faiss.read_index(os.path.join(current_dir, 'embeddings_db', 'faiss_index.idx'))
        
        with open(os.path.join(current_dir, 'embeddings_db', 'metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        # Initialiser l'historique des conversations
        self.conversations = {}

    def get_next_question(self, lead_info):
        """Détermine la prochaine question selon la nouvelle priorité"""
        # Vérifier d'abord nom et email
        if not lead_info.get('name'):
            return np.random.choice(self.QUALIFICATION_QUESTIONS['name'])
        if not lead_info.get('contact'):
            return np.random.choice(self.QUALIFICATION_QUESTIONS['contact'])
            
        # Ensuite les autres informations
        for field in ['profession', 'patrimoine']:
            if not lead_info.get(field):
                return np.random.choice(self.QUALIFICATION_QUESTIONS[field])
        return None

    def get_query_embedding(self, question):
        """Crée l'embedding pour la question"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        return np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)

    def recherche_documents_pertinents(self, question, k=3):
        """Recherche les documents les plus pertinents"""
        question_embedding = self.get_query_embedding(question)
        D, I = self.index.search(question_embedding, k)
        return [self.metadata[idx] for idx in I[0]]

    def update_lead_data(self, conversation_id, lead_data):
        """Mise à jour des données du lead dans Supabase"""
        try:
            existing_lead = self.supabase.table('conversations')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()

            if not existing_lead.data:
                self.supabase.table('conversations').insert({
                    'conversation_id': conversation_id,
                    **lead_data
                }).execute()
            else:
                self.supabase.table('conversations')\
                    .update(lead_data)\
                    .eq('conversation_id', conversation_id)\
                    .execute()
            return True
        except Exception as e:
            print(f"Erreur Supabase: {str(e)}")
            return False
            
    def track_lead_info(self, conversation_id, new_info, interaction=None):
        """Gestion simplifiée des informations"""
        data = self.supabase.table('conversations')\
            .select('*')\
            .eq('conversation_id', conversation_id)\
            .execute()
    
        if data.data:
            record = data.data[0]
            lead_data = record.get('lead_data', {})
            history = record.get('conversation_history', [])
        else:
            lead_data = {}
            history = []
    
        # Mettre à jour les infos
        if new_info:
            lead_data.update(new_info)
    
        # Ajouter l'interaction à l'historique
        if interaction:
            history.append(interaction)
    
        # Sauvegarder
        if data.data:
            self.supabase.table('conversations').update({
                'lead_data': lead_data,
                'conversation_history': history
            }).eq('conversation_id', conversation_id).execute()
        else:
            self.supabase.table('conversations').insert({
                'conversation_id': conversation_id,
                'lead_data': lead_data,
                'conversation_history': history
            }).execute()
    
        return lead_data, history

    def extract_lead_info(self, text):
        """Extraire les informations du texte avec GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Corrigé de gpt-4o
                messages=[{
                    "role": "system",
                    "content": """Tu es un expert en extraction d'informations.
                    Analyse le texte et retourne UNIQUEMENT un objet JSON avec les informations trouvées.
                    - name: prénom/nom mentionnés
                    - profession: métier ou situation professionnelle
                    - patrimoine: montants ou fourchettes financières
                    - contact: email ou téléphone
                    - objectifs: buts patrimoniaux explicites
                    
                    IMPORTANT: 
                    - Renvoie null si l'information n'est pas explicitement mentionnée
                    - N'invente aucune information
                    - Ne fais aucune déduction"""
                }, {
                    "role": "user",
                    "content": text
                }],
                temperature=0.2,  # Réduit pour plus de précision
                max_tokens=200
            )
            return json.loads(response.choices[0].message.content)
        except:
            return {}

    def generer_reponse(self, question, conversation_id):
        try:
            # Extraire les infos de la question actuelle
            new_info = self.extract_lead_info(question)
            
            # Récupérer l'état actuel de la conversation
            lead_data, history = self.track_lead_info(conversation_id, new_info)
            
            # Préparer le contexte
            context = f"""
            Informations client actuelles :
            {json.dumps(lead_data, indent=2)}
    
            Historique récent :
            {json.dumps(history[-3:], indent=2) if history else "Aucun"}
            """
            
            # Générer la réponse
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}\nContexte: {context}"}
                ],
                temperature=0.7
            )
    
            reponse = response.choices[0].message.content
            
            # Sauvegarder l'interaction
            self.track_lead_info(conversation_id, None, {
                'question': question,
                'response': reponse
            })
    
            return reponse
    
        except Exception as e:
            return f"Désolé, une erreur s'est produite. Pouvez-vous reformuler votre question ?"

    def repondre_question(self, question, conversation_id=None):
        """Point d'entrée principal du chatbot"""
        if conversation_id is None:
            conversation_id = str(time.time())
        
        # Plus besoin de docs_pertinents ici
        reponse = self.generer_reponse(question, conversation_id)
        
        return {
            'reponse': reponse,
            'conversation_id': conversation_id
        }
