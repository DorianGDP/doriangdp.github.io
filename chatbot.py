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
            "Pour personnaliser mes recommandations, comment souhaitez-vous que je m'adresse à vous ?",
            "Afin d'adapter au mieux mes conseils, puis-je avoir votre nom ?"
        ],
        'profession': [
            "Votre situation professionnelle va beaucoup influencer les stratégies possibles. Quelle est votre activité actuelle ?",
            "Pour identifier les meilleures opportunités fiscales, quelle est votre profession ?"
        ],
        'patrimoine': [
            "Pour vous orienter vers les solutions les plus adaptées, dans quelle fourchette se situe votre patrimoine global ?",
            "Afin de vous conseiller les meilleurs investissements, quel est approximativement votre niveau de patrimoine ?"
        ],
        'contact': [
            "Je peux vous faire parvenir une analyse détaillée par email. Quelle est la meilleure adresse pour vous joindre ?",
            "Pour vous envoyer une étude personnalisée de votre situation, quel serait le meilleur moyen de vous contacter ?"
        ]
    }

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
        """Détermine la prochaine question à poser en fonction des informations manquantes"""
        for field in ['name', 'profession', 'patrimoine', 'contact']:
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
            existing_lead = self.supabase.table('leads')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()

            if not existing_lead.data:
                self.supabase.table('leads').insert({
                    'conversation_id': conversation_id,
                    **lead_data
                }).execute()
            else:
                self.supabase.table('leads')\
                    .update(lead_data)\
                    .eq('conversation_id', conversation_id)\
                    .execute()
            return True
        except Exception as e:
            print(f"Erreur Supabase: {str(e)}")
            return False
            
    def track_lead_info(self, conversation_id, new_info):
        """Analyse et stocke les informations du lead de manière persistante"""
        
        # Récupérer ou initialiser les infos du lead depuis Supabase
        existing_lead = self.supabase.table('leads')\
            .select('*')\
            .eq('conversation_id', conversation_id)\
            .execute()
        
        if existing_lead.data:
            lead_info = existing_lead.data[0]
        else:
            lead_info = {
                'name': None,
                'profession': None,
                'patrimoine': None,
                'objectifs': None,
                'contact': None,
                'status': 'new',
                'conversation_history': []
            }
    
        # Mettre à jour avec les nouvelles informations
        for key in new_info:
            if new_info[key] and not lead_info.get(key):
                lead_info[key] = new_info[key]
        
        # Vérifier et mettre à jour le statut
        if all([lead_info.get(k) for k in ['name', 'contact', 'patrimoine']]):
            lead_info['status'] = 'qualified'
        
        # Gérer l'historique des conversations
        if 'conversation_history' not in lead_info:
            lead_info['conversation_history'] = []
        
        # Ajouter seulement si new_info contient une question/réponse
        if new_info.get('question') or new_info.get('response'):
            lead_info['conversation_history'].append({
                'timestamp': time.time(),
                'question': new_info.get('question', ''),
                'response': new_info.get('response', '')
            })
        
        # Sauvegarder dans Supabase et mémoire locale
        self.update_lead_data(conversation_id, lead_info)
        self.lead_data[conversation_id] = lead_info
        
        return lead_info

    def extract_lead_info(self, text):
        """Extraire les informations du texte avec GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": """Analyse le texte et extrait les informations suivantes au format JSON strict :
                    {
                        "name": null,
                        "profession": null,
                        "patrimoine": null,
                        "contact": null,
                        "objectifs": null
                    }
                    Règles :
                    - Renvoie EXACTEMENT ce format JSON
                    - Remplace 'null' par la valeur si trouvée
                    - Pour le patrimoine : extrait les montants/fourchettes
                    - Pour le contact : extrait email/téléphone
                    - Ne fait AUCUNE supposition
                    - N'extrait que les informations EXPLICITEMENT mentionnées"""
                }, {
                    "role": "user",
                    "content": text
                }],
                temperature=0.3,
                max_tokens=200
            )
            return json.loads(response.choices[0].message.content)
        except:
            return {}

    def generer_reponse(self, question, documents_pertinents, conversation_id):
        try:
            # Extraire et tracker les infos du lead
            new_info = self.extract_lead_info(question)
            lead_info = self.track_lead_info(conversation_id, new_info)
            
            # Vérifier les informations manquantes
            missing_info = []
            for field in ['name', 'profession', 'patrimoine', 'contact']:
                if not lead_info.get(field):
                    missing_info.append(field)
    
            # Obtenir la prochaine question spécifique
            next_question = None
            if missing_info:
                field_to_ask = missing_info[0]  # Prendre la première information manquante
                next_question = np.random.choice(self.QUALIFICATION_QUESTIONS[field_to_ask])
    
            # Construire un contexte plus riche
            conversation_context = f"""
            Informations actuelles sur le visiteur :
            - Nom : {lead_info.get('name', 'Non renseigné')}
            - Profession : {lead_info.get('profession', 'Non renseigné')}
            - Patrimoine : {lead_info.get('patrimoine', 'Non renseigné')}
            - Contact : {lead_info.get('contact', 'Non renseigné')}
            - Objectifs : {lead_info.get('objectifs', 'Non renseigné')}
    
            Informations manquantes : {', '.join(missing_info) if missing_info else 'Aucune'}
            Prochaine question à poser : {next_question if next_question else 'Passage à la proposition de rendez-vous'}
            """
    
            # Récupérer l'historique complet
            conversation_history = self.conversations.get(conversation_id, [])
            
            messages = [
                {"role": "system", "content": """Tu es Emma, l'assistante virtuelle experte de gestiondepatrimoine.com.
                
                OBJECTIF PRINCIPAL : Collecter progressivement toutes les informations sur le visiteur.
                
                RÈGLES DE CONVERSATION :
                1. Être naturelle et empathique
                2. Ne donner que des réponses très brèves aux questions (1-2 phrases)
                3. Toujours enchaîner avec une question pour collecter une information manquante
                4. Si toutes les informations sont collectées, proposer un rendez-vous gratuit
                5. Maintenir le fil de la conversation en faisant référence aux échanges précédents
                
                STRATÉGIE DE COLLECTE :
                - Nom → Pour personnalisation
                - Profession → Pour solutions adaptées
                - Patrimoine → Pour conseils pertinents
                - Contact → Pour suivi personnalisé
                """}
            ]
            
            # Ajouter l'historique complet
            messages.extend(conversation_history)
            
            # Ajouter le contexte actuel
            messages.append({"role": "user", "content": f"Question du visiteur: {question}\n\nContexte:\n{conversation_context}"})
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
    
            reponse = response.choices[0].message.content
    
            # Sauvegarder l'historique
            self.conversations[conversation_id] = conversation_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reponse}
            ]
    
            # Si toutes les informations sont collectées, ajouter une proposition de rendez-vous
            if not missing_info:
                reponse += "\n\nJ'ai maintenant une bonne compréhension de votre situation. Le mieux serait d'organiser un échange gratuit avec l'un de nos experts qui pourra vous apporter des réponses détaillées et personnalisées. Souhaitez-vous que je planifie ce rendez-vous ?"
    
            return reponse
    
        except Exception as e:
            return f"Erreur lors de la génération de la réponse: {str(e)}"

    def repondre_question(self, question, conversation_id=None):
        """Point d'entrée principal du chatbot"""
        if conversation_id is None:
            conversation_id = str(time.time())
            
        docs_pertinents = self.recherche_documents_pertinents(question)
        reponse = self.generer_reponse(question, docs_pertinents, conversation_id)
        
        return {
            'reponse': reponse,
            'conversation_id': conversation_id
        }
