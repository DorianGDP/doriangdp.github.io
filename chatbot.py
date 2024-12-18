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
        """Analyse et stocke les informations du lead"""
        # Récupérer ou créer les données du lead
        lead_info = self.lead_data.get(conversation_id, {
            'name': None,
            'profession': None,
            'patrimoine': None,
            'objectifs': None,
            'contact': None,
            'status': 'new'
        })
        
        # Mettre à jour avec les nouvelles informations
        for key in new_info:
            if new_info[key] and not lead_info[key]:  # Ne mettre à jour que si l'information est nouvelle
                lead_info[key] = new_info[key]
        
        # Mettre à jour le statut si toutes les infos nécessaires sont collectées
        if all([lead_info.get(k) for k in ['name', 'contact', 'patrimoine']]):
            lead_info['status'] = 'qualified'
            
        # Sauvegarder dans la mémoire locale et Supabase
        self.lead_data[conversation_id] = lead_info
        self.update_lead_data(conversation_id, lead_info)
        
        return lead_info

    def extract_lead_info(self, text):
        """Extraire les informations du texte avec GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": """Tu es un expert en analyse de texte. Extrait UNIQUEMENT les informations suivantes si elles sont explicitement mentionnées (renvoie null si non mentionné) :
                    {
                        "name": "nom complet si mentionné",
                        "profession": "profession ou situation professionnelle",
                        "patrimoine": "montant ou fourchette de patrimoine",
                        "contact": "email ou téléphone",
                        "objectifs": "objectifs patrimoniaux mentionnés"
                    }
                    Ne fais pas de suppositions. N'extrait que les informations explicites."""
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
        """Génère une réponse avec GPT-4 en tenant compte de l'historique"""
        try:
            # Extraire et tracker les infos du lead
            new_info = self.extract_lead_info(question)
            lead_info = self.track_lead_info(conversation_id, new_info)
            
            # Obtenir la prochaine question à poser
            next_question = self.get_next_question(lead_info)
            
            conversation_history = self.conversations.get(conversation_id, [])
            
            system_prompt = """Tu es Emma, l'assistante virtuelle experte de gestiondepatrimoine.com. Ta mission PRINCIPALE est de collecter des informations sur les visiteurs tout en les guidant vers nos services.

RÈGLES FONDAMENTALES :
1. NE JAMAIS donner de réponses détaillées directement
2. Toujours reconnaître brièvement l'intérêt de la question (1-2 phrases max)
3. Expliquer que pour une réponse précise et personnalisée, tu as besoin d'en savoir plus
4. Poser UNE question ciblée pour obtenir une information manquante

Si toutes les informations sont collectées (nom, contact, patrimoine):
- Proposer un rendez-vous avec un expert
- Mentionner que c'est gratuit et sans engagement
- Expliquer que c'est le meilleur moyen d'obtenir des réponses précises

Ton objectif est de COLLECTER des informations, pas d'en donner."""

            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            messages.extend(conversation_history)
            
            prompt = f"""Question du visiteur: {question}

Informations collectées:
Nom: {lead_info.get('name', 'Non renseigné')}
Profession: {lead_info.get('profession', 'Non renseigné')}
Patrimoine: {lead_info.get('patrimoine', 'Non renseigné')}
Contact: {lead_info.get('contact', 'Non renseigné')}
Objectifs: {lead_info.get('objectifs', 'Non renseigné')}

Prochaine information à demander: {next_question if next_question else 'Toutes les informations sont collectées'}"""

            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stop=None
            )
            
            contenu = response.choices[0].message.content
            
            # Sauvegarder la conversation
            self.conversations[conversation_id] = conversation_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": contenu}
            ]
            
            return contenu
                
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
