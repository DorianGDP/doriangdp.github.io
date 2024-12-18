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

    QCM_OPTIONS = {
        'objectifs': [
            "Obtenir des revenus complémentaires",
            "Investir en immobilier",
            "Développer mon patrimoine",
            "Réduire mes impôts",
            "Préparer ma retraite",
            "Transmettre mon patrimoine",
            "Placer ma trésorerie excédentaire",
            "Autres"
        ],
        'patrimoine_financier': [
            "Moins de 20 000€",
            "Entre 20 000€ et 50 000€",
            "Entre 50 000€ et 100 000€",
            "Entre 100 000€ et 250 000€",
            "Entre 250 000€ et 500 000€",
            "Entre 500 000€ et 1 000 000€",
            "Entre 1 000 000€ et 2 500 000€",
            "Plus de 2 500 000€"
        ],
        'revenu_annuel': [
            "Moins de 30 000€",
            "Entre 30 000€ et 60 000€",
            "Entre 60 000€ et 90 000€",
            "Entre 90 000€ et 120 000€",
            "Entre 120 000€ et 150 000€",
            "Entre 150 000€ et 200 000€",
            "Entre 200 000€ et 250 000€",
            "Plus de 250 000€"
        ]
    }

    SYSTEM_PROMPT = """Tu es Emma, l'assistante virtuelle experte de gestiondepatrimoine.com.

    OBJECTIF PRINCIPAL :
    1. D'abord obtenir nom et email (pour envoyer un premier avis gratuit personnalisé)
    2. Ensuite guider à travers les 3 questions QCM
    3. Donner une préconisation personnalisée basée sur les réponses

    ÉTAPES DE CONVERSATION :
    1. Introduction et collecte contact :
       - Demander nom/prénom naturellement
       - Demander email pour envoyer l'analyse gratuite
       
    2. QCM (une fois email obtenu) :
       - Présenter chaque question avec ses options
       - Valider les réponses
       
    3. Conclusion :
       - Donner une préconisation personnalisée
       - Proposer des liens pertinents du site
       - Proposer un rappel téléphonique rapide

    RÈGLES :
    - Être naturel et empathique
    - Expliquer pourquoi on demande les informations
    - Valoriser l'aspect gratuit de l'analyse
    - Ne pas sauter d'étapes
    - Une question à la fois"""

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

    def extract_qcm_info(self, text):
        """Extrait les réponses QCM du texte"""
        qcm_data = {}
        for qcm_type in ['objectifs', 'patrimoine_financier', 'revenu_annuel']:
            matched = self.process_qcm_response(text, qcm_type)
            if matched:
                qcm_data[qcm_type] = matched
        return qcm_data

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
    def update_conversation_data(self, conversation_id, data):
        try:
            existing = self.supabase.table('conversations')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            if existing.data:
                self.supabase.table('conversations')\
                    .update(data)\
                    .eq('conversation_id', conversation_id)\
                    .execute()
            else:
                self.supabase.table('conversations').insert({
                    'conversation_id': conversation_id,
                    **data
                }).execute()
            return True
        except Exception as e:
            print(f"Erreur mise à jour conversation: {str(e)}")
            return False
    def track_lead_info(self, conversation_id, new_info, interaction=None):
        data = self.supabase.table('conversations')\
            .select('*')\
            .eq('conversation_id', conversation_id)\
            .execute()
    
        if data.data:
            record = data.data[0]
            lead_data = record.get('lead_data', {})
            qcm_data = record.get('qcm_responses', {})
            history = record.get('conversation_history', [])
        else:
            lead_data = {}
            qcm_data = {"objectifs": None, "patrimoine_financier": None, "revenu_annuel": None}
            history = []
    
        # Mettre à jour les infos et QCM
        if new_info:
            lead_data.update(new_info.get('lead_data', {}))
            qcm_data.update(new_info.get('qcm_data', {}))
    
        # Sauvegarder
        updated_data = {
            'lead_data': lead_data,
            'qcm_responses': qcm_data,
            'conversation_history': history
        }
        
        if interaction:
            updated_data['conversation_history'].append(interaction)
    
        self.update_conversation_data(conversation_id, updated_data)
        
        return lead_data, qcm_data

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
            # Récupérer l'état actuel
            lead_data, qcm_data = self.get_conversation_state(conversation_id)
            
            # Déterminer l'étape actuelle
            etape = self.determiner_etape(lead_data, qcm_data)
            
            context = {
                'lead_data': lead_data,
                'qcm_data': qcm_data,
                'etape': etape,
                'options_qcm': self.QCM_OPTIONS
            }
            
            # Générer la réponse appropriée
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.format_context(question, context)}
                ],
                temperature=0.7
            )
    
            reponse = response.choices[0].message.content
            
            # Ajouter après l'extraction des infos lead
            qcm_info = self.extract_qcm_info(question)
            if qcm_info:
                new_info['qcm_data'] = qcm_info
                
            # Si c'est la dernière étape et tout est complété, ajouter des recommandations
            if etape == 'conclusion' and self.is_all_data_complete(lead_data, qcm_data):
                reponse += self.generer_preconisations(lead_data, qcm_data)
    
            return reponse
    
        except Exception as e:
            return f"Désolé, une erreur s'est produite. Pouvez-vous reformuler?"

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
    def get_conversation_state(self, conversation_id):
        """Récupère l'état actuel de la conversation"""
        data = self.supabase.table('conversations')\
            .select('lead_data, qcm_responses')\
            .eq('conversation_id', conversation_id)\
            .execute()
        
        if data.data:
            return data.data[0].get('lead_data', {}), data.data[0].get('qcm_responses', {})
        return {}, {
            "objectifs": None,
            "patrimoine_financier": None,
            "revenu_annuel": None
        }
    def process_qcm_response(self, text, qcm_type):
        """Analyse la réponse à une question QCM"""
        try:
            options = self.QCM_OPTIONS[qcm_type]
            # Normaliser le texte pour la comparaison
            normalized_text = text.lower().strip()
            
            # Chercher les correspondances
            matched_options = [
                opt for opt in options
                if opt.lower() in normalized_text
            ]
            
            return matched_options if matched_options else None
            
        except Exception as e:
            print(f"Erreur traitement QCM: {str(e)}")
            return None    
    def determiner_etape(self, lead_data, qcm_data):
        """Détermine l'étape actuelle de la conversation"""
        # Vérifier si nous avons les informations de contact
        if not lead_data.get('name') or not lead_data.get('email'):
            return 'contact_initial'
        
        # Vérifier les réponses QCM
        if qcm_data.get('objectifs') is None:
            return 'qcm_objectifs'
        if qcm_data.get('patrimoine_financier') is None:
            return 'qcm_patrimoine'
        if qcm_data.get('revenu_annuel') is None:
            return 'qcm_revenu'
        
        # Si tout est complété
        return 'conclusion'
    
    def format_context(self, question, context):
        """Formate le contexte pour GPT"""
        etape = context['etape']
        lead_data = context['lead_data']
        qcm_data = context['qcm_data']
        
        formatted_context = f"""
        Question du client : {question}
    
        État actuel :
        - Nom : {lead_data.get('name', 'Non renseigné')}
        - Email : {lead_data.get('email', 'Non renseigné')}
        - Téléphone : {lead_data.get('phone', 'Non renseigné')}
    
        Réponses QCM :
        - Objectifs : {qcm_data.get('objectifs', 'Non renseigné')}
        - Patrimoine : {qcm_data.get('patrimoine_financier', 'Non renseigné')}
        - Revenu : {qcm_data.get('revenu_annuel', 'Non renseigné')}
    
        Étape actuelle : {etape}
    
        {self.get_next_step_instructions(etape, context)}
        """
        return formatted_context
    
    def get_next_step_instructions(self, etape, context):
        """Donne les instructions spécifiques pour l'étape en cours"""
        if etape == 'contact_initial':
            return """
            INSTRUCTIONS :
            - Si pas de nom : demander le nom pour personnaliser l'échange
            - Si pas d'email : expliquer qu'un email permettra d'envoyer une première analyse gratuite
            - Être naturel et empathique
            """
        
        elif etape == 'qcm_objectifs':
            options = '\n'.join([f"- {opt}" for opt in self.QCM_OPTIONS['objectifs']])
            return f"""
            INSTRUCTIONS :
            - Introduire naturellement la première question QCM
            - Présenter les options disponibles :
            {options}
            - Inviter à choisir une ou plusieurs options
            """
        
        elif etape in ['qcm_patrimoine', 'qcm_revenu']:
            options = '\n'.join([f"- {opt}" for opt in self.QCM_OPTIONS[etape.split('_')[1]]])
            return f"""
            INSTRUCTIONS :
            - Enchaîner naturellement avec la question suivante
            - Présenter les tranches :
            {options}
            """
        
        else:  # conclusion
            return """
            INSTRUCTIONS :
            - Remercier pour les informations partagées
            - Donner une préconisation personnalisée
            - Proposer des liens pertinents du site
            - Proposer un rappel téléphonique si souhaité
            """
    
    def is_all_data_complete(self, lead_data, qcm_data):
        """Vérifie si toutes les informations nécessaires sont collectées"""
        required_lead = ['name', 'email']
        required_qcm = ['objectifs', 'patrimoine_financier', 'revenu_annuel']
        
        return all(lead_data.get(field) for field in required_lead) and \
               all(qcm_data.get(field) for field in required_qcm)
    
    def generer_preconisations(self, lead_data, qcm_data):
        """Génère des préconisations personnalisées basées sur les réponses"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": """Tu es un expert en gestion de patrimoine.
                    Analyse les informations du client et génère :
                    1. Une préconisation personnalisée succincte
                    2. 2-3 liens pertinents vers des articles du site
                    3. Une invitation à être rappelé pour plus de détails
                    
                    Format de réponse :
                    1. "Basé sur votre profil, je vous conseille..."
                    2. "Voici des ressources pertinentes :"
                    3. "Souhaitez-vous être rappelé rapidement pour..."
                    """
                }, {
                    "role": "user",
                    "content": json.dumps({
                        "lead_data": lead_data,
                        "qcm_data": qcm_data
                    }, indent=2)
                }],
                temperature=0.7
            )
            
            return "\n\n" + response.choices[0].message.content
            
        except Exception as e:
            return "\n\nJe peux vous proposer un échange avec l'un de nos experts pour une analyse plus détaillée. Souhaitez-vous être rappelé ?"
