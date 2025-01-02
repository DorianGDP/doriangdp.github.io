import faiss
import numpy as np
import json
from openai import OpenAI
import os
import time
from supabase import create_client

class ChatBot:
    # Questions initiales plus naturelles
    INITIAL_GREETINGS = [
        "Bonjour ! Je suis Emma, votre conseillère en gestion de patrimoine. Comment puis-je vous aider aujourd'hui ?",
        "Bonjour ! Je suis Emma, ravie de vous rencontrer. En quoi puis-je vous être utile aujourd'hui ?",
        "Bonjour ! Je m'appelle Emma et je suis là pour vous accompagner dans vos projets patrimoniaux. Que puis-je faire pour vous ?"
    ]

    # Questions de qualification plus naturelles
    QUALIFICATION_QUESTIONS = {
        'name': {
            'natural_triggers': [
                "Je serais ravie de vous aider. Pour personnaliser nos échanges, puis-je connaître votre prénom ?",
                "Pour mieux vous accompagner dans votre projet, comment souhaitez-vous que je vous appelle ?",
                "Avant d'aller plus loin dans notre discussion, pourriez-vous me dire comment vous vous appelez ?"
            ],
            'followup': [
                "Enchantée {name} ! Parlons de votre projet. Que souhaitez-vous réaliser ?",
                "Ravi de vous rencontrer {name} ! Dites-moi ce qui vous préoccupe en matière de patrimoine."
            ]
        },
        'contact': {
            'natural_triggers': [
                "D'ailleurs {name}, pour pouvoir vous envoyer une analyse détaillée de votre situation, sur quelle adresse email puis-je vous la faire parvenir ?",
                "{name}, afin de pouvoir vous transmettre des informations personnalisées, quelle est votre adresse email ?"
            ],
            'followup': [
                "Parfait ! Je note votre email. Pour affiner mon analyse, j'aurais besoin d'en savoir un peu plus sur votre situation."
            ]
        }
    }
    QCM_QUESTIONS = {
        'objectifs': {
            'question': "Quels objectifs souhaitez-vous atteindre ?",
            'options': [
                "Obtenir des revenus complémentaires",
                "Investir en immobilier",
                "Développer mon patrimoine",
                "Réduire mes impôts",
                "Préparer ma retraite",
                "Transmettre mon patrimoine",
                "Placer ma trésorerie excédentaire",
                "Autres"
            ]
        },
        'patrimoine': {
            'question': "Quel est votre patrimoine financier ?",
            'options': [
                "Moins de 20 000€",
                "Entre 20 000€ et 50 000€",
                "Entre 50 000€ et 100 000€",
                "Entre 100 000€ et 250 000€",
                "Entre 250 000€ et 500 000€",
                "Entre 500 000€ et 1 000 000€",
                "Entre 1 000 000€ et 2 500 000€",
                "Plus de 2 500 000€"
            ]
        },
        'revenus': {
            'question': "Quel est votre revenu annuel ?",
            'options': [
                "Moins de 30 000€",
                "Entre 30 000€ et 60 000€",
                "Entre 60 000€ et 90 000€",
                "Entre 90 000€ et 120 000€",
                "Entre 120 000€ et 150 000€",
                "Entre 150 000€ et 200 000€",
                "Entre 200 000€ et 250 000€",
                "Plus de 250 000€"
            ]
        },
        'telephone': {
            'question': "Souhaitez-vous être rappelé rapidement ? Si oui, quel est votre numéro de téléphone ?"
        }
    }
    SYSTEM_PROMPT = """Tu es Emma, l'assistante virtuelle de gestiondepatrimoine.com.
    
        SÉQUENCE DE CONVERSATION :
        1. Obtenir nom et email (PRIORITAIRE)
        2. Présenter le questionnaire QCM en expliquant son intérêt
        3. Guider à travers les 3 questions une par une
        4. Proposer le numéro de téléphone pour un rappel rapide
    
        RÈGLES DE PRÉSENTATION DES QCM :
        1. Ne présenter qu'UNE question à la fois
        2. Attendre la réponse avant de passer à la suivante
        3. Confirmer chaque réponse reçue
        4. Si réponse hors options, guider gentiment vers les choix disponibles
    
        APRÈS COLLECTE COMPLÈTE :
        1. Remercier pour les informations
        2. Fournir une préconisation personnalisée basée sur les réponses
        3. Inclure des liens pertinents vers le site
        4. Proposer un contact téléphonique rapide
    
        FORMAT DE PRÉCONISATION :
        1. Résumé de la situation
        2. Recommandations principales
        3. Liens vers contenus pertinents
        4. Proposition de contact personnalisé"""

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
        
    def generer_preconisation(self, lead_data):
            """Génère une préconisation personnalisée basée sur les informations collectées"""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "system",
                        "content": """Génère une préconisation patrimoniale personnalisée.
                        Format requis:
                        1. Synthèse de la situation
                        2. 2-3 recommandations principales
                        3. Liens vers des articles pertinents du site
                        4. Proposition de suivi
    
                        IMPORTANT: 
                        - Rester concret et actionnable
                        - Inclure des liens réels du site
                        - Maintenir un ton professionnel mais accessible"""
                    }, {
                        "role": "user",
                        "content": f"Informations client:\n{json.dumps(lead_data, indent=2)}"
                    }],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                return "Erreur lors de la génération de la préconisation"
                
    def get_next_question(self, lead_info, qcm_progress):
        """Détermine la prochaine question dans la séquence"""
        # Vérifier d'abord nom et email
        if not lead_info.get('name'):
            return np.random.choice(self.QUALIFICATION_QUESTIONS['name'])
        if not lead_info.get('contact'):
            return np.random.choice(self.QUALIFICATION_QUESTIONS['contact'])
        
        # Ensuite passer aux questions QCM dans l'ordre
        if not qcm_progress['objectifs']:
            return self.QCM_QUESTIONS['objectifs']
        if not qcm_progress['patrimoine']:
            return self.QCM_QUESTIONS['patrimoine']
        if not qcm_progress['revenus']:
            return self.QCM_QUESTIONS['revenus']
        if not qcm_progress['telephone']:
            return self.QCM_QUESTIONS['telephone']
        
        return None

    def update_lead_data(self, conversation_id, lead_data):
        try:
            existing_lead = self.supabase.table('conversations')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            data_to_update = {
                'lead_data': lead_data.get('lead_data', {}),
                'qcm_responses': lead_data.get('qcm_responses', {}),
                'status': lead_data.get('status', 'new'),
                'needs_followup': lead_data.get('needs_followup', False),
                'wants_callback': lead_data.get('wants_callback', False)
            }
    
            if existing_lead.data:
                self.supabase.table('conversations')\
                    .update(data_to_update)\
                    .eq('conversation_id', conversation_id)\
                    .execute()
            else:
                self.supabase.table('conversations').insert({
                    'conversation_id': conversation_id,
                    **data_to_update
                }).execute()
            return True
        except Exception as e:
            print(f"Erreur Supabase: {str(e)}")
            return False
            
    def track_lead_info(self, conversation_id, new_info, interaction=None):
        """Analyse et stocke les informations du lead"""
        try:
            data = self.supabase.table('conversations')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            if data.data:
                record = data.data[0]
                lead_data = record.get('lead_data', {})
                history = record.get('conversation_history', [])
                qcm_progress = record.get('qcm_progress', {})
            else:
                lead_data = {}
                history = []
                qcm_progress = {
                    'objectifs': False,
                    'patrimoine': False,
                    'revenus': False,
                    'telephone': False
                }
    
            # Mettre à jour les informations et la progression
            if new_info:
                lead_data.update(new_info)
                for key in new_info:
                    if key in qcm_progress:
                        qcm_progress[key] = True
    
            # Vérifier si toutes les infos sont collectées
            if all(qcm_progress.values()) and not lead_data.get('preconisation'):
                lead_data['preconisation'] = self.generer_preconisation(lead_data)
    
            # Sauvegarder les mises à jour
            data_to_save = {
                'lead_data': lead_data,
                'conversation_history': history if interaction else history + [interaction],
                'qcm_progress': qcm_progress
            }
    
            if data.data:
                self.supabase.table('conversations').update(data_to_save)\
                    .eq('conversation_id', conversation_id).execute()
            else:
                self.supabase.table('conversations').insert({
                    'conversation_id': conversation_id,
                    **data_to_save
                }).execute()
    
            return lead_data, history, qcm_progress
            
        except Exception as e:
            print(f"Erreur lors du tracking des informations: {str(e)}")
            return {}, [], {}

    def extract_lead_info(self, text):
        """Extraire les informations du texte avec GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Correction du modèle
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
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Erreur dans extract_lead_info: {str(e)}")
            return {}

    def generer_reponse(self, question, conversation_id):
        try:
            # Extraire les infos de la question
            new_info = self.extract_lead_info(question)
            lead_data, history = self.track_lead_info(conversation_id, new_info)
            
            # Si premier message
            if not history:
                return {
                    'reponse': np.random.choice(self.INITIAL_GREETINGS),
                    'conversation_id': conversation_id,
                    'type': 'text'
                }

            # Construire le contexte pour GPT
            context = self.build_conversation_context(lead_data, history)
            
            # Générer la réponse avec GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}\nContexte: {context}"}
                ],
                temperature=0.7
            )

            reponse = response.choices[0].message.content
            
            # Si toutes les infos sont collectées, générer une préconisation
            if self.is_lead_complete(lead_data):
                preconisation = self.generer_preconisation(lead_data)
                reponse += f"\n\n{preconisation}"
                
            return {
                'reponse': reponse,
                'conversation_id': conversation_id,
                'type': 'text'
            }

        except Exception as e:
            print(f"Erreur dans generer_reponse: {str(e)}")
            return {
                'reponse': "Désolé, pourriez-vous reformuler votre question ?",
                'conversation_id': conversation_id,
                'type': 'text'
            }


    def valider_reponse_qcm(self, question_type, reponse):
        """Vérifie si la réponse correspond aux options du QCM"""
        if question_type not in self.QCM_QUESTIONS:
            return False
            
        if question_type == 'telephone':
            # Validation basique pour numéro de téléphone
            return bool(reponse and len(reponse.replace(' ', '').replace('.', '')) >= 10)
            
        return reponse in self.QCM_QUESTIONS[question_type]['options']
    
    def repondre_question(self, question, conversation_id=None):
        """Point d'entrée principal du chatbot"""
        if conversation_id is None:
            conversation_id = str(time.time())
        
        try:
            reponse = self.generer_reponse(question, conversation_id)
            
            # Extraction des infos pour déterminer l'étape suivante
            new_info = self.extract_lead_info(question)
            lead_data, history, qcm_progress = self.track_lead_info(conversation_id, new_info)
            
            # Si c'est le premier message, demander le nom
            if not history:
                return {
                    'reponse': "Bonjour ! 👋 Je suis votre assistant personnel en gestion de patrimoine. Pour mieux vous accompagner, puis-je connaître votre nom ?",
                    'conversation_id': conversation_id,
                    'type': 'text'
                }
    
            # Séquence de qualification
            if not lead_data.get('name'):
                return {
                    'reponse': np.random.choice(self.QUALIFICATION_QUESTIONS['name']),
                    'conversation_id': conversation_id,
                    'type': 'text'
                }
            
            if not lead_data.get('contact'):
                return {
                    'reponse': np.random.choice(self.QUALIFICATION_QUESTIONS['contact']),
                    'conversation_id': conversation_id,
                    'type': 'text'
                }
            
            # Séquence QCM
            if not qcm_progress.get('objectifs'):
                return {
                    'type': 'qcm',
                    'question': self.QCM_QUESTIONS['objectifs']['question'],
                    'options': self.QCM_QUESTIONS['objectifs']['options'],
                    'conversation_id': conversation_id
                }
            
            if not qcm_progress.get('patrimoine'):
                return {
                    'type': 'qcm',
                    'question': self.QCM_QUESTIONS['patrimoine']['question'],
                    'options': self.QCM_QUESTIONS['patrimoine']['options'],
                    'conversation_id': conversation_id
                }
            
            if not qcm_progress.get('revenus'):
                return {
                    'type': 'qcm',
                    'question': self.QCM_QUESTIONS['revenus']['question'],
                    'options': self.QCM_QUESTIONS['revenus']['options'],
                    'conversation_id': conversation_id
                }
            
            if not qcm_progress.get('telephone'):
                # Question pour le numéro de téléphone
                return {
                    'type': 'telephone',
                    'question': self.QCM_QUESTIONS['telephone']['question'],
                    'conversation_id': conversation_id
                }
            
            # Si toutes les infos sont collectées, générer une préconisation
            if all(qcm_progress.values()) and not lead_data.get('preconisation'):
                preconisation = self.generer_preconisation(lead_data)
                lead_data['preconisation'] = preconisation
                self.update_lead_data(conversation_id, lead_data)
                
                return {
                    'type': 'preconisation',
                    'reponse': preconisation,
                    'conversation_id': conversation_id
                }
            
            # Si on arrive ici, c'est une conversation normale
            return {
                'reponse': reponse,
                'conversation_id': conversation_id,
                'type': 'text'
            }
            
        except Exception as e:
            print(f"Erreur dans repondre_question: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                'reponse': "Désolé, une erreur s'est produite. Pouvez-vous reformuler votre question ?",
                'conversation_id': conversation_id,
                'type': 'text'
            }
    
    def valider_et_nettoyer_telephone(self, numero):
        """Valide et nettoie un numéro de téléphone"""
        # Supprimer tous les caractères non numériques
        numero_clean = ''.join(filter(str.isdigit, numero))
        
        # Vérifier la longueur (10 chiffres pour la France)
        if len(numero_clean) == 10:
            # Format: 06 12 34 56 78
            return ' '.join([numero_clean[i:i+2] for i in range(0, 10, 2)])
        
        # Format international
        if len(numero_clean) > 10 and numero_clean.startswith('33'):
            numero_clean = '0' + numero_clean[2:]
            if len(numero_clean) == 10:
                return ' '.join([numero_clean[i:i+2] for i in range(0, 10, 2)])
        
        return None
