import openai
from supabase import create_client, Client
import os
import json
import re
from typing import Optional

class ConversationStorage:
    def __init__(self):
        self._conversations = {}

    def get_conversation(self, conversation_id: str) -> dict:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                'state': {},
                'messages': []
            }
        return self._conversations[conversation_id]

    def update_state(self, conversation_id: str, new_info: dict):
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {'state': {}, 'messages': []}
        self._conversations[conversation_id]['state'].update(new_info)

    def get_state(self, conversation_id: str) -> dict:
        return self._conversations.get(conversation_id, {}).get('state', {})

    def add_message(self, conversation_id: str, message: dict):
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {'state': {}, 'messages': []}
        self._conversations[conversation_id]['messages'].append(message)

class ChatBot:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.storage = ConversationStorage()
        
        # Initialisation de Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
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


    def extract_number(self, text: str) -> float:
        try:
            cleaned = re.sub(r'[€\s]', '', text)
            if 'K' in cleaned.upper():
                cleaned = str(float(cleaned.upper().replace('K', '')) * 1000)
            elif 'M' in cleaned.upper():
                cleaned = str(float(cleaned.upper().replace('M', '')) * 1000000)
            elif 'million' in text.lower():
                cleaned = str(float(re.sub(r'[^\d.]', '', cleaned)) * 1000000)
            elif 'mille' in text.lower():
                cleaned = str(float(re.sub(r'[^\d.]', '', cleaned)) * 1000)
            return float(re.sub(r'[^\d.]', '', cleaned))
        except:
            return 0.0
    
    def analyze_user_message(self, message: str, current_state: dict) -> dict:
        """Analyse le message pour en extraire les informations"""
        info = {}
        
        # Si nous n'avons pas encore le nom
        if not current_state.get('name'):
            name_match = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', message)
            if name_match:
                info['name'] = ' '.join(name_match)
                print(f"Nom trouvé : {info['name']}")

        # Si nous n'avons pas encore l'email
        if not current_state.get('email'):
            email_match = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message)
            if email_match:
                info['email'] = email_match[0]
                print(f"Email trouvé : {info['email']}")

        # Si nous n'avons pas encore le patrimoine
        if not current_state.get('patrimoine'):
            if any(word in message.lower() for word in ['euro', '€', 'euros', 'million']):
                numbers = re.findall(r'\d+(?:\s*\d*)*(?:\s*[kKmM])?', message)
                if numbers:
                    montant = self.extract_number(message)
                    if montant > 0:
                        info['patrimoine'] = montant
                        print(f"Patrimoine trouvé : {info['patrimoine']}")

        # Analyse des revenus
        if not current_state.get('revenus') and ('revenu' in message.lower() or 'gagne' in message.lower()):
            montant_match = re.findall(r'\b\d+(?:\s*[kKmM€]?\s*€?)?\b', message)
            if montant_match:
                info['revenus'] = self.extract_number(message)

        # Analyse du téléphone
        if not current_state.get('phone'):
            phone_match = re.findall(r'(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}', message)
            if phone_match:
                info['phone'] = re.sub(r'[^\d+]', '', phone_match[0])

        # Analyse des objectifs
        if not current_state.get('objectifs'):
            objectifs_keywords = {
                'épargne': 'épargne',
                'per': 'préparation retraite',
                'retraite': 'préparation retraite',
                'invest': 'investissement',
                'immobilier': 'investissement immobilier',
                'impôt': 'optimisation fiscale',
                'fiscal': 'optimisation fiscale'
            }
            
            for keyword, objectif in objectifs_keywords.items():
                if keyword in message.lower():
                    info['objectifs'] = objectif
                    break

        return info
    
    def validate_email(self, email: str) -> bool:
        """Valide le format d'une adresse email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
        
    def validate_phone(self, phone: str) -> bool:
        """Valide le format d'un numéro de téléphone français"""
        # Nettoyer le numéro
        phone = re.sub(r'[^\d+]', '', phone)
        # Vérifier le format (français)
        pattern = r'^(?:(?:\+|00)33|0)\d{9}$'
        return bool(re.match(pattern, phone))
        
    def extract_info_from_message(self, message: str) -> dict:
        """Extrait les informations pertinentes du message utilisateur"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4oo",
                messages=[{
                    "role": "system",
                    "content": """Analyse le message et extrait les informations suivantes au format JSON :
                    - name: prénom et/ou nom mentionnés
                    - email: adresse email
                    - phone: numéro de téléphone
                    - objectifs: objectifs patrimoniaux mentionnés
                    - patrimoine: montant ou fourchette de patrimoine
                    - revenus: montant ou fourchette de revenus
                    Renvoie uniquement les informations explicitement mentionnées."""
                }, {
                    "role": "user",
                    "content": message
                }],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message['content'])
        except Exception as e:
            print(f"Erreur lors de l'extraction d'informations : {str(e)}")
            return {}

    async def update_supabase(self, state: dict, conversation_id: str) -> None:
        """Met à jour progressivement les données dans Supabase"""
        try:
            # Créer ou mettre à jour le lead
            lead_data = {}
            if state.get('name'):
                names = state['name'].split()
                lead_data['first_name'] = names[0]
                lead_data['last_name'] = ' '.join(names[1:]) if len(names) > 1 else None
            if state.get('email'):
                lead_data['email'] = state['email']
            if state.get('phone'):
                lead_data['phone'] = state['phone']
            
            if lead_data:
                lead_response = await self.supabase.table('leads').upsert(lead_data).execute()
                lead_id = lead_response.data[0]['id'] if lead_response.data else None
                
                if lead_id:
                    # Mettre à jour les informations patrimoniales
                    patrimoine_data = {
                        "lead_id": lead_id,
                    }
                    if state.get('objectifs'):
                        patrimoine_data['objectifs'] = [state['objectifs']]
                    if state.get('patrimoine'):
                        patrimoine_data['patrimoine_total'] = float(state['patrimoine'])
                    if state.get('revenus'):
                        patrimoine_data['revenus_annuels'] = float(state['revenus'])
                    
                    if patrimoine_data:
                        await self.supabase.table('patrimoine_info').upsert(patrimoine_data).execute()
                    
                    # Mettre à jour la conversation
                    conversation_data = {
                        "lead_id": lead_id,
                        "conversation_id": conversation_id,
                        "status": "en_cours"
                    }
                    await self.supabase.table('conversations').upsert(conversation_data).execute()

        except Exception as e:
            print(f"Erreur Supabase : {str(e)}")
    
    def get_next_question(self, state: dict) -> str:
        if not state.get('name'):
            return "Pour mieux vous accompagner, pourriez-vous me dire comment vous vous appelez ?"
        
        if not state.get('email'):
            return f"Merci {state['name']}. Pour vous envoyer une analyse personnalisée, quelle est votre adresse email ?"
        
        if not state.get('patrimoine'):
            return f"Pour adapter mes recommandations à votre situation {state['name']}, quel est approximativement votre patrimoine actuel ?"
        
        if not state.get('revenus'):
            return f"Merci. Et quels sont vos revenus annuels environ ?"
        
        if not state.get('phone'):
            return "Parfait. Pour qu'un de nos experts puisse vous recontacter rapidement, quel est votre numéro de téléphone ?"
        
        return self.generate_analysis(state)
        
    def generate_analysis(self, state: dict) -> str:
        """Génère une analyse personnalisée"""
        try:
            prompt = f"""
            Génère une analyse patrimoniale personnalisée pour un client avec le profil suivant :
            - Nom : {state.get('name')}
            - Objectifs : {state.get('objectifs')}
            - Patrimoine : {state.get('patrimoine')}€
            - Revenus : {state.get('revenus')}€
            
            Format requis :
            1. Synthèse de la situation
            2. 2-3 recommandations principales
            3. Suggestion d'investissements adaptés
            4. Point sur la fiscalité
            
            Ton : professionnel mais accessible
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": prompt
                }],
                temperature=0.7
            )
            
            return response.choices[0].message['content']
        except Exception as e:
            print(f"Erreur lors de la génération de l'analyse : {str(e)}")
            return "Désolé, je n'ai pas pu générer l'analyse pour le moment."
            
    async def save_to_supabase(self, state: dict, conversation_id: str):
        """Sauvegarde les informations dans Supabase"""
        try:
            # Créer ou mettre à jour le lead
            lead_data = {
                "first_name": state.get('name', '').split()[0] if state.get('name') else None,
                "last_name": ' '.join(state.get('name', '').split()[1:]) if state.get('name') else None,
                "email": state.get('email'),
                "phone": state.get('phone'),
                "status": "nouveau"
            }
            
            lead_response = await self.supabase.table('leads').upsert(lead_data).execute()
            lead_id = lead_response.data[0]['id']
            
            # Sauvegarder les informations patrimoniales
            patrimoine_data = {
                "lead_id": lead_id,
                "objectifs": state.get('objectifs', '').split(','),
                "patrimoine_total": self.extract_number(state.get('patrimoine', '0')),
                "revenus_annuels": self.extract_number(state.get('revenus', '0'))
            }
            
            await self.supabase.table('patrimoine_info').upsert(patrimoine_data).execute()
            
            # Mettre à jour la conversation
            conversation_data = {
                "lead_id": lead_id,
                "conversation_id": conversation_id,
                "status": "en_cours",
                "needs_followup": True
            }
            
            await self.supabase.table('conversations').upsert(conversation_data).execute()
            
            return True
        except Exception as e:
            print(f"Erreur Supabase : {str(e)}")
            return False

    def format_response(self, state: dict) -> str:
        """Formate la réponse en fonction de l'état de la conversation"""
        if not state.get('name'):
            return "Pour mieux vous accompagner, pourriez-vous me dire comment vous vous appelez ?"
            
        if not state.get('email'):
            return f"Merci {state['name']}. Pour vous envoyer une analyse personnalisée, quelle est votre adresse email ?"
            
        if not state.get('objectifs'):
            if state.get('initial_question'):
                return f"D'accord {state['name']}, j'ai bien noté votre intérêt pour {state['initial_question']}. Pour vous conseiller au mieux, quels sont vos autres objectifs patrimoniaux ?"
            return "Quels sont vos objectifs patrimoniaux ? (épargne, investissement, retraite...)"
            
        if not state.get('patrimoine'):
            return f"Pour adapter mes recommandations à votre situation {state['name']}, quel est approximativement votre patrimoine actuel ?"
            
        if not state.get('revenus'):
            return "Et quels sont vos revenus annuels ?"
            
        if not state.get('phone'):
            return f"Parfait {state['name']}. Pour qu'un de nos experts puisse vous recontacter et approfondir votre projet, quel est votre numéro de téléphone ?"
            
        # Si toutes les informations sont collectées
        return self.generate_analysis(state)
    
    async def repondre_question(self, question: str, conversation_id: str) -> dict:
        """Point d'entrée principal pour traiter une question"""
        try:
            # Initialiser ou récupérer l'état
            state = self.storage.get_state(conversation_id)
            
            # Vérifier si c'est la première question
            if not state:
                if 'per' in question.lower():
                    state['objectifs'] = 'préparation retraite'
            
            # Analyser le message
            new_info = self.analyze_user_message(question, state)
            print(f"Nouvelles informations extraites: {new_info}")
            
            # Mettre à jour l'état
            if new_info:
                self.storage.update_state(conversation_id, new_info)
                # Mise à jour dans Supabase
                await self.update_supabase(state, conversation_id)
            
            # Obtenir la prochaine question
            response = self.get_next_question(state)
            
            return {
                'reponse': response,
                'conversation_id': conversation_id,
                'type': 'text' if 'analyse' not in response.lower() else 'analysis'
            }
            
        except Exception as e:
            print(f"Erreur dans repondre_question : {str(e)}")
            return {
                'reponse': "Désolé, une erreur s'est produite. Pouvez-vous reformuler ?",
                'conversation_id': conversation_id,
                'type': 'text'
            }
