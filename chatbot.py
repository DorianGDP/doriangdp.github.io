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

    def __init__(self, api_key: str):
        """Initialise le chatbot avec les clés API nécessaires"""
        openai.api_key = api_key
        
        # Initialisation de Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # État de la conversation
        self.conversation_states = {}
        
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
        
    def extract_info_from_message(self, message: str) -> Dict:
        """Extrait les informations pertinentes du message utilisateur"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
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
            
    def get_next_question(self, state: Dict) -> Optional[str]:
        """Détermine la prochaine question à poser basée sur l'état actuel"""
        if not state.get('name'):
            return "Pour mieux vous accompagner, pourriez-vous me dire comment vous vous appelez ?"
        
        if not state.get('email'):
            return f"Merci {state['name']}. Pour pouvoir vous envoyer une analyse détaillée, quelle est votre adresse email ?"
            
        if not state.get('objectifs'):
            return "Quels sont vos objectifs patrimoniaux ? (Par exemple : épargne, investissement, préparation retraite...)"
            
        if not state.get('patrimoine'):
            return "Pour vous conseiller au mieux, quel est approximativement votre patrimoine actuel ?"
            
        if not state.get('revenus'):
            return "Et quels sont vos revenus annuels ?"
            
        if not state.get('phone'):
            return "Enfin, pour qu'un de nos experts puisse vous recontacter, quel est votre numéro de téléphone ?"
            
        return None
        
    def generate_analysis(self, state: Dict) -> str:
        """Génère une analyse personnalisée basée sur les informations collectées"""
        try:
            prompt = f"""
            Génère une analyse patrimoniale personnalisée pour un client avec le profil suivant :
            - Nom : {state.get('name')}
            - Objectifs : {state.get('objectifs')}
            - Patrimoine : {state.get('patrimoine')}
            - Revenus : {state.get('revenus')}
            
            Format requis :
            1. Synthèse de la situation
            2. 2-3 recommandations principales
            3. Suggestion d'investissements adaptés
            4. Point sur la fiscalité
            
            Ton : professionnel mais accessible
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
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
            
    async def save_to_supabase(self, state: Dict, conversation_id: str):
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
            
    def extract_number(self, text: str) -> float:
        """Extrait un nombre d'une chaîne de caractères"""
        try:
            # Supprimer les symboles monétaires et les espaces
            cleaned = re.sub(r'[€\s]', '', text)
            # Convertir les K/M en milliers/millions
            if 'K' in cleaned.upper():
                cleaned = str(float(cleaned.upper().replace('K', '')) * 1000)
            elif 'M' in cleaned.upper():
                cleaned = str(float(cleaned.upper().replace('M', '')) * 1000000)
            return float(re.sub(r'[^\d.]', '', cleaned))
        except:
            return 0.0
            
    def repondre_question(self, question: str, conversation_id: str) -> Dict:
        """Point d'entrée principal pour traiter une question"""
        try:
            # Initialiser ou récupérer l'état de la conversation
            state = self.conversation_states.get(conversation_id, {})
            
            # Extraire les nouvelles informations de la question
            new_info = self.extract_info_from_message(question)
            state.update(new_info)
            
            # Valider les informations critiques
            if new_info.get('email') and not self.validate_email(new_info['email']):
                return {
                    'reponse': "Cette adresse email ne semble pas valide. Pourriez-vous la vérifier ?",
                    'conversation_id': conversation_id,
                    'type': 'text'
                }
                
            if new_info.get('phone') and not self.validate_phone(new_info['phone']):
                return {
                    'reponse': "Ce numéro de téléphone ne semble pas valide. Pourriez-vous le vérifier ?",
                    'conversation_id': conversation_id,
                    'type': 'text'
                }
                
            # Sauvegarder l'état mis à jour
            self.conversation_states[conversation_id] = state
            
            # Déterminer la prochaine question
            next_question = self.get_next_question(state)
            
            # Si toutes les informations sont collectées
            if not next_question:
                # Générer l'analyse
                analysis = self.generate_analysis(state)
                # Sauvegarder dans Supabase
                self.save_to_supabase(state, conversation_id)
                return {
                    'reponse': analysis,
                    'conversation_id': conversation_id,
                    'type': 'analysis'
                }
                
            # Sinon, poser la prochaine question
            return {
                'reponse': next_question,
                'conversation_id': conversation_id,
                'type': 'text'
            }
            
        except Exception as e:
            print(f"Erreur dans repondre_question : {str(e)}")
            return {
                'reponse': "Désolé, je n'ai pas pu traiter votre demande. Pouvez-vous reformuler ?",
                'conversation_id': conversation_id,
                'type': 'text'
            }
