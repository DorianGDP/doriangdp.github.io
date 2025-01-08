from openai import OpenAI
from supabase import create_client
import os
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List, Callable
import asyncio
import logging
import re
import random

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationStatus(Enum):
    EN_COURS = 'en_cours'
    TERMINEE = 'terminée'
    NON_TERMINEE = 'non_terminée'

class ConversationManager:
    CONVERSATION_TIMEOUT = timedelta(hours=2)

    @staticmethod
    def is_conversation_expired(start_time: datetime) -> bool:
        """Vérifie si la conversation a dépassé la durée limite"""
        return datetime.utcnow() - start_time > ConversationManager.CONVERSATION_TIMEOUT

    @staticmethod
    async def handle_conversation_timeout(chatbot, conversation_id: str):
        """Gère une conversation expirée"""
        try:
            await chatbot.supabase.table('conversations')\
                .update({
                    'status': ConversationStatus.NON_TERMINEE.value,
                    'updated_at': datetime.utcnow().isoformat(),
                    'timeout_reason': 'conversation_duration_exceeded'
                })\
                .eq('conversation_id', conversation_id)\
                .execute()
            
            return {
                'type': 'text',
                'content': "Je suis désolée, mais notre conversation a dépassé la durée limite. "
                          "Souhaitez-vous recommencer ou être recontacté par un conseiller ?",
                'options': ["Recommencer la conversation", "Être recontacté"]
            }
        except Exception as e:
            logging.error(f"Erreur lors de la gestion du timeout: {str(e)}")
            return None

    @staticmethod
    async def handle_page_close(chatbot, conversation_id: str):
        """Gère la fermeture de la page"""
        try:
            # Vérifier si la conversation était en cours
            conv_data = await chatbot.supabase.table('conversations')\
                .select('status, created_at')\
                .eq('conversation_id', conversation_id)\
                .execute()

            if conv_data.data and conv_data.data[0]['status'] == ConversationStatus.EN_COURS.value:
                await chatbot.supabase.table('conversations')\
                    .update({
                        'status': ConversationStatus.NON_TERMINEE.value,
                        'updated_at': datetime.utcnow().isoformat(),
                        'termination_reason': 'page_closed'
                    })\
                    .eq('conversation_id', conversation_id)\
                    .execute()
        except Exception as e:
            logging.error(f"Erreur lors de la gestion de fermeture de page: {str(e)}")

    @staticmethod
    async def reset_conversation(chatbot, conversation_id: str):
        """Réinitialise complètement une conversation"""
        try:
            new_conversation_id = chatbot.generate_unique_id()
            
            # Créer une nouvelle conversation
            await chatbot.supabase.table('conversations').insert({
                'conversation_id': new_conversation_id,
                'status': ConversationStatus.EN_COURS.value,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }).execute()
            
            return new_conversation_id
        except Exception as e:
            logging.error(f"Erreur lors de la réinitialisation: {str(e)}")
            return None
            
class ConversationStorage:
    """Gère le stockage des conversations en mémoire"""
    def __init__(self):
        self._conversations = {}

    def get_conversation(self, conversation_id: str) -> dict:
        """Récupère ou crée une nouvelle conversation"""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = self.create_empty_conversation()
        return self._conversations[conversation_id]

    def create_empty_conversation(self) -> dict:
        """Crée une nouvelle conversation vide avec la structure initiale"""
        return {
            'initial_query': None,
            'messages': [],
            'info_collected': {},
            'current_step': 0,
            'is_complete': False
        }

    def reset_conversation(self, conversation_id: str):
        """Réinitialise une conversation à son état initial"""
        self._conversations[conversation_id] = self.create_empty_conversation()

    def clear_all_conversations(self):
        """Efface toutes les conversations en mémoire"""
        self._conversations.clear()

    def add_message(self, conversation_id: str, message: dict):
        """Ajoute un message à la conversation"""
        conv = self.get_conversation(conversation_id)
        conv['messages'].append(message)

    def update_info(self, conversation_id: str, new_info: dict):
        """Met à jour les informations collectées"""
        conv = self.get_conversation(conversation_id)
        conv['info_collected'].update(new_info)

    def set_initial_query(self, conversation_id: str, query: str):
        """Définit la question initiale de la conversation"""
        conv = self.get_conversation(conversation_id)
        if not conv['initial_query']:
            conv['initial_query'] = query

    def get_collected_info(self, conversation_id: str) -> dict:
        """Récupère les informations collectées"""
        conv = self.get_conversation(conversation_id)
        return conv.get('info_collected', {})

    async def sync_with_database(self, conversation_id: str, supabase_client) -> None:
        try:
            conv_data = await supabase_client.table('conversations')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            if conv_data.data:
                conv = conv_data.data[0]
                self._conversations[conversation_id] = {
                    'initial_query': conv.get('initial_query'),
                    'messages': conv.get('messages', []),
                    'info_collected': {
                        'first_name': conv.get('first_name'),
                        'last_name': conv.get('last_name'),
                        'age': conv.get('age'),
                        'email': conv.get('email'),
                        'phone': conv.get('phone'),
                        'profession': conv.get('profession'),
                        'income': conv.get('revenus'),  # Mis à jour
                        'impot_revenu': conv.get('impot_revenu'),  # Nouveau champ
                        'patrimoine': conv.get('patrimoine'),  # Mis à jour
                        'situation_familiale': conv.get('situation_familiale'),
                        'objectifs': conv.get('objectifs', [])
                    },
                    'is_complete': conv.get('status') == 'terminée'
                }
            else:
                self.reset_conversation(conversation_id)
        except Exception as e:
            logging.error(f"Erreur lors de la synchronisation avec la base de données: {str(e)}")
            self.reset_conversation(conversation_id)

class InfoCollector:
    def __init__(self):
        self.info_sequence = [
            {
                'field': 'first_name',
                'question': "Pour commencer, quel est votre prénom ?",
                'required': True,
                'type': 'text',
                'validation_rules': {
                    'min_length': 2,
                    'no_numbers': True,
                    'regex': r'^[A-Za-zÀ-ÿ\-\s]{2,30}$'
                },
                'error_message': "Je n'ai pas bien compris votre prénom. Pourriez-vous le répéter ?",
                'extraction_hints': ['prénom', 'je m\'appelle', 'je suis']
            },
            {
                'field': 'last_name',
                'question': lambda info: f"Merci {info.get('first_name')}. Et quel est votre nom de famille ?",
                'required': True,
                'type': 'text',
                'validation_rules': {
                    'min_length': 2,
                    'no_numbers': True,
                    'regex': r'^[A-Za-zÀ-ÿ\-\s]{2,30}$'
                },
                'error_message': "Je n'ai pas bien saisi votre nom de famille. Pourriez-vous le répéter ?",
                'extraction_hints': ['nom', 'nom de famille']
            },
            {
                'field': 'age',
                'question': lambda info: f"Parfait {info.get('first_name')}. Pour mieux vous conseiller, quel âge avez-vous ?",
                'required': True,
                'type': 'number',
                'validation_rules': {
                    'min_value': 18,
                    'max_value': 100,
                    'regex': r'\b\d{1,2}\b'
                },
                'error_message': "Pourriez-vous me préciser votre âge en chiffres ? Il doit être compris entre 18 et 100 ans.",
                'extraction_hints': ['ans', 'age', 'âge']
            },
            {
                'field': 'email',
                'question': "Pour pouvoir vous envoyer des informations personnalisées, quelle est votre adresse email ?",
                'required': True,
                'type': 'email',
                'validation_rules': {
                    'email_format': True,
                    'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                },
                'error_message': "Cette adresse email ne semble pas valide. Pourriez-vous la vérifier ?",
                'extraction_hints': ['email', '@', 'mail', 'adresse électronique']
            },
            {
                'field': 'phone',
                'question': "Et votre numéro de téléphone pour un échange plus personnalisé ?",
                'required': True,
                'type': 'phone',
                'validation_rules': {
                    'phone_format': True,
                    'regex': r'^(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}$'
                },
                'error_message': "Ce numéro ne semble pas valide. Pourriez-vous me donner un numéro à 10 chiffres ?",
                'extraction_hints': ['téléphone', 'portable', 'mobile', 'fixe']
            },
            {
                'field': 'profession',
                'question': "Quelle est votre situation professionnelle actuelle ?",
                'required': True,
                'type': 'choice',
                'options': [
                    "Salarié du secteur privé",
                    "Fonctionnaire",
                    "Chef d'entreprise",
                    "Profession libérale",
                    "Indépendant / Auto-entrepreneur",
                    "Retraité",
                    "Autre"
                ],
                'error_message': "Je vous propose de choisir parmi les options suivantes :",
                'extraction_hints': ['travail', 'métier', 'profession', 'emploi']
            },
            {
                'field': 'income',
                'question': lambda info: self._generate_income_question(info),
                'required': True,
                'type': 'choice',
                'options': [
                    "Moins de 30 000€",
                    "30 000€ - 40 000€",
                    "40 000€ - 60 000€",
                    "60 000€ - 80 000€",
                    "80 000€ - 100 000€",
                    "100 000€ - 250 000€",
                    "Plus de 250 000€"
                ],
                'error_message': "Pourriez-vous choisir parmi les tranches de revenus suivantes :",
                'extraction_hints': ['revenus', 'salaire', 'gagner', 'euros par an', '€/an']
            },
            {
                'field': 'impot_revenu',
                'question': "Quel est votre montant annuel d'impôt sur le revenu ?",
                'required': True,
                'type': 'choice',
                'options': [
                    "Moins de 2 000€",
                    "Entre 2 000€ et 5 000€",
                    "Entre 5 000€ et 7 500€",
                    "Entre 7 500€ et 15 000€",
                    "Entre 15 000€ et 30 000€",
                    "Plus de 30 000€"
                ],
                'error_message': "Merci de choisir parmi les tranches d'impôt suivantes :",
                'extraction_hints': ['impôt', 'impot', 'ir', 'impôt sur le revenu']
            },
            {
                'field': 'patrimoine',
                'question': lambda info: self._generate_patrimoine_question(info),
                'required': True,
                'type': 'choice',
                'options': [
                    "Moins de 20 000€",
                    "20 000€ - 50 000€",
                    "50 000€ - 100 000€",
                    "100 000€ - 250 000€",
                    "250 000€ - 500 000€",
                    "500 000€ - 1 000 000€",
                    "1 000 000€ - 2 500 000€",
                    "Plus de 2 500 000€"
                ],
                'error_message': "Merci de choisir parmi les tranches de patrimoine suivantes :",
                'extraction_hints': ['patrimoine', 'possède', 'valeur', 'fortune']
            },
            {
                'field': 'situation_familiale',
                'question': "Quelle est votre situation familiale ?",
                'required': True,
                'type': 'choice',
                'options': [
                    "Célibataire",
                    "Marié(e)",
                    "Pacsé(e)",
                    "Divorcé(e)",
                    "Veuf/Veuve"
                ],
                'error_message': "Pouvez-vous préciser votre situation parmi les choix suivants :",
                'extraction_hints': ['marié', 'célibataire', 'pacsé', 'divorcé', 'veuf']
            },
            {
                'field': 'objectifs',
                'question': lambda info: self._generate_objectives_question(info),
                'required': True,
                'type': 'choice',
                'options': [
                    "Obtenir des revenus complémentaires",
                    "Investir en immobilier",
                    "Développer mon patrimoine",
                    "Réduire mes impôts",
                    "Préparer ma retraite",
                    "Protéger ma famille",
                    "Transmettre mon patrimoine",
                    "Placer ma trésorerie excédentaire",
                    "Autre"
                ],
                'multiple': True,
                'error_message': "Quels sont vos principaux objectifs parmi les suivants :",
                'extraction_hints': ['objectif', 'but', 'souhaite', 'veux', 'aimerais']
            }
        ]

    def _generate_income_question(self, info: dict) -> str:
        """Génère une question personnalisée sur les revenus selon le profil"""
        if info.get('profession') == "Retraité":
            return f"D'accord {info.get('first_name')}. Quel est le montant de vos pensions de retraite annuelles ?"
        elif info.get('profession') in ["Chef d'entreprise", "Profession libérale", "Indépendant / Auto-entrepreneur"]:
            return f"En tant que {info.get('profession').lower()}, dans quelle tranche se situent vos revenus annuels ?"
        return "Dans quelle tranche de revenus annuels vous situez-vous ?"

    def _generate_patrimoine_question(self, info: dict) -> str:
        """Génère une question personnalisée sur le patrimoine selon le profil"""
        if int(info.get('age', 0)) > 50:
            return "Après ces années d'activité, dans quelle tranche estimez-vous votre patrimoine global ?"
        elif info.get('profession') in ["Chef d'entreprise", "Profession libérale"]:
            return "En incluant votre outil professionnel, dans quelle tranche de patrimoine vous situez-vous ?"
        return "Concernant votre patrimoine global actuel, dans quelle tranche vous situez-vous ?"

    def _generate_objectives_question(self, info: dict) -> str:
        """Génère une question personnalisée sur les objectifs selon le profil"""
        age = int(info.get('age', 0))
        profession = info.get('profession', '')
        
        if age > 50:
            return f"À {age} ans, quels sont vos principaux objectifs patrimoniaux ? (plusieurs choix possibles)"
        elif profession in ["Chef d'entreprise", "Profession libérale"]:
            return f"En tant que {profession.lower()}, quels sont vos objectifs patrimoniaux prioritaires ?"
        elif age < 35:
            return "En tant que jeune actif, quels sont vos objectifs patrimoniaux ? (plusieurs choix possibles)"
        return "Parmi les objectifs suivants, lesquels vous intéressent le plus ? (plusieurs choix possibles)"

    def get_current_field(self, collected_info: dict) -> Optional[str]:
        """Détermine le prochain champ à collecter"""
        for info in self.info_sequence:
            if info['field'] not in collected_info or not collected_info[info['field']]:
                return info['field']
        return None

    def get_field_info(self, field: str) -> Optional[dict]:
        """Récupère toutes les informations d'un champ"""
        for info in self.info_sequence:
            if info['field'] == field:
                return info
        return None

    def get_field_question(self, field: str, collected_info: dict) -> str:
        """Récupère la question pour un champ donné avec un contexte personnalisé"""
        field_info = self.get_field_info(field)
        if not field_info:
            return None
            
        question = field_info['question']
        if callable(question):
            try:
                return question(collected_info)
            except Exception as e:
                logging.error(f"Erreur lors de la génération de la question dynamique: {str(e)}")
                # Question de secours si la génération échoue
                return self._get_fallback_question(field)
        return question
        
    def get_field_options(self, field: str) -> list:
        """Récupère les options pour un champ donné"""
        field_info = self.get_field_info(field)
        if field_info and field_info.get('type') == 'choice':
            return field_info.get('options', [])
        return []

    def _get_fallback_question(self, field: str) -> str:
        """Fournit une question de secours si la génération dynamique échoue"""
        fallback_questions = {
            'income': "Dans quelle tranche de revenus annuels vous situez-vous ?",
            'patrimoine': "Dans quelle tranche de patrimoine vous situez-vous ?",
            'objectifs': "Quels sont vos principaux objectifs patrimoniaux ?",
        }
        return fallback_questions.get(field, "Pourriez-vous préciser cette information ?")
        
    def validate_response(self, field: str, value: str, collected_info: dict) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Valide la réponse pour un champ donné
        Retourne: (is_valid, normalized_value, error_message)
        """
        field_info = self.get_field_info(field)
        if not field_info:
            return False, None, "Champ inconnu"
    
        value = value.strip()
        if not value:
            return False, None, field_info['error_message']
    
        # Validation selon le type de champ
        if field_info['type'] == 'choice':
            # Gestion des choix multiples
            if field_info.get('multiple', False):
                values = [v.strip() for v in value.split(',')]
                valid_values = [v for v in values if v in field_info['options']]
                if valid_values:
                    return True, ','.join(valid_values), None
                return False, None, field_info['error_message']
            # Choix simple
            if value in field_info['options']:
                return True, value, None
            return False, None, field_info['error_message']
    
        # Validation des autres types avec regex
        if 'validation_rules' in field_info:
            if 'regex' in field_info['validation_rules']:
                pattern = field_info['validation_rules']['regex']
                if not re.match(pattern, value):
                    return False, None, field_info['error_message']
    
            # Validations numériques
            if field_info['type'] == 'number':
                try:
                    num_value = int(value)
                    min_val = field_info['validation_rules'].get('min_value', float('-inf'))
                    max_val = field_info['validation_rules'].get('max_value', float('inf'))
                    if not (min_val <= num_value <= max_val):
                        return False, None, field_info['error_message']
                    return True, str(num_value), None
                except ValueError:
                    return False, None, field_info['error_message']
    
        return True, value, None

    def is_collection_complete(self, collected_info: dict) -> bool:
        """Vérifie si toutes les informations requises ont été collectées"""
        return all(
            info['field'] in collected_info and collected_info[info['field']]
            for info in self.info_sequence 
            if info['required']
        )

    def get_completion_percentage(self, collected_info: dict) -> int:
        """Calcule le pourcentage de complétion des informations"""
        required_fields = [info['field'] for info in self.info_sequence if info['required']]
        if not required_fields:
            return 100
            
        collected_required = sum(
            1 for field in required_fields 
            if field in collected_info and collected_info[field]
        )
        return int((collected_required / len(required_fields)) * 100)
        
class ChatBot:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"  
        self.conv_storage = ConversationStorage()
        self.conversation_manager = ConversationManager()
        self.info_collector = InfoCollector()
        
        # Initialisation de Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)

    async def generate_gpt_response(self, user_message: str, collected_info: dict, is_valid: bool, next_question: str = None) -> str:
        """Génère une réponse GPT contextuelle"""
        try:
            system_prompt = f"""Tu es Patty, un assistante en gestion de patrimoine professionnelle et empathique.
            
            CONTEXTE:
            - Question initiale du client: {collected_info.get('initial_query', '')}
            - Prénom: {collected_info.get('first_name', '')}
            - Dernière réponse valide: {is_valid}
            - Prochaine question: {next_question}
            
            RÈGLES:
            1. Si la réponse était valide:
               - Faire un bref retour positif sans répéter la réponse
               - Poser directement la question suivante
               - Ne pas répéter "merci pour votre réponse"
            2. Rester naturel et empathique
            3. Une seule question à la fois
            4. Éviter les formules répétitives"""
    
            user_prompt = f"""Message du client: {user_message}
            Prochaine question à poser: {next_question}"""
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
    
            return response.choices[0].message.content
    
        except Exception as e:
            logging.error(f"Erreur dans generate_gpt_response: {str(e)}")
            return "Je suis désolée, pouvons-nous continuer notre conversation ?"

    async def extract_info_from_message(self, message: str, current_field: str) -> dict:
        """
        Extrait les informations pertinentes d'un message utilisateur
        """
        try:
            prompt = f"""Analyse ce message et extrait les informations pertinentes.
            Contexte: le champ actuellement demandé est '{current_field}'
            Message: {message}
            
            Format de réponse attendu: JSON avec les informations extraites"""
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es un assistante spécialisé dans l'extraction d'informations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
    
            # Tenter de parser la réponse comme du JSON
            try:
                extracted_info = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                extracted_info = {}
    
            return extracted_info
    
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction d'informations: {str(e)}")
            return {}
    
    async def generate_error_response(self, user_message: str, field_info: dict, error_msg: str, collected_info: dict) -> str:
        """Génère une réponse pour une erreur de validation"""
        try:
            system_prompt = f"""Tu es Patty, un assistante en gestion de patrimoine empathique.
            
            CONTEXTE:
            - Prénom du client: {collected_info.get('first_name', '')}
            - Type d'information demandée: {field_info['field']}
            - Message d'erreur: {error_msg}
            
            RÈGLES:
            1. Expliquer poliment pourquoi la réponse n'est pas valide
            2. Donner un exemple de réponse acceptable
            3. Reformuler la question de manière plus claire
            4. Rester encourageant et professionnel"""
    
            user_prompt = f"""Réponse invalide du client: {user_message}
            Message d'erreur technique: {error_msg}
            Format attendu: {field_info.get('validation_rules', {})}"""
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
    
            return response.choices[0].message.content
    
        except Exception as e:
            logging.error(f"Erreur dans generate_error_response: {str(e)}")
            return error_msg
    
    async def process_response(self, user_message: str, conversation_id: str, collected_info: dict, current_field: str) -> dict:
        try:
            field_info = self.info_collector.get_field_info(current_field)
            if not field_info:
                return {
                    'type': 'text',
                    'content': "Je suis désolée, j'ai perdu le fil de notre conversation. Pouvons-nous reprendre ?",
                    'options': [],
                    'valid': False,
                    'should_proceed': False
                }
    
            # Validation de la réponse
            is_valid, validated_value, error_msg = self.info_collector.validate_response(
                current_field, user_message, collected_info
            )
    
            # Extraction et sauvegarde du message
            extracted_info = await self.extract_info_from_message(user_message, current_field)
            await self.save_conversation_message(
                conversation_id, 
                user_message, 
                'user',
                extracted_info
            )
    
            if is_valid:
                # Mise à jour des informations
                self.conv_storage.update_info(conversation_id, {current_field: validated_value})
                await self.update_database(conversation_id, {current_field: validated_value})
                
                # Mise à jour du score
                await self.update_lead_score(conversation_id)
                
                # Mise à jour du contexte
                updated_info = collected_info.copy()
                updated_info[current_field] = validated_value
    
                # Vérifier si toutes les informations sont collectées
                if self.info_collector.is_collection_complete(updated_info):
                    final_analysis = await self.generate_final_analysis(updated_info, conversation_id)
                    await self.save_conversation_message(
                        conversation_id,
                        final_analysis,
                        'bot',
                        {'type': 'final_analysis'}
                    )
                    return {
                        'type': 'text',
                        'content': final_analysis,
                        'options': [],
                        'valid': True,
                        'should_proceed': True
                    }
    
                # Sinon, continuer avec la prochaine question
                next_field = self.info_collector.get_current_field(updated_info)
                if next_field:
                    next_question = self.info_collector.get_field_question(next_field, updated_info)
                    response = await self.generate_gpt_response(
                        user_message,
                        updated_info,
                        is_valid,
                        next_question
                    )
    
                    await self.save_conversation_message(
                        conversation_id,
                        response,
                        'bot',
                        {'next_field': next_field}
                    )
    
                    return {
                        'type': 'text',
                        'content': response,
                        'options': self.info_collector.get_field_options(next_field),
                        'valid': True,
                        'should_proceed': True
                    }
    
            # Gestion des réponses invalides
            error_response = await self.generate_error_response(
                user_message,
                field_info,
                error_msg,
                collected_info
            )
    
            await self.save_conversation_message(
                conversation_id,
                error_response,
                'bot',
                {'error': error_msg}
            )
    
            return {
                'type': 'text',
                'content': error_response,
                'options': field_info.get('options', []),
                'valid': False,
                'should_proceed': False
            }
    
        except Exception as e:
            logging.error(f"Erreur dans process_response: {str(e)}")
            return {
                'type': 'text',
                'content': "Je suis désolée, j'ai rencontré une difficulté. Pourriez-vous reformuler votre réponse ?",
                'options': [],
                'valid': False,
                'should_proceed': False
            }

    async def generate_response(self, message: str, field: str, next_info: dict, collected_info: dict) -> dict:
        try:
            # Récupérer le prénom s'il existe
            first_name = collected_info.get('first_name', '')
            last_name = collected_info.get('last_name', '')
            initial_query = collected_info.get('initial_query', '')
            
            system_prompt = """Tu es Patty, assistante patrimoniale. Réponds de façon concise et naturelle.
            Quelques règles importantes:
            1. Commence toujours par demander le prénom avant le nom
            2. Si tu as le prénom, utilise-le
            3. Si tu as le nom de famille mais pas le prénom, demande poliment le prénom
            4. Justifie naturellement chaque question par un objectif précis
            5. Ne dis jamais "enchantée" sauf au tout premier message
            6. Sois concise et évite les formules répétitives"""
    
            user_prompt = f"""Contexte :
            Message reçu : {message}
            Prénom client : {first_name}
            Nom client : {last_name}
            Question initiale : {initial_query}
            Prochaine information à demander : {next_info['question']}"""
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
    
            return {
                'type': next_info['type'],
                'content': response.choices[0].message.content,
                'options': next_info.get('options', [])
            }
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return {
                'type': 'text',
                'content': "Je suis désolée, une erreur s'est produite.",
                'options': []
            }

    def generate_unique_id(self) -> str:  # Ajout du 'self'
        timestamp = int(datetime.utcnow().timestamp() * 1000)
        random_suffix = ''.join(random.choices('0123456789abcdef', k=8))
        return f"conv_{timestamp}_{random_suffix}"
    
    async def get_or_create_conversation(self, conversation_id: str) -> str:
        """Récupère une conversation existante ou en crée une nouvelle"""
        try:
            # Vérifier si la conversation existe déjà
            existing_conv = self.supabase.table('conversations')\
                .select('conversation_id')\
                .eq('conversation_id', conversation_id)\
                .execute()
            
            # Si la conversation existe, retourner l'ID
            if existing_conv.data:
                return conversation_id
            
            # Sinon, générer un nouvel ID
            new_id = self.generate_unique_id()
            return new_id
                
        except Exception as e:
            logging.error(f"Erreur lors de la vérification de la conversation: {str(e)}")
            # En cas d'erreur, générer un nouvel ID
            return self.generate_unique_id()
    
    # Ajout d'une méthode pour sauvegarder les messages
    async def save_conversation_message(self, conversation_id: str, content: str, message_type: str, extracted_info: dict = None):
        """Sauvegarde un message dans la conversation"""
        try:
            # Récupérer la conversation existante
            response = self.supabase.table('conversations')\
                .select('messages, id')\
                .eq('conversation_id', conversation_id)\
                .execute()

            if not response.data:
                # Créer une nouvelle conversation
                response = self.supabase.table('conversations').insert({
                    'conversation_id': conversation_id,
                    'messages': [{
                        'type': message_type,
                        'content': content,
                        'timestamp': datetime.utcnow().isoformat(),
                        'metadata': extracted_info or {}
                    }],
                    'status': 'en_cours',
                    'score': 0,
                    'needs_followup': False,
                    'preconisations': [],
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }).execute()
                return

            # Mettre à jour les messages existants
            existing_messages = response.data[0]['messages']
            new_message = {
                'type': message_type,
                'content': content,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': extracted_info or {}
            }
            updated_messages = existing_messages + [new_message]

            # Mettre à jour la conversation
            response = self.supabase.table('conversations')\
                .update({'messages': updated_messages})\
                .eq('id', response.data[0]['id'])\
                .execute()

        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde du message: {str(e)}")
            raise


    def get_conversation_stage(self, conversation: dict) -> str:
        """
        Détermine l'étape actuelle de la conversation
        """
        collected_info = conversation.get('info_collected', {})
        
        if not collected_info:
            return 'début'
            
        if not collected_info.get('first_name'):
            return 'identification'
            
        if not collected_info.get('email') or not collected_info.get('phone'):
            return 'contact'
            
        if not collected_info.get('profession') or not collected_info.get('income'):
            return 'situation_professionnelle'
            
        if not collected_info.get('patrimoine'):
            return 'situation_patrimoniale'
            
        if not collected_info.get('objectifs'):
            return 'objectifs'
            
        return 'conclusion'

    async def update_database(self, conversation_id: str, info: dict):
        """Met à jour les informations de la conversation dans la base de données"""
        try:
            # Préparer les données à mettre à jour
            update_data = {}
            
            # Mapping des champs
            field_mappings = {
                'initial_query': 'initial_query',
                'first_name': 'first_name',
                'last_name': 'last_name',
                'email': 'email',
                'phone': 'phone',
                'age': 'age',
                'profession': 'profession',
                'income': 'revenus',
                'impot_revenu': 'impot_revenu',
                'patrimoine': 'patrimoine',
                'situation_familiale': 'situation_familiale'
            }
    
            # Ajouter directement les valeurs qualitatives pour les champs simples
            for source_field, target_field in field_mappings.items():
                if source_field in info and info[source_field]:
                    update_data[target_field] = info[source_field]
    
            # Traitement spécial pour les objectifs (champ array)
            if 'objectifs' in info:
                # Si c'est déjà une liste
                if isinstance(info['objectifs'], list):
                    update_data['objectifs'] = info['objectifs']
                # Si c'est une chaîne avec des virgules
                elif isinstance(info['objectifs'], str) and ',' in info['objectifs']:
                    update_data['objectifs'] = [obj.strip() for obj in info['objectifs'].split(',')]
                # Si c'est une chaîne simple
                elif isinstance(info['objectifs'], str):
                    update_data['objectifs'] = [info['objectifs'].strip()]
    
            # Mettre à jour la conversation
            if update_data:
                self.supabase.table('conversations')\
                    .update(update_data)\
                    .eq('conversation_id', conversation_id)\
                    .execute()
    
        except Exception as e:
            logging.error(f"Erreur de mise à jour de la base de données: {str(e)}")
            raise

    async def save_recommendations(self, conversation_id: str, recommendations: List[str]):
        """Sauvegarde les préconisations dans la conversation"""
        try:
            # Récupérer la conversation
            conv_record = self.supabase.table('conversations')\
                .select('id, preconisations')\
                .eq('conversation_id', conversation_id)\
                .execute()

            if not conv_record.data:
                return

            # Préparer les préconisations
            preconisations = [
                {
                    'contenu': rec,
                    'priorite': idx + 1,
                    'date_creation': datetime.utcnow().isoformat()
                }
                for idx, rec in enumerate(recommendations)
            ]

            # Mettre à jour la conversation
            self.supabase.table('conversations')\
                .update({
                    'preconisations': preconisations,
                    'status': 'terminée'
                })\
                .eq('id', conv_record.data[0]['id'])\
                .execute()

        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des préconisations: {str(e)}")
            raise
    
    async def extract_recommendations(self, gpt_response: str) -> List[str]:
        """Extrait les recommandations d'une réponse GPT"""
        try:
            system_prompt = """Extrais les recommandations principales de cette réponse.
            Format attendu : Une liste de recommandations claires et concises."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": gpt_response}
                ],
                temperature=0.3
            )
            
            # Traiter la réponse pour extraire les recommandations
            recommendations_text = response.choices[0].message.content
            recommendations = [
                rec.strip() 
                for rec in recommendations_text.split('\n') 
                if rec.strip() and not rec.strip().startswith(('•', '-', '*', '1.', '2.', '3.'))
            ]
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction des recommandations: {str(e)}")
            return []

    async def update_lead_score(self, conversation_id: str) -> None:
        try:
            # Récupérer les informations de la conversation
            conv_response = self.supabase.table('conversations')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            if not conv_response.data:
                return
    
            conv_data = conv_response.data[0]
            base_score = 0
    
            # Score pour les informations de contact
            if conv_data.get('email'):
                base_score += 20
            if conv_data.get('phone'):
                base_score += 15
            if conv_data.get('first_name') and conv_data.get('last_name'):
                base_score += 15
            elif conv_data.get('first_name') or conv_data.get('last_name'):
                base_score += 10
    
            # Score basé sur le patrimoine
            patrimoine = conv_data.get('patrimoine', '')
            if "Plus de 2 500 000€" in patrimoine:
                base_score += 50
            elif "1 000 000€" in patrimoine:
                base_score += 40
            elif "500 000€" in patrimoine:
                base_score += 30
            elif "250 000€" in patrimoine:
                base_score += 20
            elif "100 000€" in patrimoine:
                base_score += 10
    
            # Score basé sur les revenus
            revenus = conv_data.get('revenus', '')
            if "Plus de 250 000€" in revenus:
                base_score += 30
            elif "100 000€" in revenus:
                base_score += 25
            elif "80 000€" in revenus:
                base_score += 20
            elif "60 000€" in revenus:
                base_score += 15
            elif "40 000€" in revenus:
                base_score += 10
    
            # Score basé sur l'impôt sur le revenu
            impot = conv_data.get('impot_revenu', '')
            if "Plus de 30 000€" in impot:
                base_score += 20
            elif "15 000€" in impot:
                base_score += 15
            elif "7 500€" in impot:
                base_score += 10
    
            # Autres scores inchangés
            age = int(conv_data.get('age', 0) or 0)
            objectifs = conv_data.get('objectifs', []) or []
            profession = conv_data.get('profession', '')
    
            if objectifs and isinstance(objectifs, list):
                base_score += len(objectifs) * 5
    
            if 35 <= age <= 65:
                base_score += 15
    
            professions_privilegiees = [
                "Chef d'entreprise",
                "Profession libérale",
                "Cadre supérieur"
            ]
            if profession in professions_privilegiees:
                base_score += 20
    
            # Mise à jour du score
            self.supabase.table('conversations')\
                .update({
                    'score': base_score,
                    'updated_at': datetime.utcnow().isoformat(),
                    'needs_followup': base_score >= 70
                })\
                .eq('conversation_id', conversation_id)\
                .execute()
    
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour du score: {str(e)}")
    
    async def check_need_followup(self, lead_id: str) -> bool:
        """Détermine si un suivi est nécessaire en fonction du score"""
        try:
            # Récupérer le lead
            lead_response = self.supabase.table('leads')\
                .select('score')\
                .eq('id', lead_id)\
                .execute()
                
            # Vérifier si le lead existe et a un score suffisant
            if lead_response.data and lead_response.data[0]['score'] >= 70:
                self.supabase.table('conversations')\
                    .update({
                        'needs_followup': True,
                        'updated_at': datetime.utcnow().isoformat()
                    })\
                    .eq('lead_id', lead_id)\
                    .execute()
                return True
                
            return False
    
        except Exception as e:
            logging.error(f"Erreur lors de la vérification du besoin de suivi: {str(e)}")
            return False

    async def generate_final_analysis(self, collected_info: dict, conversation_id: str) -> str:
        """Génère l'analyse finale et les recommandations"""
        try:
            prompt = f"""En tant que conseillère en gestion de patrimoine, réalise une analyse personnalisée et naturelle.
            
                        PROFIL CLIENT:
                        {json.dumps(collected_info, indent=2)}
            
                        QUESTION INITIALE:
                        {collected_info.get('initial_query')}
            
                        CONSIGNES DE STYLE:
                        - Adopte un ton chaleureux et professionnel
                        - Évite les sections numérotées et les titres
                        - Utilise des transitions naturelles entre les sujets
                        - Garde un style conversationnel tout en restant professionnel
                        - Intègre les recommandations de manière fluide dans le texte
                        - Fais référence aux informations personnelles du client pour personnaliser le message
            
                        POINTS À COUVRIR:
                        1. Un accueil personnalisé qui montre que tu as compris leur situation
                        2. Une analyse concise de leur situation actuelle
                        3. Une réponse ciblée à leur question initiale
                        4. 2-3 recommandations pertinentes intégrées naturellement
                        5. Une conclusion qui :
                           - Souligne l'importance d'un accompagnement personnalisé
                           - Justifie pourquoi un rendez-vous avec un conseiller serait bénéfique
                           - Mentionne qu'ils seront recontactés via leurs coordonnées fournies
                           - Se termine sur une note positive et engageante"""
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es Patty, assistante patrimoniale expérimentée."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
    
            gpt_response = response.choices[0].message.content
    
            # Extraire et sauvegarder les recommandations
            recommendations = await self.extract_recommendations(gpt_response)
            if recommendations:
                await self.save_recommendations(conversation_id, recommendations)
    
            # Mettre à jour le statut de la conversation
            await self.supabase.table('conversations')\
                .update({
                    'status': ConversationStatus.TERMINEE.value,
                    'updated_at': datetime.utcnow().isoformat(),
                    'termination_reason': 'analysis_completed'
                })\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            # Déclencher l'analyse pour le conseiller
            await self.handle_conversation_end(conversation_id)
            
            return gpt_response
    
        except Exception as e:
            logging.error(f"Erreur dans generate_final_analysis: {str(e)}")
            return "Je suis désolée, je ne peux pas générer l'analyse complète pour le moment. Un conseiller va vous recontacter rapidement."

    async def reset_conversation(self, conversation_id: str) -> None:
        try:
            # Réinitialiser la mémoire
            self.conv_storage.reset_conversation(conversation_id)
            
            # Réinitialiser dans la base de données
            await self.supabase.table('conversations')\
                .update({
                    'initial_query': None,
                    'messages': [],
                    'first_name': None,
                    'last_name': None,
                    'age': None,
                    'email': None,
                    'phone': None,
                    'profession': None,
                    'revenus': None,  
                    'impot_revenu': None,  
                    'patrimoine': None,  
                    'situation_familiale': None,
                    'objectifs': None,
                    'score': 0,
                    'status': 'en_cours',
                    'preconisations': [],
                    'needs_followup': False,
                    'updated_at': datetime.utcnow().isoformat()
                })\
                .eq('conversation_id', conversation_id)\
                .execute()
        except Exception as e:
            logging.error(f"Erreur lors de la réinitialisation de la conversation: {str(e)}")

    async def initialize_session(self, conversation_id: str) -> str:
        """Initialise ou récupère une session de conversation"""
        try:
            # Vérifier si la conversation existe
            response = self.supabase.table('conversations')\
                .select('conversation_id, status')\
                .eq('conversation_id', conversation_id)\
                .execute()

            if not response.data:
                # Créer une nouvelle conversation
                new_id = self.generate_unique_id()
                response = self.supabase.table('conversations').insert({
                    'conversation_id': new_id,
                    'status': 'en_cours',
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }).execute()
                
                if hasattr(response, 'error') and response.error:
                    raise Exception(response.error)
                    
                return new_id

            return conversation_id

        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de la session: {str(e)}")
            return self.generate_unique_id()

    async def analyze_conversation_for_advisor(self, conversation_id: str) -> List[str]:
        """Analyse la conversation complète pour extraire des informations pertinentes pour le conseiller"""
        try:
            # Récupérer la conversation
            conv_data = await self.supabase.table('conversations')\
                .select('*')\
                .eq('conversation_id', conversation_id)\
                .execute()
            
            if not conv_data.data:
                return []

            conversation = conv_data.data[0]
            messages = conversation.get('messages', [])
            
            prompt = f"""Analyste tous les messages de cette conversation entre un prospect et le chatbot.
            
            CONTEXTE:
            Messages: {json.dumps(messages)}
            Informations déjà collectées:
            - Prénom: {conversation.get('first_name')}
            - Nom: {conversation.get('last_name')}
            - Âge: {conversation.get('age')}
            - Email: {conversation.get('email')}
            - Téléphone: {conversation.get('phone')}
            - Profession: {conversation.get('profession')}
            - Revenus: {conversation.get('revenus')}
            - Impôt sur le revenu: {conversation.get('impot_revenu')}
            - Patrimoine: {conversation.get('patrimoine')}
            - Situation familiale: {conversation.get('situation_familiale')}
            - Objectifs: {conversation.get('objectifs')}

            TÂCHE:
            1. Analyse tous les messages pour trouver des informations supplémentaires intéressantes non capturées par les champs standards
            2. Identifie les signaux d'intérêt ou d'urgence dans la demande
            3. Repère les mentions de projets spécifiques ou de timing
            4. Note tout détail sur la situation familiale élargie ou professionnelle
            5. Capture les préoccupations ou inquiétudes exprimées

            FORMAT DE RÉPONSE:
            - Liste de points clés pertinents pour le conseiller
            - Chaque point doit être concis et actionnable
            - Ne pas répéter les informations déjà dans les champs standards
            - Inclure uniquement les informations vraiment utiles au conseiller"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es un analyste expert en gestion de patrimoine."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            analysis = [
                point.strip()
                for point in response.choices[0].message.content.split('\n')
                if point.strip() and not point.startswith(('•', '-', '*', '1.', '2.', '3.'))
            ]

            # Sauvegarder l'analyse dans un nouveau champ 'advisor_notes'
            await self.supabase.table('conversations')\
                .update({
                    'advisor_notes': analysis,
                    'updated_at': datetime.utcnow().isoformat()
                })\
                .eq('conversation_id', conversation_id)\
                .execute()

            return analysis

        except Exception as e:
            logging.error(f"Erreur lors de l'analyse pour le conseiller: {str(e)}")
            return []
    
    async def check_conversation_timeout(self, conversation_id: str) -> bool:
        """Vérifie si une conversation a dépassé la durée limite"""
        try:
            conv_data = await self.supabase.table('conversations')\
                .select('created_at, status')\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            if not conv_data.data:
                return True
    
            created_at = datetime.fromisoformat(conv_data.data[0]['created_at'])
            is_expired = ConversationManager.is_conversation_expired(created_at)
    
            if is_expired and conv_data.data[0]['status'] == ConversationStatus.EN_COURS.value:
                # Mettre à jour le statut si la conversation est expirée
                await self.handle_timeout(conversation_id)
                return True
    
            return is_expired
    
        except Exception as e:
            logging.error(f"Erreur lors de la vérification du timeout: {str(e)}")
            return False
    
    async def handle_timeout(self, conversation_id: str):
        """Gère une conversation qui a expiré"""
        try:
            await self.supabase.table('conversations')\
                .update({
                    'status': ConversationStatus.NON_TERMINEE.value,
                    'updated_at': datetime.utcnow().isoformat(),
                    'termination_reason': 'timeout',
                    'completion_date': datetime.utcnow().isoformat()
                })\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            # Analyser la conversation même si elle est incomplète
            await self.analyze_conversation_for_advisor(conversation_id)
    
        except Exception as e:
            logging.error(f"Erreur lors de la gestion du timeout: {str(e)}")
    
    async def handle_page_unload(self, conversation_id: str):
        """Gère la fermeture de la page"""
        try:
            conv_data = await self.supabase.table('conversations')\
                .select('status')\
                .eq('conversation_id', conversation_id)\
                .execute()
    
            if conv_data.data and conv_data.data[0]['status'] == ConversationStatus.EN_COURS.value:
                await self.supabase.table('conversations')\
                    .update({
                        'status': ConversationStatus.NON_TERMINEE.value,
                        'updated_at': datetime.utcnow().isoformat(),
                        'termination_reason': 'page_unload',
                        'completion_date': datetime.utcnow().isoformat()
                    })\
                    .eq('conversation_id', conversation_id)\
                    .execute()
    
                # Analyser la conversation même si elle est incomplète
                await self.analyze_conversation_for_advisor(conversation_id)
    
        except Exception as e:
            logging.error(f"Erreur lors de la gestion de la fermeture de page: {str(e)}")
    
    async def handle_conversation_end(self, conversation_id: str):
        """Gère la fin d'une conversation"""
        try:
            # Générer l'analyse pour le conseiller
            advisor_notes = await self.analyze_conversation_for_advisor(conversation_id)
            
            # Mettre à jour les informations finales de la conversation
            await self.supabase.table('conversations')\
                .update({
                    'status': ConversationStatus.TERMINEE.value,
                    'updated_at': datetime.utcnow().isoformat(),
                    'advisor_notes': advisor_notes,
                    'completion_date': datetime.utcnow().isoformat(),
                })\
                .eq('conversation_id', conversation_id)\
                .execute()
    
        except Exception as e:
            logging.error(f"Erreur lors de la gestion de fin de conversation: {str(e)}")

    
    async def repondre_question(self, question: str, conversation_id: str) -> dict:
        try:
            # Vérifier d'abord si la conversation existe et n'est pas expirée
            conv_data = await self.supabase.table('conversations')\
                .select('created_at, status')\
                .eq('conversation_id', conversation_id)\
                .execute()

            if not conv_data.data:
                return {
                    'type': 'text',
                    'content': "Désolé, je ne trouve pas votre conversation. Voulez-vous en commencer une nouvelle ?",
                    'options': ["Commencer une nouvelle conversation"]
                }

            conv = conv_data.data[0]
            created_at = datetime.fromisoformat(conv['created_at'])

            # Vérifier si la conversation est expirée
            if ConversationManager.is_conversation_expired(created_at):
                return await ConversationManager.handle_conversation_timeout(self, conversation_id)

            # Vérifier le statut de la conversation
            if conv['status'] == ConversationStatus.TERMINEE.value:
                return {
                    'type': 'text',
                    'content': "Cette conversation est terminée. Souhaitez-vous en commencer une nouvelle ?",
                    'options': ["Commencer une nouvelle conversation"]
                }
            elif conv['status'] == ConversationStatus.NON_TERMINEE.value:
                return {
                    'type': 'text',
                    'content': "Votre dernière conversation n'a pas abouti. Voulez-vous la reprendre ou en commencer une nouvelle ?",
                    'options': ["Reprendre la conversation", "Nouvelle conversation"]
                }
            conversation = self.conv_storage.get_conversation(conversation_id)
            collected_info = conversation['info_collected']

            # Première interaction
            if not collected_info.get('initial_query'):
                # Sauvegarder la question initiale en mémoire
                self.conv_storage.update_info(conversation_id, {'initial_query': question})
                
                try:
                    # Sauvegarder dans Supabase - Correction de la syntaxe
                    response = self.supabase.table('conversations')\
                        .upsert({
                            'conversation_id': conversation_id,
                            'initial_query': question,
                            'updated_at': datetime.utcnow().isoformat()
                        }).execute()
                    
                    # Vérifier la réponse
                    if hasattr(response, 'error') and response.error:
                        raise Exception(response.error)
                        
                except Exception as e:
                    logging.warning(f"Impossible de sauvegarder initial_query: {str(e)}")
                
                system_prompt = """Tu es Patty, assistante en gestion de patrimoine. 
                
                TÂCHE:
                - Accueillir le client chaleureusement
                - Faire un bref commentaire sur sa demande initiale pour montrer que tu l'as comprise
                - Introduire naturellement la demande du prénom
                
                RÈGLES:
                - Ne pas répéter mot pour mot sa question
                - Rester concise et professionnelle
                - Être empathique et naturelle"""
                
                user_prompt = f"Question du client: {question}"
                
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=150
                    )
                    
                    return {
                        'type': 'text',
                        'content': response.choices[0].message.content,
                        'options': []
                    }
                    
                except Exception as e:
                    logging.error(f"Erreur lors de la génération de la première réponse: {str(e)}")
                    return {
                        'type': 'text',
                        'content': "Bonjour ! Je suis Patty, votre assistante en gestion de patrimoine. Pour mieux vous accompagner dans votre projet, j'aimerais d'abord faire votre connaissance. Quel est votre prénom ?",
                        'options': []
                    }

            # Pour les interactions suivantes
            current_field = self.info_collector.get_current_field(collected_info)
            return await self.process_response(question, conversation_id, collected_info, current_field)

        except Exception as e:
            logging.error(f"Erreur dans repondre_question: {str(e)}")
            return {
                'type': 'text',
                'content': "Je suis désolée, une erreur s'est produite. Pouvons-nous reprendre notre conversation ?",
                'options': ["Recommencer"]
            }
