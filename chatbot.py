from openai import OpenAI
from supabase import create_client
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Callable
import asyncio
import logging
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationStorage:
    """Gère le stockage des conversations en mémoire"""
    def __init__(self):
        self._conversations = {}

    def get_conversation(self, conversation_id: str) -> dict:
        """Récupère ou crée une nouvelle conversation"""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                'initial_query': None,
                'messages': [],
                'info_collected': {},
                'lead_id': None
            }
        return self._conversations[conversation_id]

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

import re
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime

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
                    "30 000€ - 50 000€",
                    "50 000€ - 100 000€",
                    "Plus de 100 000€"
                ],
                'error_message': "Pourriez-vous choisir parmi les tranches de revenus suivantes :",
                'extraction_hints': ['revenus', 'salaire', 'gagner', 'euros par an', '€/an']
            },
            {
                'field': 'patrimoine',
                'question': lambda info: self._generate_patrimoine_question(info),
                'required': True,
                'type': 'choice',
                'options': [
                    "Moins de 50 000€",
                    "50 000€ - 200 000€",
                    "200 000€ - 500 000€",
                    "Plus de 500 000€"
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
                    "Préparer ma retraite",
                    "Optimiser ma fiscalité",
                    "Investir dans l'immobilier",
                    "Protéger mes proches",
                    "Transmettre mon patrimoine",
                    "Développer mon patrimoine",
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
        """Récupère la question pour un champ donné"""
        field_info = self.get_field_info(field)
        if not field_info:
            return None
            
        question = field_info['question']
        if callable(question):
            return question(collected_info)
        return question
        
    def get_field_options(self, field: str) -> list:
        """Récupère les options pour un champ donné"""
        field_info = self.get_field_info(field)
        if field_info and field_info.get('type') == 'choice':
            return field_info.get('options', [])
        return []
        
    def validate_response(self, field: str, value: str, collected_info: dict) -> Tuple[bool, Optional[str], Optional[str]]:
        field_info = self.get_field_info(field)
        if not field_info:
            return False, None, "Champ inconnu"
    
        value = value.strip()
        if not value:
            return False, None, field_info['error_message']
    
        # Ajout de la validation pour les choix multiples
        if field_info.get('multiple', False) and field_info['type'] == 'choice':
            values = [v.strip() for v in value.split(',')]
            valid_values = [v for v in values if v in field_info['options']]
            if valid_values:
                return True, valid_values, None

        # Validation par regex si définie
        if 'regex' in field_info['validation_rules']:
            pattern = field_info['validation_rules']['regex']
            if not re.match(pattern, value):
                return False, None, field_info['error_message']

        # Validations spécifiques
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

    def extract_info_from_message(self, message: str, field: str) -> Optional[str]:
        """Tente d'extraire l'information demandée du message"""
        field_info = self.get_field_info(field)
        if not field_info:
            return None

        # Si c'est un choix, chercher une correspondance exacte
        if field_info['type'] == 'choice':
            for option in field_info['options']:
                if option.lower() in message.lower():
                    return option
            return None

        # Pour les autres types, utiliser les indices d'extraction
        for hint in field_info.get('extraction_hints', []):
            if hint.lower() in message.lower():
                # Extraire le contexte autour de l'indice
                index = message.lower().find(hint.lower())
                start = max(0, index - 20)
                end = min(len(message), index + len(hint) + 20)
                context = message[start:end]
                
                # Appliquer la regex si définie
                if 'regex' in field_info['validation_rules']:
                    matches = re.findall(field_info['validation_rules']['regex'], context)
                    if matches:
                        return matches[0]

        return None

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
        """Initialise le chatbot avec les dépendances nécessaires"""
        self.client = OpenAI(api_key=api_key)
        self.conv_storage = ConversationStorage()  # Changed from storage to conv_storage
        self.info_collector = InfoCollector()
        
        # Initialisation de Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)

    async def process_response(self, user_message: str, conversation_id: str, collected_info: dict, current_field: str) -> dict:
        """Traite la réponse de l'utilisateur"""
        try:
            # Ajouter la progression
            completion = self.info_collector.get_completion_percentage(collected_info)
            
            system_prompt = f"""Tu es Emma, une conseillère en gestion de patrimoine professionnelle et empathique.
            
            CONTEXTE:
            - Question initiale du client: {collected_info.get('initial_query', '')}
            - Prénom connu: {collected_info.get('first_name', '')}
            - Champ actuel: {current_field}
            - Progression: {completion}%
            
            DIRECTIVES:
            1. Sois naturelle et empathique dans tes réponses
            2. Si la réponse est valide, fais un bref retour positif avant de passer à la suite
            3. Si la réponse est invalide, explique poliment pourquoi
            4. Adapte ton langage selon le profil (plus formel si patrimoine élevé)
            5. N'utilise jamais "enchantée" après le premier message
            6. Ne pose qu'une seule question à la fois
            7. Si la progression est > 75%, encourage le client en mentionnant qu'il ne reste que quelques informations"""
    
            field_info = self.info_collector.get_field_info(current_field)
            if field_info:
                is_valid, validated_value, error_msg = self.info_collector.validate_response(
                    current_field, user_message, collected_info
                )
                
                if is_valid:
                    self.conv_storage.update_info(conversation_id, {current_field: validated_value})
                    await self.update_database(conversation_id, {current_field: validated_value})
                    
                    # Obtenir la prochaine question
                    next_field = self.info_collector.get_current_field(collected_info)
                    if next_field:
                        next_question = self.info_collector.get_field_question(next_field, collected_info)
                        return {
                            'type': 'text',
                            'content': next_question,
                            'options': self.info_collector.get_field_options(next_field),
                            'valid': True,
                            'should_proceed': True
                        }
                    else:
                        # Toutes les informations sont collectées
                        return {
                            'type': 'text',
                            'content': await self.generate_final_analysis(collected_info),
                            'options': [],
                            'valid': True,
                            'should_proceed': True
                        }
                else:
                    return {
                        'type': 'text',
                        'content': error_msg,
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
            
            system_prompt = """Tu es Emma, conseillère patrimoniale. Réponds de façon concise et naturelle.
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
    
    # Ajout d'une méthode pour sauvegarder les messages
    async def save_conversation_message(self, conversation_id: str, content: str, message_type: str):
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            if conversation.get('lead_id'):
                await self.supabase.table('messages').insert({
                    'conversation_id': conversation_id,
                    'content': content,
                    'message_type': message_type,
                    'created_at': datetime.utcnow().isoformat()
                }).execute()
        except Exception as e:
            print(f"Error saving message: {str(e)}")
    
    async def update_database(self, conversation_id: str, info: dict):
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            lead_id = conversation.get('lead_id')
            
            # Vérifier si une conversation existe déjà
            if not lead_id:
                existing_conversation = self.supabase.table('conversations').select('lead_id').eq('conversation_id', conversation_id).execute()
                if existing_conversation.data:
                    lead_id = existing_conversation.data[0]['lead_id']
                    conversation['lead_id'] = lead_id
                    return
    
                # Vérifier si l'email existe déjà
                if 'email' in info:
                    existing_lead = self.supabase.table('leads').select('id').eq('email', info['email']).execute()
                    if existing_lead.data:
                        lead_id = existing_lead.data[0]['id']
                        conversation['lead_id'] = lead_id
                        return
    
                lead_data = {
                    "status": "nouveau",
                    "source": "chatbot",
                    "created_at": datetime.utcnow().isoformat(),
                    **{k: info[k] for k in ['first_name', 'last_name', 'email', 'phone'] if k in info}
                }
    
                # Créer ou mettre à jour le lead
                if lead_id:
                    self.supabase.table('leads').update(lead_data).eq('id', lead_id).execute()
                else:
                    lead_response = self.supabase.table('leads').insert(lead_data).execute()
                    lead_id = lead_response.data[0]['id']
                    conversation['lead_id'] = lead_id
                    
                    # Créer la conversation
                    self.supabase.table('conversations').upsert({
                        "lead_id": lead_id,
                        "conversation_id": conversation_id,
                        "status": "en_cours"
                    }).execute()
            
            # Mettre à jour les informations patrimoniales
            if lead_id:
                conversions = {
                    'income': {"Moins de 30 000€": 30000, "30 000€ - 50 000€": 50000,
                              "50 000€ - 100 000€": 100000, "Plus de 100 000€": 150000},
                    'patrimoine': {"Moins de 50 000€": 50000, "50 000€ - 200 000€": 200000,
                                 "200 000€ - 500 000€": 500000, "Plus de 500 000€": 1000000}
                }
                
                patrimoine_fields = {
                    'age': ('age', int),
                    'profession': ('profession', str),
                    'income': ('revenus_annuels', lambda x: conversions['income'].get(x, 0)),
                    'patrimoine': ('patrimoine_total', lambda x: conversions['patrimoine'].get(x, 0))
                }
                
                patrimoine_data = {"lead_id": lead_id}
                for key, (db_field, converter) in patrimoine_fields.items():
                    if key in info:
                        try:
                            patrimoine_data[db_field] = converter(info[key])
                        except (ValueError, TypeError) as e:
                            print(f"Erreur de conversion pour {key}: {e}")
                
                if len(patrimoine_data) > 1:
                    self.supabase.table('patrimoine_info').upsert({
                        **patrimoine_data,
                        "updated_at": datetime.utcnow().isoformat()
                    }).execute()
    
        except Exception as e:
            print(f"Erreur de mise à jour de la base de données: {str(e)}")
            raise



    async def generate_final_analysis(self, collected_info: dict) -> str:
        """Génère l'analyse finale et les recommandations"""
        
        prompt = f"""En tant que conseillère en gestion de patrimoine, fais une analyse personnalisée:

        PROFIL CLIENT:
        {json.dumps(collected_info, indent=2)}

        STRUCTURE DE LA RÉPONSE:
        1. Remerciement personnalisé avec le prénom
        2. Bref résumé de la situation patrimoniale
        3. Réponse précise à la question initiale: {collected_info.get('initial_query')}
        4. 2-3 recommandations personnalisées
        5. Proposition de rendez-vous pour approfondir

        CONSIGNES:
        - Sois précise et professionnelle
        - Montre que tu as bien compris leurs enjeux
        - Donne des conseils concrets mais garde des éléments pour le RDV
        - Termine par une incitation à l'action claire"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es Emma, conseillère patrimoniale expérimentée."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logging.error(f"Erreur dans generate_final_analysis: {str(e)}")
            return "Je suis désolée, je ne peux pas générer l'analyse complète pour le moment. Un conseiller va vous recontacter rapidement."

    async def repondre_question(self, question: str, conversation_id: str) -> dict:
        """Point d'entrée principal pour le traitement des messages"""
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            collected_info = conversation['info_collected']
    
            # Enregistrer la question initiale si première interaction
            if not collected_info.get('initial_query'):
                self.conv_storage.update_info(conversation_id, {'initial_query': question})
                return {
                    'type': 'text',
                    'content': "Bonjour ! Je suis Emma, votre conseillère en gestion de patrimoine. Pour mieux répondre à votre question, j'aimerais en savoir un peu plus sur vous. Pour commencer, quel est votre prénom ?",
                    'options': []
                }
    
            # Déterminer le champ actuel et traiter la réponse
            current_field = self.info_collector.get_current_field(collected_info)
            return await self.process_response(question, conversation_id, collected_info, current_field)
    
        except Exception as e:
            logging.error(f"Erreur dans repondre_question: {str(e)}")
            return {
                'type': 'text',
                'content': "Je suis désolée, une erreur s'est produite. Pouvons-nous reprendre notre conversation ?",
                'options': []
            }
