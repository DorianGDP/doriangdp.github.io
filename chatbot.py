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
        self.model = "gpt-4o"  # ou le modèle que vous souhaitez utiliser
        self.conv_storage = ConversationStorage()
        self.info_collector = InfoCollector()
        
        # Initialisation de Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)

    async def generate_gpt_response(self, user_message: str, collected_info: dict, is_valid: bool, next_question: str = None) -> str:
        """Génère une réponse GPT contextuelle"""
        try:
            system_prompt = f"""Tu es Emma, une conseillère en gestion de patrimoine professionnelle et empathique.
            
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
                    {"role": "system", "content": "Tu es un assistant spécialisé dans l'extraction d'informations."},
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
            system_prompt = f"""Tu es Emma, une conseillère en gestion de patrimoine empathique.
            
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
    
            response = await self.client.chat.completions.create(
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
    
            # Extraction et sauvegarde des informations du message
            extracted_info = await self.extract_info_from_message(user_message, current_field)
            await self.save_conversation_message(
                conversation_id, 
                user_message, 
                'user',
                extracted_info
            )
    
            # Validation de la réponse
            is_valid, validated_value, error_msg = self.info_collector.validate_response(
                current_field, user_message, collected_info
            )
    
            if is_valid:
                # Mise à jour des informations collectées
                self.conv_storage.update_info(conversation_id, {current_field: validated_value})
                await self.update_database(conversation_id, {current_field: validated_value})
                
                # Mise à jour du contexte
                updated_info = collected_info.copy()
                updated_info[current_field] = validated_value
                
                # Mise à jour du score si un lead_id existe
                conversation = self.conv_storage.get_conversation(conversation_id)
                if conversation.get('lead_id'):
                    await self.update_lead_score(conversation['lead_id'])
                    await self.check_need_followup(conversation['lead_id'])
    
                # Déterminer la prochaine question
                next_field = self.info_collector.get_current_field(updated_info)
                if next_field:
                    next_question = self.info_collector.get_field_question(next_field, updated_info)
                else:
                    # Générer l'analyse finale si toutes les informations sont collectées
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
    
                # Générer la réponse avec GPT
                response = await self.generate_gpt_response(
                    user_message,
                    updated_info,
                    is_valid,
                    next_question if 'next_question' in locals() else None
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
                    'options': self.info_collector.get_field_options(next_field) if next_field else [],
                    'valid': True,
                    'should_proceed': True
                }
            else:
                # Générer une réponse pour une validation échouée
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
    
            response = await self.client.chat.completions.create(
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
    async def save_conversation_message(self, conversation_id: str, content: str, message_type: str, extracted_info: dict = None):
        """
        Sauvegarde un message dans la base de données avec les informations extraites
        """
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            
            # Si pas de conversation_id dans Supabase, la créer
            conv_record = self.supabase.table('conversations')\
                .select('id')\
                .eq('conversation_id', conversation_id)\
                .execute()
                
            if not conv_record.data:
                # Créer la conversation si elle n'existe pas
                conv_insert = self.supabase.table('conversations').insert({
                    'conversation_id': conversation_id,
                    'status': 'en_cours',
                    'lead_id': conversation.get('lead_id'),
                    'created_at': datetime.utcnow().isoformat()
                }).execute()
                conv_db_id = conv_insert.data[0]['id']
            else:
                conv_db_id = conv_record.data[0]['id']
    
            # Préparer les métadonnées du message
            metadata = {
                'timestamp': datetime.utcnow().isoformat(),
                'type': message_type,
                'conversation_stage': self.get_conversation_stage(conversation)
            }
    
            # Créer l'entrée du message
            message_data = {
                'conversation_id': conv_db_id,
                'content': content,
                'message_type': message_type,
                'metadata': metadata,
                'extracted_info': extracted_info or {},
                'created_at': datetime.utcnow().isoformat()
            }
    
            self.supabase.table('messages').insert(message_data).execute()
    
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
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            lead_id = conversation.get('lead_id')
            
            # Vérifier si une conversation existe déjà
            if not lead_id:
                # Rechercher la conversation par conversation_id
                existing_conversation = await self.supabase.table('conversations').select('lead_id').eq('conversation_id', conversation_id).execute()
                if existing_conversation.data:
                    lead_id = existing_conversation.data[0]['lead_id']
                    conversation['lead_id'] = lead_id
            
            # Création ou mise à jour du lead
            if not lead_id and ('email' in info or 'phone' in info):
                # Rechercher un lead existant par email ou téléphone
                existing_lead = None
                if 'email' in info:
                    existing_lead = await self.supabase.table('leads').select('id').eq('email', info['email']).execute()
                if not existing_lead and 'phone' in info:
                    existing_lead = await self.supabase.table('leads').select('id').eq('phone', info['phone']).execute()
                
                if existing_lead and existing_lead.data:
                    lead_id = existing_lead.data[0]['id']
                    conversation['lead_id'] = lead_id
                else:
                    # Créer un nouveau lead
                    lead_data = {
                        "status": "nouveau",
                        "source": "chatbot",
                        "created_at": datetime.utcnow().isoformat(),
                        **{k: info[k] for k in ['first_name', 'last_name', 'email', 'phone'] if k in info}
                    }
                    lead_response = await self.supabase.table('leads').insert(lead_data).execute()
                    lead_id = lead_response.data[0]['id']
                    conversation['lead_id'] = lead_id
    
                    # Créer la conversation associée
                    await self.supabase.table('conversations').insert({
                        "lead_id": lead_id,
                        "conversation_id": conversation_id,
                        "status": "en_cours",
                        "created_at": datetime.utcnow().isoformat()
                    }).execute()
    
            # Si nous avons un lead_id, mettre à jour les informations patrimoniales
            if lead_id:
                # Conversion des valeurs pour la table patrimoine_info
                conversions = {
                    'income': {
                        "Moins de 30 000€": 25000,
                        "30 000€ - 50 000€": 40000,
                        "50 000€ - 100 000€": 75000,
                        "Plus de 100 000€": 125000
                    },
                    'patrimoine': {
                        "Moins de 50 000€": 25000,
                        "50 000€ - 200 000€": 125000,
                        "200 000€ - 500 000€": 350000,
                        "Plus de 500 000€": 750000
                    }
                }
    
                # Préparation des données patrimoniales
                patrimoine_data = {
                    "lead_id": lead_id,
                    "updated_at": datetime.utcnow().isoformat()
                }
    
                # Mapping des champs
                field_mappings = {
                    'age': ('age', int),
                    'profession': ('profession', str),
                    'situation_familiale': ('situation_familiale', str),
                    'income': ('revenus_annuels', lambda x: conversions['income'].get(x, 0)),
                    'patrimoine': ('patrimoine_total', lambda x: conversions['patrimoine'].get(x, 0)),
                    'objectifs': ('objectifs', lambda x: x.split(',') if isinstance(x, str) else [x])
                }
    
                # Construction des données patrimoniales
                for source_field, (target_field, converter) in field_mappings.items():
                    if source_field in info:
                        try:
                            value = info[source_field]
                            converted_value = converter(value)
                            if converted_value is not None:
                                patrimoine_data[target_field] = converted_value
                        except (ValueError, TypeError) as e:
                            logging.error(f"Erreur de conversion pour {source_field}: {e}")
    
                # Mise à jour des informations patrimoniales si nous avons des données
                if len(patrimoine_data) > 2:  # Plus que juste lead_id et updated_at
                    await self.supabase.table('patrimoine_info').upsert(patrimoine_data).execute()
    
                # Mise à jour du lead si nécessaire
                lead_update_data = {
                    k: info[k]
                    for k in ['first_name', 'last_name', 'email', 'phone']
                    if k in info and info[k]
                }
                if lead_update_data:
                    lead_update_data['updated_at'] = datetime.utcnow().isoformat()
                    await self.supabase.table('leads').update(lead_update_data).eq('id', lead_id).execute()
    
        except Exception as e:
            logging.error(f"Erreur de mise à jour de la base de données: {str(e)}")
            raise

    async def save_recommendations(self, conversation_id: str, collected_info: dict, recommendations: List[str]):
        """Sauvegarde les préconisations générées dans la base de données"""
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            lead_id = conversation.get('lead_id')
            
            if not lead_id:
                return
                
            # Créer une entrée pour chaque préconisation
            for idx, recommendation in enumerate(recommendations, 1):
                preconisation_data = {
                    "lead_id": lead_id,
                    "conversation_id": conversation_id,
                    "contenu": recommendation,
                    "priorite": idx,
                    "type_preconisation": "chatbot",
                    "statut": "générée",
                    "created_at": datetime.utcnow().isoformat()
                }
                
                await self.supabase.table('preconisations').insert(preconisation_data).execute()
                
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des préconisations: {str(e)}")
            raise
    
    async def extract_recommendations(self, gpt_response: str) -> List[str]:
        """Extrait les recommandations d'une réponse GPT"""
        try:
            system_prompt = """Extrais les recommandations principales de cette réponse.
            Format attendu : Une liste de recommandations claires et concises."""
            
            response = await self.client.chat.completions.create(
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

    async def update_lead_score(self, lead_id: str) -> None:
        """
        Met à jour le score du lead en fonction des informations collectées
        """
        try:
            # Récupérer toutes les informations du lead
            lead_data = await self.supabase.table('leads')\
                .select('*')\
                .eq('id', lead_id)\
                .single()\
                .execute()
    
            patrimoine_data = await self.supabase.table('patrimoine_info')\
                .select('*')\
                .eq('lead_id', lead_id)\
                .single()\
                .execute()
    
            # Calcul du score de base
            base_score = 0
            
            # Score pour les informations de contact
            if lead_data.data:
                if lead_data.data.get('email'):
                    base_score += 20
                if lead_data.data.get('phone'):
                    base_score += 15
                if lead_data.data.get('first_name') and lead_data.data.get('last_name'):
                    base_score += 15
                elif lead_data.data.get('first_name') or lead_data.data.get('last_name'):
                    base_score += 10
    
            # Score pour les informations patrimoniales
            if patrimoine_data.data:
                # Score basé sur le patrimoine
                patrimoine_total = patrimoine_data.data.get('patrimoine_total', 0)
                if patrimoine_total > 1000000:
                    base_score += 50
                elif patrimoine_total > 500000:
                    base_score += 30
                elif patrimoine_total > 100000:
                    base_score += 20
                elif patrimoine_total > 0:
                    base_score += 10
    
                # Score basé sur les revenus
                revenus = patrimoine_data.data.get('revenus_annuels', 0)
                if revenus > 100000:
                    base_score += 30
                elif revenus > 50000:
                    base_score += 20
                elif revenus > 30000:
                    base_score += 10
    
                # Score basé sur les objectifs définis
                objectifs = patrimoine_data.data.get('objectifs', [])
                if objectifs:
                    base_score += len(objectifs) * 5
    
                # Score basé sur l'âge (segment privilégié)
                age = patrimoine_data.data.get('age', 0)
                if 35 <= age <= 65:
                    base_score += 15
    
                # Score basé sur la profession
                professions_privilegiees = [
                    "Chef d'entreprise",
                    "Profession libérale",
                    "Cadre supérieur"
                ]
                if patrimoine_data.data.get('profession') in professions_privilegiees:
                    base_score += 20
    
            # Mise à jour du score dans la base de données
            await self.supabase.table('leads')\
                .update({'score': base_score, 'updated_at': datetime.utcnow().isoformat()})\
                .eq('id', lead_id)\
                .execute()
    
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour du score: {str(e)}")
            raise
    
    async def check_need_followup(self, lead_id: str) -> bool:
        """
        Détermine si un suivi est nécessaire en fonction du score et des informations
        """
        try:
            lead_data = await self.supabase.table('leads')\
                .select('score')\
                .eq('id', lead_id)\
                .single()\
                .execute()
                
            if lead_data.data and lead_data.data.get('score', 0) >= 70:
                await self.supabase.table('conversations')\
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
        - Présente les recommandations de manière claire et structurée
        - Termine par une incitation à l'action claire"""
    
        try:
            # Générer l'analyse avec GPT
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es Emma, conseillère patrimoniale expérimentée."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
    
            gpt_response = response.choices[0].message.content
    
            # Extraire et sauvegarder les recommandations
            recommendations = await self.extract_recommendations(gpt_response)
            if recommendations:
                await self.save_recommendations(conversation_id, collected_info, recommendations)
    
            # Mettre à jour le statut de la conversation
            conversation = self.conv_storage.get_conversation(conversation_id)
            if conversation.get('lead_id'):
                await self.supabase.table('conversations').update({
                    'status': 'terminée',
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('conversation_id', conversation_id).execute()
    
            return gpt_response
    
        except Exception as e:
            logging.error(f"Erreur dans generate_final_analysis: {str(e)}")
            return "Je suis désolée, je ne peux pas générer l'analyse complète pour le moment. Un conseiller va vous recontacter rapidement."

    async def repondre_question(self, question: str, conversation_id: str) -> dict:
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            collected_info = conversation['info_collected']
    
            # Première interaction
            if not collected_info.get('initial_query'):
                # Sauvegarder la question initiale
                self.conv_storage.update_info(conversation_id, {'initial_query': question})
                
                # Générer une réponse personnalisée pour la première interaction
                system_prompt = """Tu es Emma, conseillère en gestion de patrimoine. 
                
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
                        'content': "Bonjour ! Je suis Emma, votre conseillère en gestion de patrimoine. Pour mieux vous accompagner dans votre projet, j'aimerais d'abord faire votre connaissance. Quel est votre prénom ?",
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
                'options': []
            }
