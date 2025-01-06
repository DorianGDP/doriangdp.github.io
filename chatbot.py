from openai import OpenAI
from supabase import create_client
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import random
import re

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
                'field': 'initial_query',
                'type': 'text',
                'store': False,
                'required': False
            },
            {
                'field': 'first_name',
                'question': "Pour commencer, quel est votre prénom ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: len(x.strip()) > 1,
                'error_message': "Pourriez-vous me donner votre prénom ?"
            },
            {
                'field': 'last_name',
                'question': "Et votre nom de famille ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: len(x.strip()) > 1,
                'error_message': "Pourriez-vous me donner votre nom de famille ?"
            },
            {
                'field': 'email',
                'question': "Merci {first_name}. Pour pouvoir vous envoyer des informations détaillées, quelle est votre adresse email ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: '@' in x and '.' in x.split('@')[1],
                'error_message': "Cette adresse email ne semble pas valide. Pourriez-vous la vérifier ?"
            },
            {
                'field': 'phone',
                'question': "Parfait. Quel est votre numéro de téléphone pour un échange plus personnalisé ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: x.replace(' ', '').isdigit() and len(x.replace(' ', '')) == 10,
                'error_message': "Ce numéro ne semble pas valide. Pourriez-vous me donner un numéro à 10 chiffres ?"
            },
            {
                'field': 'age',
                'question': "Pour adapter au mieux mes conseils, quel âge avez-vous ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: x.isdigit() and 18 <= int(x) <= 100,
                'error_message': "Pourriez-vous me donner votre âge en chiffres ?"
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
                ]
            },
            {
                'field': 'income',
                'question': "Dans quelle tranche de revenus annuels vous situez-vous ?",
                'required': True,
                'type': 'choice',
                'options': [
                    "Moins de 30 000€",
                    "30 000€ - 50 000€",
                    "50 000€ - 100 000€",
                    "Plus de 100 000€"
                ]
            },
            {
                'field': 'patrimoine',
                'question': "Quel est le montant approximatif de votre patrimoine actuel ?",
                'required': True,
                'type': 'choice',
                'options': [
                    "Moins de 50 000€",
                    "50 000€ - 200 000€",
                    "200 000€ - 500 000€",
                    "Plus de 500 000€"
                ]
            }
        ]

    def get_next_info(self, collected_info: dict) -> tuple:
        for info in self.info_sequence:
            if info['field'] not in collected_info or (
                info.get('required', True) and not collected_info[info['field']]
            ):
                question = info.get('question', '')
                if '{first_name}' in question and 'first_name' in collected_info:  # Changé de 'name' à 'first_name'
                    first_name = collected_info['first_name']  # Utilise directement first_name
                    question = question.format(first_name=first_name)
                return info['field'], {
                    'question': question,
                    'type': info.get('type', 'text'),
                    'options': info.get('options', [])
                }
        return None, None

    def is_collection_complete(self, collected_info: dict) -> bool:
        """Vérifie si toutes les informations requises ont été collectées"""
        return all(
            info['field'] in collected_info and 
            (not info.get('required', True) or collected_info[info['field']])
            for info in self.info_sequence 
        )

    def validate_input(self, field: str, value: str, collected_info: dict) -> tuple:
        """Valide une entrée utilisateur"""
        for info in self.info_sequence:
            if info['field'] == field:
                if field == 'age':
                    try:
                        # Nettoyer la valeur pour extraire juste le nombre
                        age_value = ''.join(filter(str.isdigit, str(value)))
                        if age_value and 18 <= int(age_value) <= 100:
                            return True, None
                        return False, "Pourriez-vous me donner votre âge (entre 18 et 100 ans) ?"
                    except Exception:
                        return False, "Pourriez-vous me donner votre âge en chiffres ?"
                elif info.get('validator'):
                    try:
                        is_valid = info['validator'](value)
                        return is_valid, info.get('error_message') if not is_valid else None
                    except Exception:
                        return False, info.get('error_message')
                elif info.get('type') == 'choice':
                    return value in info['options'], "Veuillez choisir une des options proposées."
        return True, None

    def is_collection_complete(self, collected_info: dict) -> bool:
        """Vérifie si toutes les informations requises ont été collectées"""
        return all(
            info['field'] in collected_info and collected_info[info['field']]
            for info in self.info_sequence 
            if info['required']
        )
        
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

    async def extract_info_from_message(self, message: str) -> dict:
        try:
            # Vérifier d'abord si c'est un nom
            words = message.strip().split()
            if len(words) == 2:
                return {
                    'first_name': words[0].strip().capitalize(),
                    'last_name': words[1].strip().capitalize()
                }
            elif len(words) == 1 and not any(char.isdigit() for char in words[0]):
                return {'first_name': words[0].strip().capitalize()}
    
            # Vérifier l'âge
            age_match = re.search(r'\b(\d+)(?:\s*(?:ans?))?\b', message)
            if age_match and not '@' in message and not any(c.isalpha() for c in message.replace('ans', '')):
                age = age_match.group(1)
                if 18 <= int(age) <= 100:
                    return {'age': age}
    
            # Vérifier l'email
            if '@' in message and '.' in message:
                email = message.strip()
                if '@' in email and '.' in email.split('@')[1]:
                    return {'email': email.lower()}
    
            # Vérifier le téléphone
            phone = ''.join(filter(str.isdigit, message))
            if len(phone) == 10 and phone.isdigit():
                return {'phone': phone}
    
            # Pour les autres cas, utiliser GPT
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extrais uniquement les informations explicitement mentionnées."},
                    {"role": "user", "content": f"""Extrait en JSON :
                        - profession: métier exact mentionné
                        - income: tranche de revenus
                        - patrimoine: montant patrimoine
                        Message: {message}"""}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
    
            return json.loads(response.choices[0].message.content)
    
        except Exception as e:
            print(f"Error in extract_info_from_message: {str(e)}")
            return {}

    async def generate_response(self, message: str, field: str, next_info: dict, collected_info: dict) -> dict:
        try:
            name = collected_info.get('name', '').split()[0] if collected_info.get('name') else ''
            initial_query = collected_info.get('initial_query', '')
            
            system_prompt = """Tu es Emma, conseillère patrimoniale. Réponds de façon concise et naturelle.
            Informe toujours que tu as besoin d'informations avant de répondre aux questions techniques."""
    
            user_prompt = f"""Question initiale : {initial_query}
            Message reçu : {message}
            Prénom client : {name}
            Question suivante : {next_info['question']}
            
            Réponds brièvement :
            1. Accuse réception si pertinent
            2. Pose la question suivante"""
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,  # Limite le nombre de tokens de la réponse
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


    async def generer_analyse_finale(self, info_collected: dict) -> str:
        try:
            prompt = f"""En tant que conseillère patrimoniale, fais une analyse concise :
    
            Question initiale : {info_collected.get('initial_query')}
            Infos client : {json.dumps(info_collected, indent=2)}
    
            Format court :
            1. Résumé de la situation
            2. Réponse à la question
            3. 1-2 recommandations clés
            4. Proposition de RDV"""
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es Emma, conseillère patrimoniale. Sois concise et précise."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # Limite la longueur de l'analyse
                temperature=0.7
            )
    
            return response.choices[0].message.content
    
        except Exception as e:
            print(f"Error in generer_analyse_finale: {str(e)}")
            return "Je suis désolée, je ne peux pas générer l'analyse pour le moment."


    async def repondre_question(self, question: str, conversation_id: str) -> dict:
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            collected_info = conversation['info_collected']
    
            # Enregistrer la question initiale si c'est le premier message
            if not collected_info.get('initial_query'):
                self.conv_storage.update_info(conversation_id, {'initial_query': question})
    
            # Extraction des informations du message
            extracted_info = await self.extract_info_from_message(question)
            field, next_info = self.info_collector.get_next_info(collected_info)
    
            # Si des informations ont été extraites, les valider et mettre à jour
            if extracted_info:
                if field in extracted_info:
                    is_valid, error_message = self.info_collector.validate_input(
                        field,
                        extracted_info[field],
                        collected_info
                    )
                    if is_valid:
                        self.conv_storage.update_info(conversation_id, {field: extracted_info[field]})
                        await self.update_database(conversation_id, {field: extracted_info[field]})
                        collected_info = conversation['info_collected']  # Mettre à jour les infos collectées
                        field, next_info = self.info_collector.get_next_info(collected_info)
                    else:
                        return {
                            'type': 'text',
                            'content': error_message or "Cette réponse ne semble pas valide. Pourriez-vous réessayer ?",
                            'options': []
                        }
    
            # Vérifier si la collecte est terminée
            if self.info_collector.is_collection_complete(collected_info):
                return {
                    'type': 'text',
                    'content': await self.generer_analyse_finale(collected_info),
                    'options': []
                }
    
            # Générer la prochaine question
            return await self.generate_response(question, field, next_info, collected_info)
    
        except Exception as e:
            print(f"Error in repondre_question: {str(e)}")
            return {
                'type': 'text',
                'content': "Je suis désolée, une erreur s'est produite. Pourriez-vous reformuler votre réponse ?",
                'options': []
            }
