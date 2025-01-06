from openai import OpenAI
from supabase import create_client
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import random

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
                'required': False  # Ajout de l'attribut required
            },
            {
                'field': 'name',
                'question': "Pour commencer et mieux vous conseiller, pourriez-vous me donner votre nom et prénom ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: len(x.split()) >= 2,
                'error_message': "J'ai besoin de votre nom complet pour mieux vous accompagner."
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
        """Détermine la prochaine information à collecter"""
        for info in self.info_sequence:
            if info['field'] not in collected_info or (
                info.get('required', True) and not collected_info[info['field']]
            ):
                question = info.get('question', '')
                if '{first_name}' in question and 'name' in collected_info:
                    first_name = collected_info['name'].split()[0]
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
                if info.get('validator'):
                    try:
                        is_valid = info['validator'](value)
                        return is_valid, info.get('error_message') if not is_valid else None
                    except Exception:
                        return False, info.get('error_message')
                elif info.get('type') == 'choice':
                    return value in info['options'], "Veuillez choisir une des options proposées."
                return True, None
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
            system_prompt = "Extrais les informations personnelles du message suivant."
            
            user_prompt = f"""Format JSON requis avec uniquement les informations présentes :
            - name: prénom et nom
            - email: adresse email
            - phone: numéro téléphone
            - age: âge (nombre)
            - profession: métier actuel
            - revenus: revenus annuels
            - patrimoine: montant patrimoine

            Message: {message}"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            extracted_info = json.loads(response.choices[0].message.content)
            return {k: v.strip() if isinstance(v, str) else v 
                   for k, v in extracted_info.items() 
                   if v is not None and v != ""}

        except Exception as e:
            print(f"Error in extract_info_from_message: {str(e)}")
            return {}

    async def generate_response(self, message: str, field: str, next_info: dict, collected_info: dict) -> dict:
        try:
            name = collected_info.get('name', '').split()[0] if collected_info.get('name') else ''
            initial_query = collected_info.get('initial_query', '')
            
            system_prompt = """Tu es Emma, une conseillère patrimoniale professionnelle.
            Ta mission est de collecter des informations sur le client avant de répondre à ses questions techniques.
            Réponds de manière naturelle et empathique."""
    
            user_prompt = f"""Contexte :
            - Question initiale du client : {initial_query}
            - Message actuel : {message}
            - Prénom du client : {name}
            - Prochaine information nécessaire : {next_info['question']}
            
            Génère une réponse qui :
            1. Accuse réception du message précédent si pertinent
            2. Fait référence à la question initiale pour montrer que tu ne l'as pas oubliée
            3. Explique poliment que tu as besoin d'informations supplémentaires pour répondre
            4. Pose la question suivante de manière naturelle"""
    
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
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
            
            # Gestion des leads
            if not lead_id:
                lead_data = {
                    "status": "nouveau",
                    "source": "chatbot",
                    "created_at": datetime.utcnow().isoformat()
                }
                
                if 'name' in info:
                    name_parts = info['name'].split()
                    lead_data["first_name"] = name_parts[0]
                    if len(name_parts) > 1:
                        lead_data["last_name"] = ' '.join(name_parts[1:])
                
                if 'email' in info:
                    lead_data["email"] = info['email']
                if 'phone' in info:
                    lead_data["phone"] = info['phone']
                
                lead_response = self.supabase.table('leads').insert(lead_data).execute()
                lead_id = lead_response.data[0]['id']
                conversation['lead_id'] = lead_id
                
                self.supabase.table('conversations').insert({
                    "lead_id": lead_id,
                    "conversation_id": conversation_id,
                    "status": "en_cours"
                }).execute()
            else:
                lead_update = {}
                if 'email' in info:
                    lead_update["email"] = info['email']
                if 'phone' in info:
                    lead_update["phone"] = info['phone']
                
                if lead_update:
                    self.supabase.table('leads').update(lead_update).eq('id', lead_id).execute()
    
            def convert_income(value):
                ranges = {
                    "Moins de 30 000€": 30000,
                    "30 000€ - 50 000€": 50000,
                    "50 000€ - 100 000€": 100000,
                    "Plus de 100 000€": 150000
                }
                return ranges.get(value, 0)
    
            def convert_patrimoine(value):
                ranges = {
                    "Moins de 50 000€": 50000,
                    "50 000€ - 200 000€": 200000,
                    "200 000€ - 500 000€": 500000,
                    "Plus de 500 000€": 1000000
                }
                return ranges.get(value, 0)
            
            # Mapping des champs pour patrimoine_info
            patrimoine_fields = {
                'age': ('age', int),
                'profession': ('profession', str),
                'income': ('revenus_annuels', convert_income),
                'patrimoine': ('patrimoine_total', convert_patrimoine),
                'objectifs': ('objectifs', lambda x: [x])
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
            prompt = f"""En tant que conseillère en gestion de patrimoine, analyse cette situation:

            Question initiale: {info_collected.get('initial_query')}
            Informations client:
            {json.dumps(info_collected, indent=2)}

            Structure de l'analyse:
            1. Résumé personnalisé
            2. Réponse à la question initiale
            3. 2-3 recommandations avec avantages
            4. Proposition de rendez-vous"""

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
            print(f"Error in generer_analyse_finale: {str(e)}")
            return "Je suis désolée, je ne peux pas générer l'analyse pour le moment. Pouvons-nous reprendre ?"


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
