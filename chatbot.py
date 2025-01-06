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
                'field': 'name',
                'question': "Pour mieux vous conseiller, quel est votre nom et prénom ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: len(x.split()) >= 2
            },
            {
                'field': 'email',
                'question': "À quelle adresse email puis-je vous recontacter ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: '@' in x and '.' in x.split('@')[1]
            },
            {
                'field': 'phone',
                'question': "Quel est votre numéro de téléphone pour un échange plus personnalisé ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: x.replace(' ', '').isdigit() and len(x.replace(' ', '')) == 10
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
                'field': 'age',
                'question': "Quel âge avez-vous ?",
                'required': True,
                'type': 'text',
                'validator': lambda x: x.isdigit() and 18 <= int(x) <= 100
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
            },
            {
                'field': 'objectifs',
                'question': "Quel est votre principal objectif patrimonial ?",
                'required': True,
                'type': 'choice',
                'options': [
                    "Préparer ma retraite",
                    "Optimiser ma fiscalité",
                    "Investir dans l'immobilier",
                    "Protéger mes proches"
                ]
            }
        ]

    def get_next_question(self, collected_info: dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Récupère la prochaine question à poser"""
        for info in self.info_sequence:
            field = info['field']
            if field not in collected_info or not collected_info[field]:
                return field, {
                    'question': info['question'],
                    'type': info.get('type', 'text'),
                    'options': info.get('options', []),
                    'expectedInfo': field
                }
        return None, None

    def validate_input(self, field: str, value: str) -> bool:
        """Valide une entrée utilisateur pour un champ donné"""
        for info in self.info_sequence:
            if info['field'] == field:
                if 'validator' in info:
                    return info['validator'](value)
                elif info['type'] == 'choice':
                    return value in info['options']
                return True
        return True

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
        """Extrait les informations du message de l'utilisateur"""
        try:
            system_prompt = """Tu es un expert en analyse de texte spécialisé dans l'extraction 
            d'informations personnelles. Extrait précisément les informations suivantes si elles 
            sont présentes dans le message. Si une information n'est pas présente, ne pas l'inclure 
            dans le JSON."""

            user_prompt = f"""Analyse ce message et extrait uniquement les informations explicitement 
            mentionnées dans un format JSON valide. Inclure uniquement les champs avec des informations :
            - name: prénom et nom (exactement comme mentionnés)
            - email: adresse email
            - phone: numéro de téléphone
            - age: âge (nombre uniquement)
            - profession: métier actuel
            - situation_familiale: situation familiale
            - revenus: revenus annuels (nombre uniquement)
            - patrimoine: montant du patrimoine (nombre uniquement)

            Message à analyser: {message}"""

            response = self.client.chat.completions.create(  # Retiré le await
                model="gpt-4o",  # Corrigé le nom du modèle
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={ "type": "json_object" }
            )

            extracted_info = json.loads(response.choices[0].message.content)
            return {k: v.strip() if isinstance(v, str) else v 
                   for k, v in extracted_info.items() 
                   if v is not None and v != ""}

        except Exception as e:
            print(f"Erreur d'extraction: {str(e)}")
            return {}

    async def generate_response(self, message: str, collected_info: dict, next_question: Optional[dict], initial_query: Optional[str]) -> dict:
        try:
            response = {
                'type': 'text',
                'content': '',
                'options': []
            }
    
            # Si c'est le premier message, on commence toujours par demander le nom
            if not collected_info.get('name'):
                return {
                    'type': 'text',
                    'content': "Bonjour ! Je suis Emma, votre conseillère en gestion de patrimoine. Pour mieux vous accompagner dans votre projet, pourriez-vous me donner votre nom et prénom ?",
                    'options': []
                }
    
            # Vérifier si on a une question suivante à poser
            if next_question:
                system_prompt = """Tu es Emma, une conseillère patrimoniale professionnelle. 
                Tu dois répondre de manière naturelle et empathique, en expliquant que tu as besoin 
                d'informations supplémentaires pour mieux conseiller la personne."""
    
                name = collected_info.get('name', '').split()[0] if collected_info.get('name') else ''
                
                user_prompt = f"""En tenant compte de ces éléments :
                - Message reçu : {message}
                - Prénom du client : {name}
                - Information à obtenir : {next_question['question']}
                
                Génère une réponse qui :
                1. Accusé réception de sa réponse précédente si pertinent
                2. Explique naturellement que tu as besoin d'une information supplémentaire
                3. Pose la question suivante : {next_question['question']}"""
    
                chat_completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7
                )
                
                response['content'] = chat_completion.choices[0].message.content
    
                if next_question.get('type') == 'choice':
                    response['type'] = 'choice'
                    response['options'] = next_question.get('options', [])
    
                return response
            
            # Si toutes les informations sont collectées, générer l'analyse finale
            return {
                'type': 'text',
                'content': await self.generer_analyse_finale(collected_info, initial_query),
                'options': []
            }
    
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return {
                'type': 'text',
                'content': "Je suis désolée, pourriez-vous reformuler votre réponse ?",
                'options': []
            }
    
    def get_conversation_messages(self) -> List[Dict]:
        """Récupère l'historique des messages de la conversation actuelle"""
        return self._conversations.get(self._current_conversation_id, {}).get('messages', [])

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


    async def generer_analyse_finale(self, info_collected: dict, initial_query: str) -> str:
        """Génère une analyse finale basée sur toutes les informations collectées"""
        try:
            prompt = f"""En tant que conseillère en gestion de patrimoine, génère une analyse personnalisée 
            et détaillée basée sur ces informations :

            Question initiale: {initial_query}
            Informations collectées:
            {json.dumps(info_collected, indent=2)}

            L'analyse doit :
            1. Commencer par un résumé personnalisé de la situation
            2. Répondre spécifiquement à la question/demande initiale
            3. Proposer 2-3 recommandations pertinentes
            4. Expliquer les avantages de chaque recommandation
            5. Se terminer par une proposition de rendez-vous personnalisé"""

            response = self.client.chat.completions.create(  # Retiré le await
                model="gpt-4o",  # Corrigé le nom du modèle
                messages=[
                    {"role": "system", "content": "Tu es Emma, une conseillère en gestion de patrimoine expérimentée et empathique."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Erreur dans la génération de l'analyse finale: {str(e)}")
            return "Je suis désolée, je rencontre des difficultés pour générer l'analyse finale. Pouvons-nous reprendre notre conversation ?"

    async def repondre_question(self, question: str, conversation_id: str) -> dict:
        try:
            conversation = self.conv_storage.get_conversation(conversation_id)
            collected_info = self.conv_storage.get_collected_info(conversation_id)
            
            # Si c'est le premier message
            if not conversation.get('initial_query'):
                self.conv_storage.set_initial_query(conversation_id, question)
                return await self.generate_response(question, collected_info, {'question': ''}, None)
                
            extracted_info = await self.extract_info_from_message(question)
            if extracted_info:
                self.conv_storage.update_info(conversation_id, extracted_info)
                await self.update_database(conversation_id, extracted_info)
    
            field, next_question = self.info_collector.get_next_question(collected_info)
            
            if self.info_collector.is_collection_complete(collected_info):
                return {
                    'type': 'text',
                    'content': await self.generer_analyse_finale(collected_info, conversation.get('initial_query')),
                    'options': []
                }
            
            return await self.generate_response(
                question, 
                collected_info, 
                next_question,
                conversation.get('initial_query')
            )
    
        except Exception as e:
            print(f"Error in repondre_question: {str(e)}")
            return {
                'type': 'text',
                'content': "Je suis désolée, pourriez-vous reformuler votre réponse ?",
                'options': []
            }
