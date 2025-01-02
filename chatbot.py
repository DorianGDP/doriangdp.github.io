from openai import OpenAI
from supabase import create_client
import os
import json
import re
import time
from typing import Optional, Dict, Any
from datetime import datetime

class ConversationStorage:
    """Gère le stockage des conversations en mémoire"""
    def __init__(self):
        self._conversations = {}

    def get_conversation(self, conversation_id: str) -> dict:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                'state': {},
                'messages': [],
                'info_collected': {},
                'lead_id': None  # Ajout pour suivre l'ID du lead
            }
        return self._conversations[conversation_id]

    def add_message(self, conversation_id: str, message: dict):
        conv = self.get_conversation(conversation_id)
        conv['messages'].append(message)

    def update_info(self, conversation_id: str, new_info: dict):
        conv = self.get_conversation(conversation_id)
        conv['info_collected'].update(new_info)

    def set_lead_id(self, conversation_id: str, lead_id: str):
        conv = self.get_conversation(conversation_id)
        conv['lead_id'] = lead_id

class ChatBot:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.storage = ConversationStorage()
        
        # Initialisation de Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)

        # Séquence ordonnée des informations à collecter (identique à votre code)
        self.question_sequence = [
            {
                'field': 'name',
                'question': "Pour commencer, puis-je connaître votre nom et prénom ?",
                'required': True
            },
            {
                'field': 'contact',
                'question': "Pour pouvoir vous recontacter et vous proposer les meilleures solutions, pourriez-vous me donner votre email ou numéro de téléphone ?",
                'required': True
            },
            {
                'field': 'age',
                'question': "Merci. Pouvez-vous me dire quel âge avez-vous ?",
                'required': True
            },
            {
                'field': 'situation_familiale',
                'question': "Quelle est votre situation familiale (célibataire, marié(e), en couple, etc.) ?",
                'required': True
            },
            {
                'field': 'profession',
                'question': "Quelle est votre profession actuelle ?",
                'required': True
            },
            {
                'field': 'revenus',
                'question': "Pour mieux évaluer votre capacité d'épargne, pourriez-vous m'indiquer vos revenus annuels approximatifs ?",
                'required': True
            },
            {
                'field': 'patrimoine',
                'question': "Quel est le montant approximatif de votre patrimoine actuel (épargne, investissements, immobilier...) ?",
                'required': True
            },
            {
                'field': 'objectifs',
                'question': "Quels sont vos principaux objectifs patrimoniaux (épargne, investissement immobilier, préparation retraite, transmission...) ?",
                'required': True
            }
        ]

    async def extract_info_from_message(self, message: str, initial_query: str = None) -> dict:
        """Extraction améliorée des informations avec contexte"""
        try:
            system_prompt = """Tu es un expert en analyse de texte spécialisé dans l'extraction d'informations personnelles.
            Tu dois extraire avec précision les informations tout en comprenant le contexte de la conversation."""
            
            user_prompt = f"""Analyse ce message et extrait les informations au format JSON :
            - name: prénom et nom (exactement comme mentionnés)
            - email: adresse email
            - phone: numéro de téléphone
            - profession: métier
            - age: âge (nombre)
            - situation_familiale: situation familiale
            - revenus: revenus annuels (nombre uniquement)
            - patrimoine: montant du patrimoine (nombre uniquement)
            - objectifs: liste des objectifs patrimoniaux
            {f'- initial_query: "{initial_query}"' if initial_query else ''}
            
            Message: {message}
            
            Notes importantes:
            1. Extrait uniquement les informations explicitement mentionnées
            2. Pour les noms, conserve l'ordre exact (prénom nom ou nom prénom)
            3. Ne fait pas d'hypothèses sur les informations manquantes
            4. Considère les variations d'écriture (ex: "je m'appelle", "je suis", etc.)"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            extracted_info = json.loads(response.choices[0].message.content)
            
            # Post-traitement des informations extraites
            if 'name' in extracted_info and isinstance(extracted_info['name'], str):
                extracted_info['name'] = extracted_info['name'].strip()
            
            return extracted_info

        except Exception as e:
            print(f"Erreur d'extraction: {str(e)}")
            return {}

    def generate_contextual_response(self, initial_query: str, extracted_info: dict, 
                                   missing_info: list) -> str:
        """Génère une réponse contextuelle basée sur la question initiale et les informations manquantes"""
        try:
            prompt = f"""En tant que conseiller en gestion de patrimoine, génère une réponse naturelle qui:
            1. Accuse réception de la question initiale: "{initial_query}"
            2. Reconnaît les informations déjà fournies: {json.dumps(extracted_info)}
            3. Demande la prochaine information manquante: {missing_info[0] if missing_info else None}
            4. Maintient un ton professionnel mais chaleureux
            
            La réponse doit:
            - Être naturelle et conversationnelle
            - Expliquer pourquoi l'information est nécessaire
            - Éviter les formulations robotiques
            - Faire le lien avec la question initiale du client"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es Emma, une conseillère en gestion de patrimoine empathique et professionnelle."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Erreur de génération de réponse: {str(e)}")
            return "Je suis désolée, pourriez-vous reformuler votre demande ?"

    def build_acknowledgment(self, extracted_info: dict) -> str:
        """Construit un accusé de réception naturel des informations reçues"""
        acknowledgments = []
        
        if 'name' in extracted_info:
            acknowledgments.append(f"Enchantée de faire votre connaissance, {extracted_info['name']}")
        if 'email' in extracted_info:
            acknowledgments.append(f"j'ai bien noté votre email")
        if 'phone' in extracted_info:
            acknowledgments.append(f"j'ai bien noté votre numéro de téléphone")
        if 'profession' in extracted_info:
            acknowledgments.append(f"je note que vous êtes {extracted_info['profession']}")
        if 'age' in extracted_info:
            acknowledgments.append(f"vous avez {extracted_info['age']} ans")
        if 'situation_familiale' in extracted_info:
            acknowledgments.append(f"vous êtes {extracted_info['situation_familiale']}")
        if 'revenus' in extracted_info:
            acknowledgments.append(f"avec des revenus annuels de {extracted_info['revenus']}€")
        if 'patrimoine' in extracted_info:
            acknowledgments.append(f"et un patrimoine d'environ {extracted_info['patrimoine']}€")
        if 'objectifs' in extracted_info:
            objectifs = extracted_info['objectifs']
            if isinstance(objectifs, list):
                objectifs_str = ", ".join(objectifs)
                acknowledgments.append(f"vos objectifs sont : {objectifs_str}")
        
        if acknowledgments:
            response = ". ".join(acknowledgments) + "."
            return response[0].upper() + response[1:]

    async def update_supabase_lead(self, lead_id: str, info: dict):
        try:
            lead_data = {
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            if 'name' in info:
                name_parts = info['name'].split()
                lead_data["first_name"] = name_parts[0]
                lead_data["last_name"] = ' '.join(name_parts[1:]) if len(name_parts) > 1 else None
            
            if 'email' in info:
                lead_data["email"] = info['email']
            if 'phone' in info:
                lead_data["phone"] = info['phone']

            await self.supabase.table('leads').update(lead_data).eq('id', lead_id).execute()

            patrimoine_data = {
                "updated_at": datetime.utcnow().isoformat(),
            }

            if 'objectifs' in info:
                patrimoine_data["objectifs"] = info['objectifs']
            if 'patrimoine' in info:
                patrimoine_data["patrimoine_total"] = float(info['patrimoine'])
            if 'revenus' in info:
                patrimoine_data["revenus_annuels"] = float(info['revenus'])
            if 'age' in info:
                patrimoine_data["age"] = int(info['age'])
            if 'profession' in info:
                patrimoine_data["profession"] = info['profession']
            if 'situation_familiale' in info:
                patrimoine_data["situation_familiale"] = info['situation_familiale']

            await self.supabase.table('patrimoine_info').upsert({
                "lead_id": lead_id,
                **patrimoine_data
            }).execute()

        except Exception as e:
            print(f"Erreur lors de la mise à jour Supabase : {str(e)}")
            raise

    async def create_or_update_supabase(self, conversation_id: str, info: dict) -> str:
        try:
            conv = self.storage.get_conversation(conversation_id)
            lead_id = conv.get('lead_id')

            if not lead_id:
                lead_data = {
                    "status": "nouveau",
                    "source": "chatbot",
                    "created_at": datetime.utcnow().isoformat(),
                }
                
                if 'name' in info:
                    name_parts = info['name'].split()
                    lead_data["first_name"] = name_parts[0]
                    lead_data["last_name"] = ' '.join(name_parts[1:]) if len(name_parts) > 1 else None
                
                if 'email' in info:
                    lead_data["email"] = info['email']
                if 'phone' in info:
                    lead_data["phone"] = info['phone']

                result = await self.supabase.table('leads').insert(lead_data).execute()
                lead_id = result.data[0]['id']
                self.storage.set_lead_id(conversation_id, lead_id)

                await self.supabase.table('conversations').insert({
                    "lead_id": lead_id,
                    "conversation_id": conversation_id,
                    "status": "en_cours",
                    "created_at": datetime.utcnow().isoformat()
                }).execute()

            await self.update_supabase_lead(lead_id, info)
            
            return lead_id

        except Exception as e:
            print(f"Erreur lors de la création/mise à jour Supabase : {str(e)}")
            raise

    async def save_message(self, conversation_id: str, message: dict):
        try:
            conv = self.storage.get_conversation(conversation_id)
            lead_id = conv.get('lead_id')
            
            if lead_id:
                conv_result = await self.supabase.table('conversations')\
                    .select('id')\
                    .eq('conversation_id', conversation_id)\
                    .execute()
                
                if conv_result.data:
                    await self.supabase.table('messages').insert({
                        "conversation_id": conv_result.data[0]['id'],
                        "message_type": message['role'],
                        "content": message['content'],
                        "created_at": datetime.utcnow().isoformat()
                    }).execute()

        except Exception as e:
            print(f"Erreur lors de la sauvegarde du message : {str(e)}")
    
    async def get_next_question(self, conversation: dict) -> tuple:
        info_collected = conversation.get('info_collected', {})
        
        for question_info in self.question_sequence:
            field = question_info['field']
            if field not in info_collected or not info_collected[field]:
                return field, question_info['question']
                
        return None, None

    async def get_next_response(self, conversation: dict, extracted_info: dict) -> str:
        try:
            info_collected = conversation.get('info_collected', {})
            messages_history = conversation.get('messages', [])
            
            # Si des informations ont été extraites, on les confirme d'abord
            response = ""
            if extracted_info:
                response = self.build_acknowledgment(extracted_info)
                
                # On met à jour info_collected avec les nouvelles informations
                info_collected.update(extracted_info)
                conversation['info_collected'] = info_collected
            
            # On récupère la prochaine question à poser
            next_field, next_question = await self.get_next_question(conversation)
            
            # Si toutes les informations sont collectées
            if not next_field:
                return await self.generate_final_analysis(info_collected)
            
            # Si on a collecté des infos, on ajoute la prochaine question
            if response:
                response += f"\n\n{next_question}"
            else:
                # Si on n'a pas extrait d'infos, on repose la question actuelle
                # ou on passe à la suivante si c'est le premier message
                if len(messages_history) <= 1:
                    response = next_question
                else:
                    # On retrouve la question actuelle
                    current_field = None
                    for question_info in self.question_sequence:
                        if question_info['field'] not in info_collected:
                            current_field = question_info
                            break
                    
                    # On reformule gentiment la demande
                    if current_field:
                        response = f"Je n'ai pas bien saisi votre réponse. {current_field['question']}"
                    else:
                        response = next_question
            
            return response
            
        except Exception as e:
            print(f"Erreur dans get_next_response: {str(e)}")
            raise

    async def generate_final_analysis(self, info_collected: dict) -> str:
        try:
            prompt = f"""
            En tant que conseillère en gestion de patrimoine, génère une analyse personnalisée
            basée sur ces informations :
            
            Nom: {info_collected.get('name')}
            Âge: {info_collected.get('age')}
            Profession: {info_collected.get('profession')}
            Situation familiale: {info_collected.get('situation_familiale')}
            Revenus annuels: {info_collected.get('revenus')}€
            Patrimoine: {info_collected.get('patrimoine')}€
            Objectifs: {info_collected.get('objectifs')}
            
            L'analyse doit :
            1. Être personnalisée et mentionner le nom du client
            2. Résumer brièvement sa situation
            3. Proposer 2-3 pistes d'optimisation patrimoniale
            4. Se terminer par une proposition de rendez-vous personnalisé
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": "Tu es Emma, une conseillère en gestion de patrimoine expérimentée et empathique."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Erreur dans generate_final_analysis: {str(e)}")
            return "Je suis désolée, je rencontre des difficultés pour générer l'analyse finale. Pouvons-nous reprendre notre conversation ?"

    async def repondre_question(self, question: str, conversation_id: str) -> dict:
        """Traitement principal des questions avec gestion du contexte"""
        try:
            conversation = self.get_conversation(conversation_id)
            initial_query = conversation.get('initial_query', question)
            
            # Extraction des informations
            extracted_info = await self.extract_info_from_message(
                question, 
                initial_query if initial_query != question else None
            )
            
            # Mise à jour de la base de données
            if extracted_info:
                await self.update_database(conversation_id, extracted_info)
            
            # Génération de la réponse contextuelle
            missing_info = self.get_missing_info(conversation_id)
            response = self.generate_contextual_response(
                initial_query,
                extracted_info,
                missing_info
            )
            
            return {
                'reponse': response,
                'conversation_id': conversation_id,
                'type': 'text'
            }
            
        except Exception as e:
            print(f"Erreur: {str(e)}")
            return {
                'reponse': "Je suis désolée, je rencontre une difficulté technique. Pouvez-vous réessayer ?",
                'conversation_id': conversation_id,
                'type': 'text',
                'error': True
            }
