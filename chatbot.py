import openai
from supabase import create_client
import os
import json
import re
from typing import Optional, Dict, Any

class ConversationStorage:
    """Gère le stockage des conversations en mémoire"""
    def __init__(self):
        self._conversations = {}

    def get_conversation(self, conversation_id: str) -> dict:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                'state': {},
                'messages': [],
                'info_collected': {}
            }
        return self._conversations[conversation_id]

    def add_message(self, conversation_id: str, message: dict):
        conv = self.get_conversation(conversation_id)
        conv['messages'].append(message)

    def update_info(self, conversation_id: str, new_info: dict):
        conv = self.get_conversation(conversation_id)
        conv['info_collected'].update(new_info)

class ChatBot:
    """Gère l'intelligence du chatbot et les interactions avec l'utilisateur"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.storage = ConversationStorage()
        
        # Initialisation de Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)

        # Structure des informations à collecter
        self.required_info = {
            'name': 'le nom et prénom',
            'email': "l'adresse email",
            'profession': 'la profession',
            'age': "l'âge",
            'revenus': 'les revenus annuels',
            'patrimoine': 'le patrimoine total',
            'objectifs': 'les objectifs patrimoniaux'
        }

    async def extract_info_from_message(self, message: str) -> dict:
        """Utilise GPT pour extraire les informations du message"""
        try:
            prompt = f"""Analyse ce message et extrait les informations suivantes au format JSON :
            - name: prénom et/ou nom mentionnés
            - email: adresse email
            - profession: métier ou activité professionnelle
            - age: âge mentionné (en nombre)
            - revenus: montant des revenus annuels (en nombre)
            - patrimoine: montant du patrimoine (en nombre)
            - objectifs: objectifs patrimoniaux mentionnés (liste)

            Message : {message}

            Renvoie uniquement les informations explicitement mentionnées.
            Format souhaité : {{"clé": "valeur"}}
            Pour les montants, renvoie uniquement les nombres (sans € ou euros)
            """

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": "Tu es un assistant spécialisé dans l'extraction d'informations de messages."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message['content'])
        except Exception as e:
            print(f"Erreur lors de l'extraction d'informations : {str(e)}")
            return {}

    async def get_next_response(self, conversation: dict, extracted_info: dict) -> str:
        """Utilise GPT pour générer une réponse naturelle"""
        try:
            info_collected = conversation.get('info_collected', {})
            messages_history = conversation.get('messages', [])
            
            # Préparer le contexte pour GPT
            context = {
                "info_collected": info_collected,
                "missing_info": {k: v for k, v in self.required_info.items() if k not in info_collected},
                "last_messages": messages_history[-3:] if messages_history else [],
                "new_info": extracted_info
            }

            prompt = f"""En tant que conseiller en gestion de patrimoine, génère une réponse naturelle basée sur ce contexte :
            
            Informations déjà collectées : {json.dumps(context['info_collected'], ensure_ascii=False)}
            Informations manquantes : {json.dumps(context['missing_info'], ensure_ascii=False)}
            Nouvelles informations : {json.dumps(context['new_info'], ensure_ascii=False)}
            
            Règles :
            1. Si de nouvelles informations sont fournies, accuse réception naturellement
            2. Pose UNE question pour obtenir la prochaine information manquante
            3. Reste amical et professionnel
            4. Adapte la question au contexte de la conversation
            5. Si toutes les informations sont collectées, propose une analyse
            
            Ton objectif est de collecter toutes les informations manquantes de manière naturelle et conversationnelle."""

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": "Tu es Emma, une conseillère en gestion de patrimoine expérimentée et sympathique."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7
            )
            
            return response.choices[0].message['content']
        except Exception as e:
            print(f"Erreur lors de la génération de réponse : {str(e)}")
            return "Je suis désolée, je rencontre des difficultés techniques. Pouvons-nous reprendre notre conversation?"

    async def save_to_supabase(self, conversation_id: str):
        """Sauvegarde les informations dans Supabase"""
        try:
            conv = self.storage.get_conversation(conversation_id)
            info = conv['info_collected']
            
            if not info:
                return
            
            # Créer ou mettre à jour le lead
            lead_data = {
                "first_name": info.get('name', '').split()[0] if info.get('name') else None,
                "last_name": ' '.join(info.get('name', '').split()[1:]) if info.get('name') else None,
                "email": info.get('email'),
                "phone": info.get('phone'),
                "status": "nouveau"
            }
            
            lead_response = await self.supabase.table('leads').upsert(lead_data).execute()
            if lead_response.data:
                lead_id = lead_response.data[0]['id']
                
                # Créer les informations patrimoniales
                patrimoine_data = {
                    "lead_id": lead_id,
                    "objectifs": info.get('objectifs', []),
                    "patrimoine_total": float(info.get('patrimoine', 0)),
                    "revenus_annuels": float(info.get('revenus', 0)),
                    "age": info.get('age'),
                    "profession": info.get('profession')
                }
                
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

    async def repondre_question(self, question: str, conversation_id: str) -> dict:
        """Point d'entrée principal pour traiter une question"""
        try:
            # Récupérer ou créer la conversation
            conversation = self.storage.get_conversation(conversation_id)
            
            # Ajouter le message à l'historique
            self.storage.add_message(conversation_id, {
                "role": "user",
                "content": question
            })
            
            # Extraire les informations du message
            extracted_info = await self.extract_info_from_message(question)
            
            # Mettre à jour les informations collectées
            if extracted_info:
                self.storage.update_info(conversation_id, extracted_info)
                # Sauvegarder dans Supabase
                await self.save_to_supabase(conversation_id)
            
            # Générer la réponse
            response = await self.get_next_response(conversation, extracted_info)
            
            # Ajouter la réponse à l'historique
            self.storage.add_message(conversation_id, {
                "role": "assistant",
                "content": response
            })
            
            return {
                'reponse': response,
                'conversation_id': conversation_id,
                'type': 'text'
            }
            
        except Exception as e:
            print(f"Erreur dans repondre_question : {str(e)}")
            return {
                'reponse': "Désolé, une erreur s'est produite. Pouvez-vous reformuler ?",
                'conversation_id': conversation_id,
                'type': 'text'
            }
