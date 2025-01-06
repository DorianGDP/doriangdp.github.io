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
    """Gère la séquence de collecte d'informations"""
    def __init__(self):
        self.info_sequence = [
            {
                'field': 'name',
                'required': True,
                'questions': [
                    "Pour mieux vous accompagner dans votre projet {}, puis-je avoir votre nom ?",
                    "Pour personnaliser mes conseils concernant {}, comment dois-je vous appeler ?",
                    "Afin de vous proposer les meilleures solutions pour {}, quel est votre nom ?"
                ]
            },
            {
                'field': 'contact',
                'required': True,
                'questions': [
                    "Pour pouvoir vous recontacter avec des informations détaillées sur {}, quel est votre email ou téléphone ?",
                    "Afin d'approfondir notre discussion sur {}, quelle est la meilleure façon de vous joindre ?",
                    "Pour vous envoyer une analyse personnalisée concernant {}, comment puis-je vous contacter ?"
                ]
            },
            {
                'field': 'age',
                'required': True,
                'questions': [
                    "L'âge est un facteur important pour optimiser {}. Quel âge avez-vous ?",
                    "Pour adapter au mieux la stratégie concernant {}, pouvez-vous me dire votre âge ?",
                    "Votre âge nous permettra de mieux personnaliser les solutions pour {}. Quel est-il ?"
                ]
            },
            {
                'field': 'situation_familiale',
                'required': True,
                'questions': [
                    "Votre situation familiale peut influencer les choix concernant {}. Êtes-vous marié(e), en couple, célibataire ?",
                    "Pour optimiser {} en fonction de votre situation, êtes-vous en couple ou célibataire ?",
                    "Quelle est votre situation familiale ? Cela nous aidera à mieux adapter les solutions pour {}"
                ]
            },
            {
                'field': 'profession',
                'required': True,
                'questions': [
                    "Votre profession peut ouvrir des opportunités spécifiques pour {}. Que faites-vous dans la vie ?",
                    "Pour identifier les meilleures options concernant {}, quelle est votre profession ?",
                    "Certaines solutions pour {} dépendent de votre activité professionnelle. Que faites-vous ?"
                ]
            },
            {
                'field': 'revenus',
                'required': True,
                'questions': [
                    "Pour évaluer les possibilités concernant {}, dans quelle tranche de revenus annuels vous situez-vous ?",
                    "Afin d'optimiser {} en fonction de vos moyens, quels sont vos revenus annuels approximatifs ?",
                    "Pour vous proposer des solutions adaptées pour {}, quel est votre niveau de revenus ?"
                ]
            },
            {
                'field': 'patrimoine',
                'required': True,
                'questions': [
                    "Le patrimoine actuel est important pour optimiser {}. Quel est le montant approximatif de votre patrimoine ?",
                    "Pour une stratégie efficace concernant {}, pouvez-vous m'indiquer votre patrimoine global ?",
                    "Afin d'adapter nos recommandations pour {}, quel est votre patrimoine actuel ?"
                ]
            }
        ]

    def get_next_question(self, collected_info: dict, initial_query: str) -> tuple:
        """Récupère la prochaine question à poser"""
        for info in self.info_sequence:
            field = info['field']
            if field not in collected_info or not collected_info[field]:
                question = random.choice(info['questions']).format(initial_query or "votre projet patrimonial")
                return field, question
        return None, None

    def is_collection_complete(self, collected_info: dict) -> bool:
        """Vérifie si toutes les informations requises ont été collectées"""
        return all(
            info['field'] in collected_info 
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
            sont présentes dans le message."""

            user_prompt = f"""Analyse ce message et extrait uniquement les informations explicitement 
            mentionnées au format JSON :
            - name: prénom et nom (exactement comme mentionnés)
            - email: adresse email
            - phone: numéro de téléphone
            - age: âge (nombre uniquement)
            - profession: métier actuel
            - situation_familiale: situation familiale
            - revenus: revenus annuels (nombre uniquement)
            - patrimoine: montant du patrimoine (nombre uniquement)

            Message à analyser: {message}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            extracted_info = json.loads(response.choices[0].message.content)
            return {k: v.strip() if isinstance(v, str) else v 
                   for k, v in extracted_info.items()}

        except Exception as e:
            print(f"Erreur d'extraction: {str(e)}")
            return {}

    def generate_response(self, collected_info: dict, next_question: str, initial_query: str) -> str:
        """Génère une réponse contextuelle"""
        try:
            prompt = f"""En tant que conseillère en gestion de patrimoine, génère une réponse naturelle qui:
            1. Si c'est la première interaction et qu'il y a une question initiale ("{initial_query}"), 
               commence par y faire référence
            2. Si des informations ont été collectées, fait un bref accusé de réception
            3. Pose la question suivante: "{next_question}"
            4. Maintient un ton professionnel mais chaleureux

            Informations déjà collectées:
            {json.dumps(collected_info, indent=2)}

            La réponse doit:
            - Être naturelle et conversationnelle
            - Expliquer pourquoi l'information est nécessaire
            - Faire le lien avec le projet du client
            - Éviter les formulations robotiques"""

            response = await self.client.chat.completions.create(
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

    async def update_database(self, conversation_id: str, info: dict):
            """Met à jour la base de données avec les nouvelles informations"""
            try:
                # Récupère ou crée un nouvel enregistrement lead
                conversation = self.storage.get_conversation(conversation_id)
                lead_id = conversation.get('lead_id')
    
                if not lead_id:
                    # Création d'un nouveau lead
                    lead_data = {
                        "status": "nouveau",
                        "source": "chatbot",
                        "created_at": datetime.utcnow().isoformat()
                    }
    
                    # Ajout des informations de base si disponibles
                    if 'name' in info:
                        name_parts = info['name'].split()
                        lead_data["first_name"] = name_parts[0]
                        if len(name_parts) > 1:
                            lead_data["last_name"] = ' '.join(name_parts[1:])
    
                    if 'email' in info:
                        lead_data["email"] = info['email']
                    if 'phone' in info:
                        lead_data["phone"] = info['phone']
    
                    # Insertion du nouveau lead
                    result = await self.supabase.table('leads').insert(lead_data).execute()
                    lead_id = result.data[0]['id']
                    conversation['lead_id'] = lead_id
    
                    # Création de l'enregistrement conversation
                    await self.supabase.table('conversations').insert({
                        "lead_id": lead_id,
                        "conversation_id": conversation_id,
                        "status": "en_cours"
                    }).execute()
    
                # Mise à jour des informations patrimoniales
                patrimoine_data = {
                    "lead_id": lead_id,
                    "updated_at": datetime.utcnow().isoformat()
                }
    
                if 'age' in info:
                    patrimoine_data["age"] = int(info['age'])
                if 'profession' in info:
                    patrimoine_data["profession"] = info['profession']
                if 'situation_familiale' in info:
                    patrimoine_data["situation_familiale"] = info['situation_familiale']
                if 'revenus' in info:
                    patrimoine_data["revenus_annuels"] = float(info['revenus'])
                if 'patrimoine' in info:
                    patrimoine_data["patrimoine_total"] = float(info['patrimoine'])
    
                if patrimoine_data:
                    await self.supabase.table('patrimoine_info').upsert(patrimoine_data).execute()
    
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
            5. Se terminer par une proposition de rendez-vous personnalisé
            
            Garde un ton professionnel mais chaleureux et évite les formulations génériques."""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
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
        """Traite la question et génère une réponse appropriée"""
        try:
            # Récupère ou crée la conversation
            conversation = self.conv_storage.get_conversation(conversation_id)
            
            # Si c'est la première question, l'enregistre comme query initiale
            self.conv_storage.set_initial_query(conversation_id, question)
            initial_query = conversation.get('initial_query')

            # Extrait les informations du message
            extracted_info = await self.extract_info_from_message(question)
            
            # Met à jour les informations collectées
            if extracted_info:
                self.conv_storage.update_info(conversation_id, extracted_info)
                await self.update_database(conversation_id, extracted_info)

            collected_info = self.conv_storage.get_collected_info(conversation_id)

            # Vérifie si toutes les informations nécessaires ont été collectées
            if self.info_collector.is_collection_complete(collected_info):
                # Génère une analyse finale
                response = await self.generer_analyse_finale(collected_info, initial_query)
            else:
                # Obtient la prochaine question à poser
                _, next_question = self.info_collector.get_next_question(collected_info, initial_query)
                
                # Génère une réponse contextuelle
                response = await self.generate_response(collected_info, next_question, initial_query)

            # Enregistre le message dans l'historique
            self.conv_storage.add_message(conversation_id, {
                'role': 'user',
                'content': question
            })
            self.conv_storage.add_message(conversation_id, {
                'role': 'assistant',
                'content': response
            })

            return {
                'reponse': response,
                'conversation_id': conversation_id,
                'type': 'text'
            }

        except Exception as e:
            print(f"Erreur dans repondre_question: {str(e)}")
            return {
                'reponse': "Je suis désolée, je rencontre une difficulté technique. Pouvez-vous réessayer ?",
                'conversation_id': conversation_id,
                'type': 'text',
                'error': True
            }
