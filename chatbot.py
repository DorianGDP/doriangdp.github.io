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
                'required': True,
                'question': "Pour vous conseiller au mieux, quel est votre nom et prénom ?"
            },
            {
                'field': 'contact',
                'required': True,
                'question': "Merci de me communiquer votre email ou numéro de téléphone pour être recontacté",
                'type': 'text'
            },
            {
                'field': 'revenus',
                'required': True,
                'question': "Dans quelle tranche de revenus annuels vous situez-vous ?",
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
                'required': True,
                'question': "Quel est le montant approximatif de votre patrimoine ?",
                'type': 'choice',
                'options': [
                    "Moins de 50 000€",
                    "50 000€ - 200 000€",
                    "200 000€ - 500 000€",
                    "Plus de 500 000€"
                ]
            }
        ]

    def get_next_question(self, collected_info: dict, initial_query: str) -> tuple:
        """Récupère la prochaine question à poser"""
        for info in self.info_sequence:
            field = info['field']
            if field not in collected_info or not collected_info[field]:
                return field, {
                    'question': info['question'],
                    'type': info.get('type', 'text'),
                    'options': info.get('options', [])
                }
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

            Message à analyser: {message}

            Exemple de réponse si seul le nom est présent:
            {{"name": "Jean Dupont"}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={ "type": "json_object" }  # Force JSON response
            )

            extracted_info = json.loads(response.choices[0].message.content)
            return {k: v.strip() if isinstance(v, str) else v 
                   for k, v in extracted_info.items() 
                   if v is not None and v != ""}

        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON: {str(e)}")
            return {}
        except Exception as e:
            print(f"Erreur d'extraction: {str(e)}")
            return {}

    async def generate_response(self, message: str, collected_info: dict, next_question: Optional[str], initial_query: Optional[str]) -> str:
        """Génère une réponse contextuelle en tenant compte du message de l'utilisateur"""
        try:
            # Détermine si c'est la première interaction
            conversation = self.conv_storage.get_conversation(self.current_conversation_id)
            is_first_interaction = len(conversation['messages']) == 0
            
            # Analyse l'intention du message
            intent_prompt = f'''Analyse ce message et détermine l'intention principale:
            Message: {message}
            
            Retourne une des catégories suivantes:
            - salutation_simple: simple bonjour sans question
            - question_patrimoine: question sur la gestion de patrimoine
            - reponse_info: réponse à une demande d'information
            - autre: autre type de message
            
            Réponds uniquement avec la catégorie, sans autre texte.'''
            
            intent_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu es un expert en analyse d'intentions."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.1
            )
            
            intent = intent_response.choices[0].message.content.strip().lower()
            
            # Génère une réponse appropriée selon le contexte
            if is_first_interaction:
                if intent == "salutation_simple":
                    return """Bonjour ! Je suis Emma, votre conseillère en gestion de patrimoine. 
                    Je suis là pour vous aider à optimiser votre patrimoine et répondre à vos questions. 
                    Quel aspect de votre gestion patrimoniale vous intéresse particulièrement ?"""
                elif intent == "question_patrimoine":
                    return f"""Bonjour ! Je suis Emma, votre conseillère en gestion de patrimoine. 
                    Je vois que vous vous intéressez à {initial_query or 'la gestion de patrimoine'}. 
                    Pour vous apporter les meilleures recommandations, j'aimerais en savoir un peu plus 
                    sur vous. Tout d'abord, comment dois-je vous appeler ?"""
            
            # Pour les interactions suivantes
            prompt = f'''En tant que conseillère en gestion de patrimoine, génère une réponse naturelle qui:
            1. Prend en compte le message actuel: {message!r}
            2. Fait référence à la question initiale si pertinent: {initial_query!r}
            3. Accuse réception des informations si présentes
            4. Pose la prochaine question de manière naturelle: {next_question!r}
            5. Maintient un ton professionnel et chaleureux
            
            Informations déjà collectées:
            {json.dumps(collected_info, indent=2)}'''
            
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
    
    def get_conversation_messages(self) -> List[Dict]:
        """Récupère l'historique des messages de la conversation actuelle"""
        return self._conversations.get(self._current_conversation_id, {}).get('messages', [])

    async def update_database(self, conversation_id: str, info: dict):
        """Met à jour la base de données avec les nouvelles informations"""
        try:
            # Récupère ou crée un nouvel enregistrement lead
            conversation = self.conv_storage.get_conversation(conversation_id)
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
                lead_response = self.supabase.table('leads').insert(lead_data).execute()
                lead_id = lead_response.data[0]['id']
                conversation['lead_id'] = lead_id
    
                # Création de l'enregistrement conversation
                self.supabase.table('conversations').insert({
                    "lead_id": lead_id,
                    "conversation_id": conversation_id,
                    "status": "en_cours"
                }).execute()
    
            # Mise à jour des informations patrimoniales
            if any(key in info for key in ['age', 'profession', 'situation_familiale', 'revenus', 'patrimoine']):
                patrimoine_data = {
                    "lead_id": lead_id,
                    "updated_at": datetime.utcnow().isoformat()
                }
    
                if 'age' in info:
                    try:
                        patrimoine_data["age"] = int(info['age'])
                    except (ValueError, TypeError):
                        print(f"Erreur de conversion d'âge: {info['age']}")
    
                if 'profession' in info:
                    patrimoine_data["profession"] = info['profession']
                if 'situation_familiale' in info:
                    patrimoine_data["situation_familiale"] = info['situation_familiale']
                
                if 'revenus' in info:
                    try:
                        patrimoine_data["revenus_annuels"] = float(info['revenus'])
                    except (ValueError, TypeError):
                        print(f"Erreur de conversion des revenus: {info['revenus']}")
                
                if 'patrimoine' in info:
                    try:
                        patrimoine_data["patrimoine_total"] = float(info['patrimoine'])
                    except (ValueError, TypeError):
                        print(f"Erreur de conversion du patrimoine: {info['patrimoine']}")
    
                if patrimoine_data:
                    self.supabase.table('patrimoine_info').upsert(patrimoine_data).execute()
    
        except Exception as e:
            print(f"Erreur de mise à jour de la base de données: {str(e)}")

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

            response = self.client.chat.completions.create(
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
            # Stocke l'ID de conversation actuel pour utilisation dans d'autres méthodes
            self.current_conversation_id = conversation_id
            
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
                response = await self.generate_response(question, collected_info, next_question, initial_query)

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
