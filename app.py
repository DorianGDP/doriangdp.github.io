from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import asyncio
import traceback
import json
import re
from datetime import datetime
from supabase import create_client, Client

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://doriangdp.github.io"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# Initialisation de Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

class ConversationManager:
    def __init__(self):
        self.conversations = {}

    def get_conversation(self, conversation_id):
        return self.conversations.get(conversation_id, {'step': 0, 'info': {}})

    def update_conversation(self, conversation_id, info):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {'step': 0, 'info': {}}
        self.conversations[conversation_id]['info'].update(info)
        print(f"Conversation mise à jour - ID: {conversation_id}, Info: {self.conversations[conversation_id]}")
        return self.conversations[conversation_id]

    def get_info(self, conversation_id):
        return self.conversations.get(conversation_id, {'info': {}})['info']

# Instance globale du gestionnaire de conversations
conversation_manager = ConversationManager()

async def save_to_supabase(info, conversation_id):
    try:
        # Créer ou mettre à jour le lead
        lead_data = {
            "first_name": info.get('name', '').split()[0] if info.get('name') else None,
            "last_name": ' '.join(info.get('name', '').split()[1:]) if info.get('name') else None,
            "email": info.get('email'),
            "phone": info.get('phone'),
            "status": "nouveau"
        }
        
        lead_response = await supabase.table('leads').upsert(lead_data).execute()
        if lead_response.data:
            lead_id = lead_response.data[0]['id']
            
            if info.get('patrimoine'):
                patrimoine_data = {
                    "lead_id": lead_id,
                    "patrimoine_total": float(info['patrimoine']),
                    "revenus_annuels": float(info.get('revenus', 0))
                }
                await supabase.table('patrimoine_info').upsert(patrimoine_data).execute()
            
            conversation_data = {
                "lead_id": lead_id,
                "conversation_id": conversation_id,
                "status": "en_cours"
            }
            await supabase.table('conversations').upsert(conversation_data).execute()
            
            return True
    except Exception as e:
        print(f"Erreur Supabase: {str(e)}")
        return False

def analyze_message(message, current_info):
    """Analyse le message pour en extraire les informations pertinentes"""
    info = {}
    
    # Extraction du nom - maintenant plus flexible
    if not current_info.get('name'):
        # Première tentative : patterns spécifiques
        name_patterns = [
            r"je m'appelle ([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
            r"mon nom est ([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
        ]
        for pattern in name_patterns:
            if match := re.search(pattern, message.lower()):
                info['name'] = match.group(1).title()
                print(f"Nom trouvé (pattern): {info['name']}")
                break
                
        # Si aucun pattern ne correspond, considérer le message entier comme nom potentiel
        if not info.get('name') and re.match(r'^[a-zA-Z\s]+$', message.strip()):
            info['name'] = message.strip().title()
            print(f"Nom trouvé (message entier): {info['name']}")

    # Extraction email
    if not current_info.get('email'):
        if email_match := re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', message):
            info['email'] = email_match.group()
            print(f"Email trouvé: {info['email']}")

    # Extraction patrimoine
    if not current_info.get('patrimoine'):
        patrimoine_patterns = [
            r'(\d+(?:\s*\d*)*(?:\s*[kKmM€]?))\s*(?:€|euros?)',
            r'(\d+(?:\s*\d*)*)\s*(?:k|K|mille)',
            r'(\d+(?:\s*\d*)*)\s*(?:M|million)'
        ]
        for pattern in patrimoine_patterns:
            if match := re.search(pattern, message):
                value = parse_amount(match.group(1))
                if value:
                    info['patrimoine'] = value
                    print(f"Patrimoine trouvé: {info['patrimoine']}")
                    break

    print(f"Informations extraites: {info}")
    return info

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données manquantes'}), 400

        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id', '')

        if not conversation_id:
            conversation_id = f"conv_{int(datetime.now().timestamp())}"
            print(f"Nouvelle conversation créée: {conversation_id}")
        else:
            print(f"Conversation existante: {conversation_id}")

        if not question:
            return jsonify({'error': 'Question manquante'}), 400

        # Récupérer la conversation existante ou en créer une nouvelle
        conversation = conversation_manager.get_conversation(conversation_id)
        print(f"État actuel de la conversation: {conversation}")
        
        # Analyser le message
        current_info = conversation.get('info', {})
        new_info = analyze_message(question, current_info)
        
        if new_info:
            conversation = conversation_manager.update_conversation(conversation_id, new_info)
            print(f"Conversation mise à jour avec les nouvelles infos: {conversation}")
            await save_to_supabase(conversation['info'], conversation_id)

        # Déterminer la prochaine question
        response = get_next_question(conversation)

        return jsonify({
            'reponse': response,
            'conversation_id': conversation_id,
            'type': 'text',
            'debug_info': {
                'conversation_state': conversation,
                'new_info': new_info
            }
        })

    except Exception as e:
        print(f"Erreur serveur: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Erreur serveur',
            'details': str(e)
        }), 500

def get_next_question(conversation):
    """Détermine la prochaine question à poser"""
    info = conversation['info']
    print(f"Génération de la prochaine question avec les infos: {info}")
    
    if not info.get('name'):
        return "Pour mieux vous accompagner, pourrais-je connaître votre nom ?"
    
    if not info.get('email'):
        return f"Merci {info['name']}! Pour pouvoir vous envoyer une analyse détaillée, quelle est votre adresse email ?"
    
    if not info.get('patrimoine'):
        return f"Pour personnaliser mes recommandations, {info['name']}, quel est approximativement votre patrimoine actuel ?"
    
    # Analyse finale si toutes les informations sont collectées
    return generate_analysis(info)

def generate_analysis(info):
    """Génère une analyse personnalisée basée sur les informations collectées"""
    analysis = f"""Merci {info['name']} pour ces informations. Voici une première analyse de votre situation :

    Basé sur votre patrimoine de {format_amount(info['patrimoine'])}€, voici mes recommandations :
    
    1. Protection et optimisation
    - Diversification de vos investissements
    - Étude de votre fiscalité
    
    2. Opportunités d'investissement
    - Immobilier locatif
    - Assurance-vie multi-supports
    
    Pour approfondir cette analyse, je peux vous mettre en relation avec l'un de nos experts. 
    Souhaitez-vous être recontacté par téléphone ?"""
    
    return analysis

def parse_amount(amount_str):
    """Convertit une chaîne de montant en nombre"""
    try:
        amount = amount_str.replace(' ', '').upper()
        if 'K' in amount:
            return float(amount.replace('K', '')) * 1000
        if 'M' in amount:
            return float(amount.replace('M', '')) * 1000000
        return float(amount)
    except:
        return None

def format_amount(amount):
    """Formate un montant pour l'affichage"""
    if amount >= 1000000:
        return f"{amount/1000000:.1f}M"
    if amount >= 1000:
        return f"{amount/1000:.0f}K"
    return f"{amount:.0f}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
