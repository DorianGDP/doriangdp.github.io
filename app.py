from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import asyncio
import traceback
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Structure pour stocker les conversations
conversations = {}

def create_conversation_id():
    return f"conv_{int(datetime.now().timestamp())}"

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Données manquantes'
            }), 400

        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id', '').strip()

        if not conversation_id:
            conversation_id = create_conversation_id()

        if not question:
            return jsonify({
                'error': 'Question manquante'
            }), 400

        # Récupérer ou initialiser l'état de la conversation
        conversation = conversations.get(conversation_id, {
            'step': 0,
            'info': {}
        })

        # Analyser la question pour extraire les informations
        info = analyze_message(question, conversation['info'])
        conversation['info'].update(info)

        # Déterminer la prochaine question
        response = get_next_question(conversation)

        # Mettre à jour la conversation
        conversations[conversation_id] = conversation

        return jsonify({
            'reponse': response,
            'conversation_id': conversation_id,
            'type': 'text'
        })

    except Exception as e:
        print(f"Erreur serveur : {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Erreur serveur',
            'details': str(e)
        }), 500

def analyze_message(message, current_info):
    """Analyse le message pour en extraire les informations pertinentes"""
    info = {}
    
    # Extraction du nom
    if not current_info.get('name'):
        name_patterns = [
            r"Je m'appelle ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"mon nom est ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) est mon nom"
        ]
        for pattern in name_patterns:
            if match := re.search(pattern, message):
                info['name'] = match.group(1)
                break

    # Extraction email
    if not current_info.get('email'):
        if email_match := re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', message):
            info['email'] = email_match.group()

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
                    break

    return info

def get_next_question(conversation):
    """Détermine la prochaine question à poser"""
    info = conversation['info']
    
    if not info.get('name'):
        return "Pour mieux vous accompagner, pourrais-je connaître votre nom ?"
    
    if not info.get('email'):
        return f"Merci {info['name']}! Pour pouvoir vous envoyer une analyse détaillée, quelle est votre adresse email ?"
    
    if not info.get('patrimoine'):
        return "Pour personnaliser mes recommandations, quel est approximativement votre patrimoine actuel ?"
    
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
    
    Je vous propose un échange téléphonique gratuit avec l'un de nos experts pour approfondir cette analyse.
    
    Souhaitez-vous être recontacté ?"""
    
    return analysis

def parse_amount(amount_str):
    """Convertit une chaîne de montant en nombre"""
    try:
        # Nettoyer la chaîne
        amount = amount_str.replace(' ', '').upper()
        
        # Gérer les K/M
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
    app.run(host='0.0.0.0', port=port, debug=True)
