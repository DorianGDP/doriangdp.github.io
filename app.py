from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ChatBot
import os
import traceback

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://doriangdp.github.io"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"]
    }
})

# Initialisation du chatbot avec la clé API
chatbot = ChatBot(os.getenv("OPENAI_API_KEY"))

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        print("Nouvelle requête reçue")
        data = request.get_json()
        print(f"Données reçues : {data}")
        data = request.get_json()
        if not data:
            return jsonify({
                'reponse': {
                    'content': 'Données manquantes',
                    'type': 'error',
                    'options': []
                }
            }), 400

        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id', '')

        if not question:
            return jsonify({
                'reponse': {
                    'content': 'Question manquante',
                    'type': 'error',
                    'options': []
                }
            }), 400

        # Utiliser le chatbot pour obtenir la réponse
        response = await chatbot.repondre_question(question, conversation_id)
        
        # Structurer la réponse pour le frontend
        formatted_response = {
            'reponse': {
                'content': response.get('content', ''),
                'type': response.get('type', 'text'),
                'options': response.get('options', [])
            },
            'conversation_id': conversation_id
        }

        return jsonify(formatted_response)

    except Exception as e:
        print(f"Erreur serveur: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'reponse': {
                'content': "Une erreur s'est produite. Veuillez réessayer.",
                'type': 'error',
                'options': []
            },
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
