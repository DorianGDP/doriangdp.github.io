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

        if not data:
            return jsonify({
                'content': "Données manquantes",
                'type': 'error'
            }), 400

        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id', '')

        if not question:
            return jsonify({
                'content': "Question manquante",
                'type': 'error'
            }), 400

        # Obtenir la réponse du chatbot
        response = await chatbot.repondre_question(question, conversation_id)
        print(f"Réponse du chatbot : {response}")

        # Structurer la réponse pour le frontend
        formatted_response = {
            'content': response.get('content', ''),
            'type': response.get('type', 'text'),
            'options': response.get('options', []),
            'conversation_id': conversation_id  # Ajout de l'ID de conversation
        }

        print(f"Réponse formatée : {formatted_response}")
        return jsonify(formatted_response)

    except Exception as e:
        print(f"Erreur serveur: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'content': "Une erreur s'est produite. Veuillez réessayer.",
            'type': 'error'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
