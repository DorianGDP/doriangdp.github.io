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
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données manquantes'}), 400

        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id', '')

        if not question:
            return jsonify({'error': 'Question manquante'}), 400

        # Utiliser le chatbot pour obtenir la réponse
        response = await chatbot.repondre_question(question, conversation_id)
        
        return jsonify(response)

    except Exception as e:
        print(f"Erreur serveur: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Erreur serveur',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
