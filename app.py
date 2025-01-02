from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from chatbot import ChatBot

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialisation du chatbot comme variable globale
chatbot = None

def get_chatbot():
    global chatbot
    if chatbot is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("La clé API OpenAI n'est pas configurée")
        chatbot = ChatBot(api_key)
    return chatbot

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Récupérer et valider les données
        data = request.json
        if not data:
            return jsonify({
                'error': 'Données manquantes',
                'details': 'Le corps de la requête est vide'
            }), 400

        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id', '')

        # Validation des données
        if not question:
            return jsonify({
                'error': 'Question manquante',
                'details': 'Aucune question n\'a été fournie'
            }), 400

        if not conversation_id:
            return jsonify({
                'error': 'ID de conversation manquant',
                'details': 'L\'ID de conversation est requis'
            }), 400

        # Log pour debug
        print(f"Traitement requête - ID: {conversation_id}, Question: {question}")

        # Obtenir l'instance du chatbot et traiter la question
        bot = get_chatbot()
        response = bot.repondre_question(question, conversation_id)

        # Log pour debug
        print(f"Réponse générée: {response}")

        return jsonify(response)

    except Exception as e:
        print(f"Erreur serveur : {str(e)}")
        return jsonify({
            'error': 'Erreur du serveur',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
