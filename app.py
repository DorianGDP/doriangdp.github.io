from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from chatbot import ChatBot
import asyncio

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialisation du chatbot comme variable globale
chatbot = ChatBot(os.environ.get('OPENAI_API_KEY'))

def run_async(coroutine):
    """Helper function to run async code in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({
                'error': 'Données manquantes',
                'details': 'Le corps de la requête est vide'
            }), 400

        # Log pour debug
        print("Requête reçue:", data)
        
        question = data.get('question', '')
        conversation_id = data.get('conversation_id', '')
        
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

        # Appel au chatbot et récupération de la réponse de manière synchrone
        response = run_async(chatbot.repondre_question(question, conversation_id))
        
        # Log pour debug
        print("Réponse générée:", response)
        
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
