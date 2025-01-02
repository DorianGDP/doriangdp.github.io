from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from chatbot import ChatBot
import asyncio
import traceback

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
        # Debug log
        print("Headers reçus:", dict(request.headers))
        print("Corps de la requête brut:", request.get_data(as_text=True))
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Données manquantes',
                'details': 'Le corps de la requête est vide'
            }), 400

        # Log pour debug
        print("Données reçues:", data)
        
        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id', '').strip()
        
        # Validation plus souple du conversation_id
        if not conversation_id:
            conversation_id = f"conv_{int(time.time())}"
        
        if not question:
            return jsonify({
                'error': 'Question manquante',
                'details': 'Aucune question n\'a été fournie'
            }), 400

        # Log avant appel chatbot
        print(f"Traitement requête - ID: {conversation_id}, Question: {question}")

        # Appel au chatbot
        try:
            response = run_async(chatbot.repondre_question(question, conversation_id))
            print("Réponse chatbot:", response)
        except Exception as e:
            print("Erreur chatbot:", str(e))
            traceback.print_exc()
            raise

        if not response:
            return jsonify({
                'error': 'Réponse vide',
                'details': 'Le chatbot n\'a pas généré de réponse'
            }), 500

        # Ajout du conversation_id à la réponse
        response['conversation_id'] = conversation_id
        
        return jsonify(response)

    except Exception as e:
        print(f"Erreur serveur : {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Erreur du serveur',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
