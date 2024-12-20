# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from chatbot import ChatBot

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialisation du chatbot
api_key = os.environ.get('OPENAI_API_KEY')
chatbot = ChatBot(api_key)

@app.route('/api/chat', methods=['POST'])
def chat():
    # Log de débogage
    print("Requête reçue:")
    print(request.json)
    
    data = request.json or {}
    question = data.get('question', '')
    conversation_id = data.get('conversation_id', '')
    
    if not question:
        print("Erreur : Question manquante")
        return jsonify({
            'error': 'Question manquante', 
            'details': 'Aucune question n\'a été fournie'
        }), 400
    
    try:
        reponse = chatbot.repondre_question(question, conversation_id)
        return jsonify(reponse)
    except Exception as e:
        print(f"Erreur serveur : {str(e)}")
        return jsonify({
            'error': 'Erreur du serveur', 
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
