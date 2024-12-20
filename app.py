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
    # Log de la requête reçue
    print("Données reçues:", request.json)
    
    data = request.json or {}
    question = data.get('question')
    conversation_id = data.get('conversation_id')
    is_qcm_response = data.get('is_qcm_response', False)
    qcm_type = data.get('qcm_type')
    
    # Log détaillé
    print(f"Question: {question}")
    print(f"Conversation ID: {conversation_id}")
    print(f"Is QCM Response: {is_qcm_response}")
    print(f"QCM Type: {qcm_type}")
    
    if not question:
        print("Erreur : Question manquante")
        return jsonify({'error': 'Question manquante'}), 400
        
    try:
        reponse = chatbot.repondre_question(question, conversation_id)
        
        # Ajoutez des informations supplémentaires si nécessaire
        reponse['is_qcm_response'] = is_qcm_response
        reponse['qcm_type'] = qcm_type
        
        return jsonify(reponse)
    except Exception as e:
        print(f"Erreur détaillée: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
