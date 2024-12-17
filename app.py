# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from chatbot import ChatBot

app = Flask(__name__)
CORS(app)

# Initialisation du chatbot
api_key = os.environ.get('OPENAI_API_KEY')
chatbot = ChatBot(api_key)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'Question manquante'}), 400
        
    try:
        reponse = chatbot.repondre_question(question)
        return jsonify(reponse)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

# requirements.txt
flask==2.0.1
flask-cors==3.0.10
faiss-cpu==1.7.2
numpy==1.21.2
openai==1.3.0
gunicorn==20.1.0

# Procfile
web: gunicorn app:app

# runtime.txt
python-3.9.12
