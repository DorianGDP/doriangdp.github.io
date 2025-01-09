from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ChatBot
import os
import traceback
import asyncio

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://doriangdp.github.io"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"]
    }
})

chatbot = ChatBot(os.getenv("OPENAI_API_KEY"))

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({
                'content': "Question manquante",
                'type': 'error'
            }), 400

        question = data['question'].strip()
        conversation_id = data.get('conversation_id', '')

        response = await chatbot.repondre_question(question, conversation_id)
        
        return jsonify({
            'content': response.get('content', ''),
            'type': response.get('type', 'text'),
            'options': response.get('options', []),
            'conversation_id': conversation_id
        })

    except Exception as e:
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'content': "Une erreur technique est survenue",
            'type': 'error'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
