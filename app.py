from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ChatBot, ConversationStatus
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
        
        # La seule chose qui doit être awaited est repondre_question
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

# Ajout de nouveaux endpoints pour la gestion des conversations
@app.route('/api/chat/reset', methods=['POST'])
async def reset_chat():
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        if not conversation_id:
            return jsonify({'error': 'Conversation ID manquant'}), 400

        # La réinitialisation est une opération synchrone avec Supabase
        new_id = chatbot.conversation_manager.reset_conversation(chatbot, conversation_id)
        
        return jsonify({
            'status': 'success',
            'new_conversation_id': new_id
        })

    except Exception as e:
        print(f"Reset error: {str(e)}")
        return jsonify({'error': 'Erreur lors de la réinitialisation'}), 500

@app.route('/api/chat/timeout', methods=['POST'])
def handle_timeout():
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        if not conversation_id:
            return jsonify({'error': 'Conversation ID manquant'}), 400

        # L'opération de timeout est synchrone
        chatbot.handle_timeout(conversation_id)
        
        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"Timeout error: {str(e)}")
        return jsonify({'error': 'Erreur lors du timeout'}), 500

@app.route('/api/chat/close', methods=['POST'])
def handle_page_close():
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        if not conversation_id:
            return jsonify({'error': 'Conversation ID manquant'}), 400

        # L'opération de fermeture est synchrone
        chatbot.handle_page_unload(conversation_id)
        
        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"Close error: {str(e)}")
        return jsonify({'error': 'Erreur lors de la fermeture'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
