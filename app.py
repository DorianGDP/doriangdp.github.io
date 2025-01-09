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

@app.route('/api/check_timeout', methods=['POST'])
async def check_timeout():
    try:
        data = request.json
        if not data or 'conversation_id' not in data:
            return jsonify({
                'timeout': False,
                'error': 'Conversation ID manquant'
            }), 400

        conversation_id = data['conversation_id']
        timeout = await chatbot.check_conversation_timeout(conversation_id)
        
        if timeout:
            # Si timeout, terminer la conversation
            await chatbot.handle_conversation_end(
                conversation_id, 
                chatbot.ConversationStatus.NON_TERMINEE
            )

        return jsonify({'timeout': timeout})

    except Exception as e:
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'timeout': False,
            'error': "Une erreur technique est survenue"
        }), 500

@app.route('/api/end_conversation', methods=['POST'])
async def end_conversation():
    try:
        data = request.json
        if not data or 'conversation_id' not in data:
            return jsonify({
                'error': 'Conversation ID manquant'
            }), 400

        conversation_id = data['conversation_id']
        status = data.get('status', 'non_terminee')
        
        # Convertir le status en enum
        status_enum = chatbot.ConversationStatus.NON_TERMINEE
        if status == 'terminee':
            status_enum = chatbot.ConversationStatus.TERMINEE
        elif status == 'en_cours':
            status_enum = chatbot.ConversationStatus.EN_COURS

        await chatbot.handle_conversation_end(conversation_id, status_enum)
        return jsonify({'success': True})

    except Exception as e:
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': "Une erreur technique est survenue"
        }), 500

@app.route('/api/reset_conversation', methods=['POST'])
async def reset_conversation():
    try:
        data = request.json
        if not data or 'conversation_id' not in data:
            return jsonify({
                'error': 'Conversation ID manquant'
            }), 400

        conversation_id = data['conversation_id']
        new_conversation_id = await chatbot.reset_conversation(conversation_id)
        
        return jsonify({
            'success': True,
            'new_conversation_id': new_conversation_id
        })

    except Exception as e:
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': "Une erreur technique est survenue"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
