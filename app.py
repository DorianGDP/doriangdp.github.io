from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ChatBot, ConversationStatus, ConversationManager
import os
import traceback
import asyncio
from datetime import datetime

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

        # Vérifier si la conversation existe et n'est pas expirée
        conv_data = await chatbot.supabase.table('conversations')\
            .select('created_at, status')\
            .eq('conversation_id', conversation_id)\
            .execute()

        if conv_data.data:
            created_at = datetime.fromisoformat(conv_data.data[0]['created_at'])
            if ConversationManager.is_conversation_expired(created_at):
                return jsonify(await ConversationManager.handle_conversation_timeout(chatbot, conversation_id))

            status = conv_data.data[0]['status']
            if status == ConversationStatus.TERMINEE.value:
                return jsonify({
                    'type': 'text',
                    'content': "Cette conversation est terminée. Souhaitez-vous en commencer une nouvelle ?",
                    'options': ["Commencer une nouvelle conversation"]
                })

        response = await chatbot.repondre_question(question, conversation_id)
        return jsonify(response)

    except Exception as e:
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'content': "Une erreur technique est survenue",
            'type': 'error'
        }), 500

@app.route('/api/reset', methods=['POST'])
async def reset_conversation():
    try:
        data = request.json
        old_conversation_id = data.get('conversation_id', '')

        new_conversation_id = await ConversationManager.reset_conversation(chatbot, old_conversation_id)
        
        if new_conversation_id:
            return jsonify({
                'status': 'success',
                'new_conversation_id': new_conversation_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': "Impossible de réinitialiser la conversation"
            }), 500

    except Exception as e:
        print(f"Reset error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': "Une erreur technique est survenue"
        }), 500

@app.route('/api/page-close', methods=['POST'])
async def handle_page_close():
    try:
        data = request.json
        conversation_id = data.get('conversation_id', '')

        await ConversationManager.handle_page_close(chatbot, conversation_id)
        
        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"Page close error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': "Une erreur est survenue lors de la fermeture"
        }), 500

@app.route('/api/timeout', methods=['POST'])
async def handle_timeout():
    try:
        data = request.json
        conversation_id = data.get('conversation_id', '')

        response = await ConversationManager.handle_conversation_timeout(chatbot, conversation_id)
        return jsonify(response)

    except Exception as e:
        print(f"Timeout error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': "Une erreur est survenue lors du timeout"
        }), 500

@app.route('/api/status', methods=['GET'])
async def get_conversation_status():
    try:
        conversation_id = request.args.get('conversation_id', '')
        if not conversation_id:
            return jsonify({
                'status': 'error',
                'message': "ID de conversation manquant"
            }), 400

        conv_data = await chatbot.supabase.table('conversations')\
            .select('status, created_at')\
            .eq('conversation_id', conversation_id)\
            .execute()

        if not conv_data.data:
            return jsonify({
                'status': 'error',
                'message': "Conversation non trouvée"
            }), 404

        conversation = conv_data.data[0]
        created_at = datetime.fromisoformat(conversation['created_at'])
        
        return jsonify({
            'status': conversation['status'],
            'created_at': created_at.isoformat(),
            'is_expired': ConversationManager.is_conversation_expired(created_at)
        })

    except Exception as e:
        print(f"Status check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': "Une erreur est survenue lors de la vérification du statut"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
