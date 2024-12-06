from http.server import BaseHTTPRequestHandler
import json
from openai import OpenAI
import numpy as np
import faiss
import os
from supabase import create_client

def handler(request):
    # Lire le corps de la requête
    try:
        body = json.loads(request.get('body', '{}'))
    except:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON body'})
        }

    # Vérifier la présence de la question
    question = body.get('question')
    if not question:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Question is required'})
        }

    try:
        # Initialiser les clients
        openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        supabase_client = create_client(
            os.environ.get('SUPABASE_URL', ''),
            os.environ.get('SUPABASE_KEY', '')
        )

        # Charger l'index FAISS
        index = faiss.read_index('api/faiss_index.idx')

        # Obtenir l'embedding de la question
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        question_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)

        # Rechercher les documents similaires
        k = 3  # nombre de documents à récupérer
        D, I = index.search(question_embedding, k)

        # Récupérer les métadonnées depuis Supabase
        response = supabase_client.table('content_metadata').select('*').in_('id', I[0].tolist()).execute()
        relevant_docs = response.data

        # Générer la réponse avec GPT-4
        context = "\n\n".join([
            f"Titre: {doc['title']}\nContenu: {doc['content']}\nURL: {doc['url']}"
            for doc in relevant_docs
        ])

        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Tu es un assistant virtuel expert en gestion de patrimoine."},
                {"role": "user", "content": f"Question: {question}\n\nContexte:\n{context}"}
            ]
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': completion.choices[0].message.content,
                'sources': relevant_docs
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# Point d'entrée pour Vercel
def endpoint(request):
    return handler(request)
