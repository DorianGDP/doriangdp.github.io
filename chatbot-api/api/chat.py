from http.client import HTTPException
import json
from openai import OpenAI
import faiss
import os
from supabase import create_client, Client

def handler(request):
    # Headers CORS
    headers = {
        'Access-Control-Allow-Origin': '*',  # En production, spécifiez votre domaine
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }

    # Gérer la requête OPTIONS (preflight)
    if request.method == "OPTIONS":
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }

    try:
        body = json.loads(request.body)
        question = body.get('question')
        
        if not question:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Question manquante'})
            }

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
            'headers': headers,
            'body': json.dumps({
                'response': "Votre réponse ici",
                'sources': []
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }

# Point d'entrée pour Vercel
def endpoint(request):
    return handler(request)
