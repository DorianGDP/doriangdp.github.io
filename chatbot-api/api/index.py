# api/index.py
from http.client import HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurez CORS pour permettre les requêtes de votre site GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour le développement. À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API du Chatbot"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}
