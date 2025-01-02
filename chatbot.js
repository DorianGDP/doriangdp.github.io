import React, { useState, useEffect, useRef } from 'react';
import { AlertCircle } from 'lucide-react';

const ChatComponent = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [conversationId, setConversationId] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Générer un ID unique pour la conversation
    const newConversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setConversationId(newConversationId);

    // Ajouter le message de bienvenue
    setMessages([{
      type: 'bot',
      content: "Bonjour ! 👋 Je suis Emma, votre conseillère en gestion de patrimoine. Comment puis-je vous aider aujourd'hui ?"
    }]);
  }, []);

  const handleSubmit = async () => {
    if (!userInput.trim() || isLoading) return;

    try {
      setIsLoading(true);

      // Ajouter le message de l'utilisateur
      setMessages(prev => [...prev, {
        type: 'user',
        content: userInput.trim()
      }]);

      // Ajouter un message "en train d'écrire"
      setMessages(prev => [...prev, {
        type: 'bot',
        content: "...",
        isTyping: true
      }]);

      // Envoyer la requête
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userInput.trim(),
          conversation_id: conversationId
        })
      });

      if (!response.ok) throw new Error('Erreur réseau');

      const data = await response.json();

      // Mettre à jour les messages
      setMessages(prev => [
        ...prev.filter(msg => !msg.isTyping),
        {
          type: 'bot',
          content: data.reponse,
          messageType: data.type
        }
      ]);

      // Vider l'input
      setUserInput('');

    } catch (error) {
      console.error('Erreur:', error);
      setMessages(prev => [
        ...prev.filter(msg => !msg.isTyping),
        {
          type: 'bot',
          content: "Désolé, une erreur s'est produite. Pouvez-vous réessayer ?",
          isError: true
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatMessage = (content, messageType) => {
    if (messageType === 'analysis') {
      return (
        <div className="bg-gradient-to-r from-purple-50 to-cyan-50 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-purple-800 mb-2">
            Analyse personnalisée
          </h3>
          <div className="space-y-2 text-gray-700">
            {content.split('\n').map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </div>
        </div>
      );
    }
    
    return (
      <div className="whitespace-pre-wrap">
        {content}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* En-tête */}
      <div className="bg-gradient-to-r from-purple-800 to-indigo-800 p-4 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-white rounded-full flex items-center justify-
