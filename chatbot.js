import React, { useState, useEffect, useRef } from 'react';
import { Alert } from '@/components/ui/alert';

const ChatComponent = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [conversationId, setConversationId] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // Initialiser la conversation
    const newConversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setConversationId(newConversationId);
    
    // Message de bienvenue
    setMessages([{
      type: 'bot',
      content: "Bonjour ! 👋 Je suis Emma, votre conseillère en gestion de patrimoine. Comment puis-je vous aider aujourd'hui ?",
      messageType: 'welcome'
    }]);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    try {
      setIsLoading(true);
      setError(null);
      const question = userInput.trim();
      setUserInput('');

      // Ajouter le message utilisateur
      setMessages(prev => [...prev, {
        type: 'user',
        content: question
      }]);

      const response = await fetch('https://chatbot-gdp.onrender.com/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Origin': 'https://doriangdp.github.io'
        },
        body: JSON.stringify({
          question,
          conversation_id: conversationId
        })
      });

      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }

      const data = await response.json();
      
      // Ajouter le message du bot
      setMessages(prev => [...prev, {
        type: 'bot',
        content: data.reponse.content,
        messageType: data.reponse.type,
        options: data.reponse.options
      }]);

    } catch (error) {
      setError("Une erreur est survenue lors de l'envoi du message. Veuillez réessayer.");
      console.error('Erreur:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {error && (
        <Alert variant="destructive" className="m-2">
          {error}
        </Alert>
      )}
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[80%] p-4 rounded-lg ${
                msg.type === 'user'
                  ? 'bg-cyan-500 text-white rounded-br-none'
                  : 'bg-white shadow-md rounded-bl-none'
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>
              {msg.options && msg.options.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.options.map((option, optIdx) => (
                    <button
                      key={optIdx}
                      onClick={() => {
                        setUserInput(option);
                        handleSubmit({ preventDefault: () => {} });
                      }}
                      className="w-full p-2 text-left hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors"
                    >
                      {option}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="p-4 bg-white border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Tapez votre message..."
            className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className={`px-6 py-3 bg-cyan-500 text-white rounded-lg font-medium ${
              isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-cyan-600'
            }`}
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatComponent;
