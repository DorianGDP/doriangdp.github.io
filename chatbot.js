import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const ChatbotComponent = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const messagesEndRef = useRef(null);
  
  const MAX_RETRIES = 3;
  const RETRY_DELAY = 2000;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const question = userInput.trim();
    setUserInput('');
    setError(null);
    setIsLoading(true);

    // Ajouter le message utilisateur immédiatement
    setMessages(prev => [...prev, {
      type: 'user',
      content: question
    }]);

    try {
      const response = await fetch('https://chatbot-gdp.onrender.com/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Origin': 'https://doriangdp.github.io'
        },
        body: JSON.stringify({
          question,
          conversation_id: localStorage.getItem('chatConversationId') || `conv_${Date.now()}`
        })
      });

      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setMessages(prev => [...prev, {
        type: 'bot',
        content: data.reponse
      }]);
      
      setRetryCount(0); // Réinitialiser le compteur après un succès

    } catch (error) {
      console.error('Erreur:', error);
      setError(error.message);
      
      if (retryCount < MAX_RETRIES) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          handleSubmit(e);
        }, RETRY_DELAY);
      } else {
        setMessages(prev => [...prev, {
          type: 'bot',
          content: "Je suis désolée, je rencontre des difficultés techniques. Pourriez-vous reformuler votre question différemment ?",
          isError: true
        }]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {error}
            </AlertDescription>
          </Alert>
        )}
        
        {messages.map((msg, idx) => (
          <div key={idx} 
               className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'} 
                          message-animation`}>
            <div className={`max-w-[80%] p-4 rounded-lg shadow-sm
              ${msg.type === 'user' 
                ? 'bg-cyan-500 text-white rounded-br-none' 
                : 'bg-white rounded-bl-none'}
              ${msg.isError ? 'bg-red-50 text-red-600 border border-red-200' : ''}`}>
              {msg.content}
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
            className={`px-6 py-3 bg-cyan-500 text-white rounded-lg font-medium 
              ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-cyan-600'}
              transition-all duration-200`}
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatbotComponent;
