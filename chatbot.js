import React, { useState, useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';

const ImprovedChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // Initialize chat with welcome message
    addMessage('bot', 
      "Bonjour ! Je suis Patty, votre conseillère en gestion de patrimoine. " +
      "Je suis là pour répondre à vos questions et vous aider à optimiser votre patrimoine. " +
      "Pour commencer, pouvez-vous me dire ce qui vous amène aujourd'hui ?"
    );
  }, []);

  const handleResponse = async (userMessage) => {
    try {
      setIsLoading(true);
      
      const response = await fetch('YOUR_API_ENDPOINT', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          conversation_id: conversationId
        })
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      addMessage('bot', data.content, data.options || []);
      
    } catch (error) {
      console.error('Error:', error);
      addMessage('bot', 
        "Je suis désolée, je rencontre actuellement des difficultés techniques. " +
        "Pouvez-vous réessayer dans quelques instants ?"
      );
    } finally {
      setIsLoading(false);
    }
  };

  const addMessage = (type, content, options = []) => {
    setMessages(prev => [
      ...prev, 
      { type, content, options, timestamp: new Date().toISOString() }
    ]);
    setTimeout(scrollToBottom, 100);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const message = userInput.trim();
    setUserInput('');
    addMessage('user', message);
    handleResponse(message);
  };

  return (
    <Card className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} 
               className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-lg ${
              msg.type === 'user' 
                ? 'bg-cyan-500 text-white rounded-br-none'
                : 'bg-gray-100 rounded-bl-none'
            }`}>
              <p className="whitespace-pre-wrap">{msg.content}</p>
              
              {msg.options?.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.options.map((option, optIdx) => (
                    <button
                      key={optIdx}
                      onClick={() => {
                        addMessage('user', option);
                        handleResponse(option);
                      }}
                      className="w-full p-2 text-left hover:bg-gray-50 rounded
                               transition-colors text-gray-800 border border-gray-200"
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

      <form onSubmit={handleSubmit} 
            className="border-t p-4 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Tapez votre message..."
            className="flex-1 p-2 border rounded-lg focus:outline-none 
                     focus:ring-2 focus:ring-cyan-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-4 py-2 bg-cyan-500 text-white rounded-lg
                     hover:bg-cyan-600 disabled:opacity-50 
                     disabled:cursor-not-allowed"
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
          </button>
        </div>
      </form>
    </Card>
  );
};

export default ImprovedChatbot;
