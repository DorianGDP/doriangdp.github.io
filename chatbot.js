import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Send, X } from 'lucide-react';

const PatrimonialChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // Message d'accueil initial
    addMessage('bot', 
      "Bonjour ! Je suis Patty, votre assistante en gestion de patrimoine. " +
      "Pour vous apporter les meilleures recommandations, j'aimerais mieux vous connaître. " +
      "Quelle est votre principale préoccupation patrimoniale ?"
    );
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (type, content, options = []) => {
    setMessages(prev => [...prev, { type, content, options }]);
  };

  const handleResponse = async (message) => {
    try {
      setIsLoading(true);
      const response = await fetch('https://votre-api.com/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          conversationId
        })
      });

      const data = await response.json();
      
      if (data.conversationId) {
        setConversationId(data.conversationId);
      }

      addMessage('bot', data.content, data.options || []);
    } catch (error) {
      console.error('Error:', error);
      addMessage('bot', "Je suis désolée, je rencontre une difficulté technique. Pouvez-vous réessayer ?");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const message = userInput.trim();
    addMessage('user', message);
    setUserInput('');
    await handleResponse(message);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-purple-900 text-white p-4 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center">
            <MessageSquare className="h-6 w-6 text-purple-900" />
          </div>
          <div>
            <h1 className="text-xl font-bold">Patty - Votre Assistante Patrimoniale</h1>
            <p className="text-sm text-gray-200">En ligne</p>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} 
               className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 rounded-lg ${
              msg.type === 'user' 
                ? 'bg-cyan-500 text-white rounded-br-none'
                : 'bg-white shadow-md rounded-bl-none'
            }`}>
              <div className="whitespace-pre-wrap">{msg.content}</div>
              
              {msg.options?.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.options.map((option, optIdx) => (
                    <button
                      key={optIdx}
                      onClick={() => {
                        addMessage('user', option);
                        handleResponse(option);
                      }}
                      className="w-full p-2 text-left hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors text-gray-800"
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

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-4 bg-white border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Tapez votre message..."
            className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-6 py-3 bg-cyan-500 text-white rounded-lg font-medium hover:bg-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  );
};

export default PatrimonialChatbot;
