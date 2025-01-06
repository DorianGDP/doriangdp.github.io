import React, { useState, useEffect, useRef } from 'react';

const ImprovedChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [collectedInfo, setCollectedInfo] = useState({});
  const [conversationId, setConversationId] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  
  const handleResponse = async (userMessage) => {
    try {
      setIsLoading(true);
      
      const response = await fetch('https://chatbot-gdp.onrender.com/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMessage,
          conversation_id: conversationId,
          collected_info: collectedInfo
        })
      });

      const data = await response.json();
      
      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      if (data.collected_info) {
        setCollectedInfo(prev => ({
          ...prev,
          ...data.collected_info
        }));
      }

      addMessage('bot', data.content, data.options || []);
      
    } catch (error) {
      addMessage('bot', "Je suis désolée, je rencontre une difficulté technique. Pouvez-vous réessayer ?");
    } finally {
      setIsLoading(false);
    }
  };

  const addMessage = (type, content, options = []) => {
    setMessages(prev => [...prev, { type, content, options }]);
    setTimeout(scrollToBottom, 100);
  };

  useEffect(() => {
    addMessage('bot', "Bonjour ! Je suis Emma, votre conseillère en gestion de patrimoine. Comment puis-je vous aider aujourd'hui ?");
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const message = userInput.trim();
    setUserInput('');
    addMessage('user', message);
    handleResponse(message);
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} 
               className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'} message-animation`}>
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
                      className="w-full p-2 text-left hover:bg-gray-100 rounded-lg border 
                               border-gray-200 transition-colors text-gray-800"
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
            className="px-6 py-3 bg-cyan-500 text-white rounded-lg font-medium
                     hover:bg-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ImprovedChatbot;
