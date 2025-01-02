import React, { useState, useEffect, useRef } from 'react';

const ChatComponent = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [conversationId, setConversationId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // Message de bienvenue initial
    setMessages([{
      type: 'bot',
      content: "Bonjour ! 👋 Je suis Emma, votre conseillère en gestion de patrimoine. Comment puis-je vous aider aujourd'hui ?"
    }]);
    
    // Générer un ID de conversation unique
    setConversationId(`conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  }, []);

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    setIsLoading(true);
    const question = userInput.trim();
    setUserInput('');

    // Ajouter le message utilisateur
    setMessages(prev => [...prev, {
      type: 'user',
      content: question
    }]);

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          conversation_id: conversationId
        })
      });

      if (!response.ok) {
        throw new Error('Erreur réseau');
      }

      const data = await response.json();
      
      setMessages(prev => [...prev, {
        type: 'bot',
        content: data.reponse,
        messageType: data.type
      }]);

    } catch (error) {
      console.error('Erreur:', error);
      setMessages(prev => [...prev, {
        type: 'bot',
        content: "Désolé, une erreur s'est produite. Pouvez-vous réessayer ?",
        isError: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[75%] rounded-lg p-3 ${
                msg.type === 'user'
                  ? 'bg-cyan-500 text-white'
                  : 'bg-white text-gray-800 shadow'
              }`}
            >
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
            className="flex-1 p-2 border rounded-lg focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className={`px-4 py-2 bg-cyan-500 text-white rounded-lg ${
              isLoading ? 'opacity-50' : 'hover:bg-cyan-600'
            }`}
          >
            {isLoading ? '...' : 'Envoyer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatComponent;
