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
    const newConversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setConversationId(newConversationId);
    
    // Sauvegarder l'ID dans le localStorage pour persister entre les rafraîchissements
    localStorage.setItem('chatConversationId', newConversationId);
  }, []);

  // Restaurer l'ID de conversation depuis le localStorage si disponible
  useEffect(() => {
    const savedConversationId = localStorage.getItem('chatConversationId');
    if (savedConversationId && !conversationId) {
      setConversationId(savedConversationId);
    }
  }, [conversationId]);

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
        throw new Error('Erreur réseau');
      }

      const data = await response.json();
      
      if (data.debug_info) {
        console.log('Debug info:', data.debug_info);
      }
      
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
      scrollToBottom();
    }
  };

  const formatMessage = (content) => {
    // Fonction pour formater le contenu du message
    if (typeof content !== 'string') return content;

    // Remplacer les sauts de ligne par des balises <br>
    const formattedContent = content.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        {line}
        {i !== content.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));

    return formattedContent;
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[80%] p-4 rounded-lg ${
                msg.type === 'user'
                  ? 'bg-cyan-500 text-white rounded-br-none'
                  : 'bg-white shadow-md rounded-bl-none'
              } ${msg.isError ? 'bg-red-50 text-red-600' : ''}`}
            >
              {formatMessage(msg.content)}
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
              ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-cyan-600'}`}
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatComponent;
