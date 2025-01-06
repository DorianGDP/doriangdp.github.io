import React, { useState, useEffect, useRef } from 'react';
import { Alert } from '@/components/ui/alert';

const ChatComponent = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [conversationId, setConversationId] = useState('');
  const [collectedInfo, setCollectedInfo] = useState({});
  const [isOptionsDisabled, setIsOptionsDisabled] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // Initialiser la conversation avec un ID unique
    const newConversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setConversationId(newConversationId);
    localStorage.setItem('chatConversationId', newConversationId);
    
    // Message de bienvenue personnalisé
    setMessages([{
      type: 'bot',
      content: "Bonjour ! 👋 Je suis Emma, votre conseillère en gestion de patrimoine. Pour vous offrir les meilleurs conseils, j'aimerais en savoir plus sur votre situation. Comment puis-je vous aider aujourd'hui ?",
      messageType: 'welcome'
    }]);
  }, []);

  // Restaurer l'ID de conversation si disponible
  useEffect(() => {
    const savedConversationId = localStorage.getItem('chatConversationId');
    if (savedConversationId && !conversationId) {
      setConversationId(savedConversationId);
    }
  }, [conversationId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleOptionClick = async (option) => {
    if (isOptionsDisabled) return;
    setIsOptionsDisabled(true);
    
    // Mettre à jour l'interface immédiatement
    setUserInput(option);
    
    // Traiter la réponse
    await handleSubmit({
      preventDefault: () => {},
      optionSelected: true
    }, option);
    
    setIsOptionsDisabled(false);
  };

  const formatMessage = (content) => {
    if (typeof content !== 'string') return content;

    // Remplacer les sauts de ligne par des éléments React
    return content.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        {line}
        {i !== content.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  const handleSubmit = async (e, selectedOption = null) => {
    e.preventDefault();
    const messageText = selectedOption || userInput.trim();
    
    if ((!messageText && !selectedOption) || isLoading) return;

    try {
      setIsLoading(true);
      setError(null);
      setUserInput('');

      // Ajouter le message utilisateur
      setMessages(prev => [...prev, {
        type: 'user',
        content: messageText
      }]);

      const response = await fetch('https://chatbot-gdp.onrender.com/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Origin': 'https://doriangdp.github.io'
        },
        body: JSON.stringify({
          question: messageText,
          conversation_id: conversationId,
          collected_info: collectedInfo // Envoyer les informations déjà collectées
        })
      });

      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }

      const data = await response.json();

      // Mettre à jour les informations collectées si présentes dans la réponse
      if (data.collected_info) {
        setCollectedInfo(prev => ({
          ...prev,
          ...data.collected_info
        }));
      }
      
      // Ajouter le message du bot avec formatage amélioré
      const botMessage = {
        type: 'bot',
        content: data.reponse.content,
        messageType: data.reponse.type,
        options: data.reponse.options
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Erreur:', error);
      setError("Une erreur est survenue lors de l'envoi du message. Veuillez réessayer.");
      
      // Message d'erreur convivial pour l'utilisateur
      setMessages(prev => [...prev, {
        type: 'bot',
        content: "Je suis désolée, j'ai rencontré une difficulté technique. Pourriez-vous reformuler votre message ?",
        messageType: 'error'
      }]);
    } finally {
      setIsLoading(false);
      scrollToBottom();
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
              } ${msg.messageType === 'error' ? 'bg-red-50 text-red-600' : ''}`}
            >
              <div className="whitespace-pre-wrap">
                {formatMessage(msg.content)}
              </div>
              
              {msg.options && msg.options.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.options.map((option, optIdx) => (
                    <button
                      key={optIdx}
                      onClick={() => handleOptionClick(option)}
                      disabled={isOptionsDisabled}
                      className={`w-full p-2 text-left hover:bg-gray-100 rounded-lg border 
                        border-gray-200 transition-colors text-gray-800
                        ${isOptionsDisabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-100'}`}
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
