import React, { useState, useEffect, useRef } from 'react';

const ImprovedChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const [collectedInfo, setCollectedInfo] = useState({});

  const questions = {
    name: {
      text: "Pour mieux vous conseiller, quel est votre nom et prénom ?",
      type: "text"
    },
    phone: {
      text: "Quel est votre numéro de téléphone pour un échange plus personnalisé ?",
      type: "text"
    },
    email: {
      text: "À quelle adresse email puis-je vous recontacter ?",
      type: "text"
    },
    income: {
      text: "Dans quelle tranche de revenus annuels vous situez-vous ?",
      type: "options",
      options: [
        "Moins de 30 000€",
        "30 000€ - 50 000€",
        "50 000€ - 100 000€",
        "Plus de 100 000€"
      ]
    },
    familyStatus: {
      text: "Quelle est votre situation familiale actuelle ?",
      type: "options",
      options: [
        "Célibataire",
        "Marié(e)",
        "En couple",
        "Divorcé(e)",
        "Avec enfants"
      ]
    }
  };

  const sendMessage = async (message) => {
    try {
      setIsLoading(true);
      
      // Simuler l'appel API
      const response = await fetch('your-api-endpoint', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          collectedInfo
        })
      });
      
      const data = await response.json();
      
      // Mise à jour des informations collectées
      if (data.collectedInfo) {
        setCollectedInfo(prev => ({
          ...prev,
          ...data.collectedInfo
        }));
      }
      
      // Ajouter la réponse du bot
      addMessage('bot', data.response);
      
    } catch (error) {
      addMessage('bot', "Je suis désolée, je rencontre une difficulté technique. Pouvez-vous réessayer ?");
    } finally {
      setIsLoading(false);
    }
  };

  const addMessage = (type, content, options = []) => {
    setMessages(prev => [...prev, {
      type,
      content,
      options
    }]);
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 rounded-lg ${
              msg.type === 'user' 
                ? 'bg-cyan-500 text-white rounded-br-none'
                : 'bg-white shadow-md rounded-bl-none'
            }`}>
              <div className="whitespace-pre-wrap">{msg.content}</div>
              
              {msg.options && msg.options.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.options.map((option, optIdx) => (
                    <button
                      key={optIdx}
                      onClick={() => sendMessage(option)}
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

      <div className="p-4 bg-white border-t">
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
            onClick={() => {
              if (userInput.trim()) {
                addMessage('user', userInput);
                sendMessage(userInput);
                setUserInput('');
              }
            }}
            disabled={isLoading}
            className="px-6 py-3 bg-cyan-500 text-white rounded-lg font-medium hover:bg-cyan-600"
          >
            Envoyer
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImprovedChatbot;
