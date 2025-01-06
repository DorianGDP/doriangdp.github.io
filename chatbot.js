import React, { useState, useEffect, useRef } from 'react';
import { Alert } from '@/components/ui/alert';

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState('');
  const [collectedInfo, setCollectedInfo] = useState({});
  const messagesEndRef = useRef(null);

  const infoSequence = [
    {
      field: 'name',
      question: "Pour mieux vous conseiller, pouvez-vous me donner votre nom et prénom ?",
      validate: (value) => value.split(' ').length >= 2
    },
    {
      field: 'email',
      question: "À quelle adresse email puis-je vous recontacter ?",
      validate: (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)
    },
    {
      field: 'phone',
      question: "Quel est votre numéro de téléphone pour un échange plus personnalisé ?",
      validate: (value) => /^(\+33|0)[1-9](\d{8})$/.test(value.replace(/\s/g, ''))
    },
    {
      field: 'patrimoine',
      question: "Quel est le montant approximatif de votre patrimoine ?",
      options: [
        "Moins de 50 000€",
        "Entre 50 000€ et 200 000€",
        "Entre 200 000€ et 500 000€",
        "Plus de 500 000€"
      ]
    },
    {
      field: 'revenus',
      question: "Dans quelle tranche de revenus annuels vous situez-vous ?",
      options: [
        "Moins de 30 000€",
        "Entre 30 000€ et 60 000€",
        "Entre 60 000€ et 100 000€",
        "Plus de 100 000€"
      ]
    },
    {
      field: 'objectifs',
      question: "Quel est votre principal objectif patrimonial ?",
      options: [
        "Préparer ma retraite",
        "Optimiser ma fiscalité",
        "Investir dans l'immobilier",
        "Protéger mes proches",
        "Autre objectif"
      ]
    }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    const newConversationId = `conv_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    setConversationId(newConversationId);
    
    setMessages([{
      type: 'bot',
      content: "Bonjour ! Je suis Emma, votre conseillère en gestion de patrimoine. Pour vous apporter les meilleurs conseils, j'aimerais mieux comprendre votre situation. Comment puis-je vous aider ?",
    }]);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getNextQuestion = () => {
    for (const info of infoSequence) {
      if (!collectedInfo[info.field]) {
        return info;
      }
    }
    return null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const messageText = userInput.trim();
    setUserInput('');
    setIsLoading(true);

    try {
      setMessages(prev => [...prev, { type: 'user', content: messageText }]);

      const nextQuestion = getNextQuestion();
      let response;

      if (nextQuestion) {
        if (nextQuestion.validate && !nextQuestion.validate(messageText)) {
          response = {
            type: 'bot',
            content: `Je suis désolée, cette information ne semble pas valide. ${nextQuestion.question}`,
            options: nextQuestion.options
          };
        } else {
          setCollectedInfo(prev => ({ ...prev, [nextQuestion.field]: messageText }));
          const newQuestion = getNextQuestion();
          
          if (newQuestion) {
            response = {
              type: 'bot',
              content: `Merci pour cette information. ${newQuestion.question}`,
              options: newQuestion.options
            };
          } else {
            // Toutes les informations sont collectées
            response = {
              type: 'bot',
              content: "Merci pour toutes ces informations. Je vais analyser votre situation et vous faire un premier retour personnalisé...",
            };
            
            // Envoyer les données à Supabase
            await saveToSupabase(collectedInfo);
          }
        }
      }

      setMessages(prev => [...prev, response]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        type: 'bot',
        content: "Je suis désolée, j'ai rencontré une difficulté technique. Pourriez-vous réessayer ?",
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const saveToSupabase = async (info) => {
    try {
      const response = await fetch('/api/save-lead', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...info,
          conversationId,
          timestamp: new Date().toISOString()
        })
      });
      
      if (!response.ok) throw new Error('Failed to save lead');
      
      return await response.json();
    } catch (error) {
      console.error('Failed to save lead:', error);
      throw error;
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 rounded-lg ${
              msg.type === 'user' 
                ? 'bg-cyan-500 text-white rounded-br-none'
                : 'bg-white shadow-md rounded-bl-none'
            }`}>
              <div className="whitespace-pre-wrap">{msg.content}</div>
              
              {msg.options && (
                <div className="mt-3 space-y-2">
                  {msg.options.map((option, optIdx) => (
                    <button
                      key={optIdx}
                      onClick={() => {
                        setUserInput(option);
                        handleSubmit({ preventDefault: () => {} });
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
            className="px-6 py-3 bg-cyan-500 text-white rounded-lg font-medium hover:bg-cyan-600 disabled:opacity-50"
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatBot;
