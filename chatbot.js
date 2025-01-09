import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Send, User } from 'lucide-react';

const ImprovedChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [collectedInfo, setCollectedInfo] = useState({});
  const [conversationId, setConversationId] = useState('');
  const [finalQuestion, setFinalQuestion] = useState('');
  const [showFinalQuestion, setShowFinalQuestion] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    addMessage('bot', "Bonjour ! Je suis Patty, votre conseillère en gestion de patrimoine. Pour mieux vous accompagner, j'aimerais en savoir plus sur vous. Comment puis-je vous aider ?");
  }, []);

  const formatAnalysis = (analysis) => {
    // Divise l'analyse en sections
    const sections = analysis.split('\n\n');
    return (
      <div className="space-y-4">
        {/* Introduction */}
        <div className="font-semibold text-lg">
          Analyse Patrimoniale Personnalisée
        </div>

        {/* Situation actuelle */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-medium text-gray-900 mb-2">Votre Situation</h3>
          <p className="text-gray-700">{sections[0]}</p>
        </div>

        {/* Question et Réponse */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-medium text-gray-900 mb-2">Votre Question</h3>
          <p className="text-gray-700 italic mb-3">{finalQuestion}</p>
          <h3 className="font-medium text-gray-900 mb-2">Notre Réponse</h3>
          <p className="text-gray-700">{sections[1]}</p>
        </div>

        {/* Recommandations */}
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-medium text-blue-900 mb-2">Nos Recommandations</h3>
          <div className="space-y-2">
            {sections[2].split('\n').map((rec, index) => (
              <div key={index} className="flex items-start gap-2">
                <div className="min-w-[24px] h-6 flex items-center justify-center rounded-full bg-blue-100 text-blue-800 text-sm">
                  {index + 1}
                </div>
                <p className="text-blue-800">{rec}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Conclusion */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-medium text-gray-900 mb-2">Prochaines Étapes</h3>
          <p className="text-gray-700">{sections[3]}</p>
        </div>
      </div>
    );
  };

  const handleFinalQuestion = async () => {
    if (!finalQuestion.trim()) return;
    
    setIsLoading(true);
    // Envoyer la question finale au serveur avec toutes les informations collectées
    try {
      const response = await fetch('API_URL', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: finalQuestion,
          conversation_id: conversationId,
          collected_info: collectedInfo,
          is_final_question: true
        })
      });

      const data = await response.json();
      // Afficher l'analyse formatée
      addMessage('bot', formatAnalysis(data.content));
      setShowFinalQuestion(false);
    } catch (error) {
      console.error('Error:', error);
      addMessage('bot', "Je suis désolée, une erreur est survenue lors de l'analyse de votre situation.");
    } finally {
      setIsLoading(false);
    }
  };

  const addMessage = (type, content, options = []) => {
    setMessages(prev => [...prev, { type, content, options }]);
    setTimeout(scrollToBottom, 100);
  };

  const handleResponse = async (userMessage) => {
    try {
      setIsLoading(true);
      const response = await fetch('API_URL', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMessage,
          conversation_id: conversationId,
          collected_info: collectedInfo
        })
      });

      const data = await response.json();
      
      if (data.show_final_question) {
        setShowFinalQuestion(true);
        addMessage('bot', "Parfait ! J'ai toutes les informations nécessaires. Pour vous fournir une analyse personnalisée, pourriez-vous me préciser votre question ou votre objectif principal ?");
      } else {
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
      }
    } catch (error) {
      console.error('Error:', error);
      addMessage('bot', "Je suis désolée, je rencontre une difficulté technique. Pouvez-vous réessayer ?");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    if (showFinalQuestion) {
      setFinalQuestion(userInput.trim());
      handleFinalQuestion();
    } else {
      const message = userInput.trim();
      addMessage('user', message);
      handleResponse(message);
    }
    setUserInput('');
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 rounded-lg ${
              msg.type === 'user' 
                ? 'bg-blue-600 text-white rounded-br-none'
                : 'bg-white shadow-md rounded-bl-none'
            }`}>
              {typeof msg.content === 'string' ? (
                <div className="whitespace-pre-wrap">{msg.content}</div>
              ) : (
                msg.content
              )}
              
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

      <form onSubmit={handleSubmit} className="p-4 bg-white border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder={showFinalQuestion ? "Quelle est votre question principale ?" : "Tapez votre message..."}
            className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  );
};

export default ImprovedChatbot;
