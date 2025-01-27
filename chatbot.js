import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Send, User, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const LeadGenerationChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState('');
  const [progress, setProgress] = useState(0);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // Message d'accueil personnalisé
    addMessage('bot', 
      "Bonjour ! 👋 Je suis Patty, votre assistante en gestion de patrimoine. " +
      "Je peux vous aider à optimiser votre situation patrimoniale. " +
      "Pour commencer, quelle est votre principale préoccupation ?"
    );
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (type, content, options = []) => {
    setMessages(prev => [...prev, { type, content, options, timestamp: new Date() }]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const message = userInput.trim();
    setIsLoading(true);
    setUserInput('');

    try {
      addMessage('user', message);
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: message,
          conversation_id: conversationId
        })
      });

      const data = await response.json();
      
      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      if (data.progress) {
        setProgress(data.progress);
      }

      if (data.content) {
        addMessage('bot', data.content, data.options || []);
      }
    } catch (error) {
      console.error('Error:', error);
      addMessage('bot', "Je suis désolée, une erreur est survenue. Pouvez-vous réessayer ?");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Header avec progression */}
      <div className="bg-gradient-to-r from-purple-900 to-purple-700 text-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center">
              <MessageSquare className="h-6 w-6 text-purple-900" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Patty - Votre Assistante</h1>
              <div className="w-full bg-purple-800 rounded-full h-2 mt-2">
                <div 
                  className="bg-cyan-400 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Zone de messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 rounded-lg ${
              msg.type === 'user' 
                ? 'bg-gradient-to-r from-cyan-500 to-cyan-600 text-white rounded-br-none'
                : 'bg-gray-50 shadow-sm rounded-bl-none'
            }`}>
              <div className="whitespace-pre-wrap">{msg.content}</div>
              
              {msg.options?.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.options.map((option, optIdx) => (
                    <button
                      key={optIdx}
                      onClick={() => handleSubmit({ preventDefault: () => {}, target: { value: option } })}
                      className="w-full p-3 text-left hover:bg-white/10 rounded-lg border border-white/20 
                               transition-colors text-white flex items-center gap-2"
                    >
                      <User size={16} />
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

      {/* Zone de saisie */}
      <form onSubmit={handleSubmit} className="p-4 border-t bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Tapez votre message..."
            className="flex-1 p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-cyan-600 text-white rounded-lg 
                     font-medium hover:from-cyan-600 hover:to-cyan-700 disabled:opacity-50 
                     disabled:cursor-not-allowed flex items-center gap-2 transition-all"
          >
            {isLoading ? 'Envoi...' : 'Envoyer'}
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  );
};

export default LeadGenerationChatbot;
