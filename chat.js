// chatbot.js
(function() {
    // Création des styles
    const style = document.createElement('style');
    style.textContent = `
        .chatbot-tab {
            position: fixed;
            left: 0;
            bottom: 20%;
            transform: rotate(-90deg);
            transform-origin: left bottom;
            z-index: 1000;
            transition: transform 0.3s ease;
        }
        
        .chatbot-container {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 400px;
            height: 600px;
            transform: translateX(-100%);
            transition: transform 0.3s ease;
            z-index: 999;
            background: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        
        .chatbot-container.open {
            transform: translateX(0);
        }
        
        .chatbot-frame {
            width: 100%;
            height: 100%;
            border: none;
        }
        
        @media (max-width: 640px) {
            .chatbot-container {
                width: 100%;
                height: 500px;
            }
        }
    `;
    document.head.appendChild(style);

    // Création du bouton et du conteneur
    const chatbotHTML = `
        <button id="chatbotTab" class="chatbot-tab bg-cyan-400 text-white px-6 py-3 rounded-t-lg hover:bg-cyan-500">
            Assistant Patrimonial
        </button>
        <div id="chatbotContainer" class="chatbot-container">
            <iframe src="chat.html" class="chatbot-frame"></iframe>
        </div>
    `;

    // Ajout des éléments au DOM
    const div = document.createElement('div');
    div.innerHTML = chatbotHTML;
    document.body.appendChild(div);

    // Gestion des événements
    let isOpen = false;
    const tab = document.getElementById('chatbotTab');
    const container = document.getElementById('chatbotContainer');

    tab.addEventListener('click', () => {
        isOpen = !isOpen;
        container.classList.toggle('open');
        
        // Animation du tab
        tab.style.transform = isOpen 
            ? 'rotate(-90deg) translateX(-400px)' 
            : 'rotate(-90deg)';
    });
})();
