// chatbot.js
document.addEventListener('DOMContentLoaded', function() {
    const chatbotButton = document.getElementById('chatbotButton');
    const chatbotContainer = document.getElementById('chatbotContainer');
    let isOpen = false;

    chatbotButton.addEventListener('click', function() {
        isOpen = !isOpen;
        chatbotContainer.classList.toggle('hidden');
    });
});

// chat.html (à placer dans le même dossier que index.html)
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="p-4">
    <div id="chat-messages" class="h-[500px] overflow-y-auto mb-4"></div>
    <div class="flex gap-2">
        <input type="text" id="user-input" class="flex-1 p-2 border rounded" placeholder="Posez votre question...">
        <button id="send-button" class="bg-cyan-400 text-white px-4 py-2 rounded">Envoyer</button>
    </div>

    <script>
        const messagesContainer = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        async function sendMessage() {
            const question = userInput.value.trim();
            if (!question) return;

            // Afficher le message de l'utilisateur
            appendMessage('user', question);
            userInput.value = '';

            try {
                const response = await fetch('VOTRE_URL_API/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                appendMessage('bot', data.reponse);
            } catch (error) {
                appendMessage('bot', "Désolé, une erreur s'est produite.");
            }
        }

        function appendMessage(sender, text) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `mb-4 p-2 rounded ${sender === 'user' ? 'bg-gray-100 ml-8' : 'bg-cyan-100 mr-8'}`;
            messageDiv.textContent = text;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
