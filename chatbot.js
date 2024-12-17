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
