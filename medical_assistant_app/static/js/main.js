// medical_assistant_app/static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Function to append a message to the chat box
    function appendMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        if (sender === 'user') {
            messageDiv.classList.add('user-message');
        } else {
            messageDiv.classList.add('assistant-message');
        }
        messageDiv.innerHTML = `<p>${message}</p>`;
        chatBox.appendChild(messageDiv);
        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to send message to the backend
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') {
            return; // Don't send empty messages
        }

        appendMessage('user', message);
        userInput.value = ''; // Clear input field

        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        sendButton.disabled = true; // Disable send button while loading

        try {
            const response = await fetch('/api/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken // Get CSRF token from the global variable set in index.html
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                // Attempt to read error message from backend
                let errorData;
                try {
                    errorData = await response.json();
                } catch (jsonError) {
                    errorData = { response: `Server error: ${response.status} ${response.statusText}` };
                }
                throw new Error(errorData.response || 'Something went wrong on the server.');
            }

            const data = await response.json();
            appendMessage('assistant', data.response);
        } catch (error) {
            console.error('Error sending message:', error);
            // Display an error message to the user
            appendMessage('assistant', `Error: ${error.message}. Please try again.`);
        } finally {
            // Hide loading indicator and re-enable button
            loadingIndicator.classList.add('hidden');
            sendButton.disabled = false;
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});
