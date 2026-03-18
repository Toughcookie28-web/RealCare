// DOM Elements
const chatArea = document.getElementById('chatArea');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeScreen = document.getElementById('welcomeScreen');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');
const typingText = document.querySelector('.typing-text');
const newChatBtn = document.getElementById('newChatBtn');
const clearBtn = document.getElementById('clearBtn');
const themeBtn = document.getElementById('themeBtn');
const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('mainContent');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');
const chatList = document.getElementById('chatList');
const downloadBtn = document.getElementById('downloadBtn');

// Persistent user identity — generated once, stored in localStorage
function getUserToken() {
    let token = localStorage.getItem('mg_user_token');
    if (!token) {
        token = 'u-' + crypto.randomUUID();
        localStorage.setItem('mg_user_token', token);
    }
    return token;
}

// State
let isTyping = false;
let chatHistory = [];
let currentTheme = localStorage.getItem('theme') || 'light';
let sidebarOpen = localStorage.getItem('sidebarOpen') !== 'false';
let currentSessionId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    initializeEventListeners();
    loadChatSessions();
    loadCurrentChatHistory();
});

// Initialize App
function initializeApp() {
    if (currentTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeBtn.innerHTML = '<i class="fas fa-sun"></i>';
    }
    
    if (sidebarOpen) {
        sidebar.classList.remove('collapsed');
        mainContent.classList.add('sidebar-open');
    } else {
        sidebar.classList.add('collapsed');
        mainContent.classList.remove('sidebar-open');
    }
    
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
        sendBtn.disabled = messageInput.value.trim() === '';
    });
}

// Initialize Event Listeners
function initializeEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.dataset.question;
            messageInput.value = question;
            sendBtn.disabled = false;
            setTimeout(() => sendMessage(), 200);
        });
    });
    
    sidebarToggleBtn.addEventListener('click', toggleSidebar);
    newChatBtn.addEventListener('click', createNewChat);
    clearBtn.addEventListener('click', clearChat);
    themeBtn.addEventListener('click', toggleTheme);
    downloadBtn.addEventListener('click', downloadChat);
}

// Toggle Sidebar
function toggleSidebar() {
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('sidebar-open');
    sidebarOpen = !sidebar.classList.contains('collapsed');
    localStorage.setItem('sidebarOpen', sidebarOpen);
}

// Load Chat Sessions from Database
async function loadChatSessions() {
    try {
        const response = await fetch('/api/sessions');
        const data = await response.json();
        
        if (data.success && data.sessions) {
            displayChatSessions(data.sessions);
        }
    } catch (error) {
        console.error('Error loading sessions:', error);
        chatList.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--text-tertiary); font-size: 13px;">Failed to load chats</div>';
    }
}

// Display Chat Sessions
function displayChatSessions(sessions) {
    if (sessions.length === 0) {
        chatList.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--text-tertiary); font-size: 13px;">No chat history yet</div>';
        return;
    }

    chatList.innerHTML = '';
    sessions.forEach(session => {
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        chatItem.dataset.sessionId = session.session_id;
        
        const preview = session.preview || 'New conversation';
        const timeAgo = formatTimeAgo(session.last_active);
        
        chatItem.innerHTML = `
            <i class="fas fa-message"></i>
            <div class="chat-item-content">
                <div class="chat-item-title">${preview}</div>
                <div class="chat-item-time">${timeAgo}</div>
            </div>
            <button class="chat-item-delete" onclick="deleteSession('${session.session_id}', event)">
                <i class="fas fa-trash"></i>
            </button>
        `;
        
        chatItem.addEventListener('click', (e) => {
            if (!e.target.closest('.chat-item-delete')) {
                loadSession(session.session_id);
            }
        });
        
        chatList.appendChild(chatItem);
    });
}

// Format Time Ago
function formatTimeAgo(timestamp) {
    const now = new Date();
    const past = new Date(timestamp);
    const diffMs = now - past;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return past.toLocaleDateString();
}

// Load Session
async function loadSession(sessionId) {
    try {
        const response = await fetch(`/api/session/${sessionId}`);
        const data = await response.json();
        
        if (data.success) {
            currentSessionId = sessionId;
            messagesContainer.innerHTML = '';
            welcomeScreen.classList.add('hidden');
            
            data.messages.forEach(msg => {
                addMessage(msg.content, msg.role, msg.timestamp, msg.source, false);
            });
            
            document.querySelectorAll('.chat-item').forEach(item => {
                item.classList.remove('active');
                if (item.dataset.sessionId === sessionId) {
                    item.classList.add('active');
                }
            });
            
            showToast('Chat loaded successfully', 'success');
        }
    } catch (error) {
        console.error('Error loading session:', error);
        showToast('Failed to load chat', 'error');
    }
}

// Load Current Chat History
async function loadCurrentChatHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.success && data.messages && data.messages.length > 0) {
            welcomeScreen.classList.add('hidden');
            data.messages.forEach(msg => {
                addMessage(msg.content, msg.role, msg.timestamp, msg.source, false);
            });
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Delete Session
async function deleteSession(sessionId, event) {
    event.stopPropagation();
    
    if (!confirm('Are you sure you want to delete this chat?')) return;
    
    try {
        const response = await fetch(`/api/session/${sessionId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            loadChatSessions();
            
            if (currentSessionId === sessionId) {
                createNewChat();
            }
            
            showToast('Chat deleted successfully', 'success');
        }
    } catch (error) {
        console.error('Error deleting session:', error);
        showToast('Failed to delete chat', 'error');
    }
}

// Send Message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isTyping) return;
    
    if (!welcomeScreen.classList.contains('hidden')) {
        welcomeScreen.classList.add('hidden');
    }
    
    addMessage(message, 'user');
    
    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;
    
    showTyping();
    
    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'X-User-Token': getUserToken()
            },
            body: JSON.stringify({ message, stream: true })
        });

        if (!response.ok || !response.body) {
            throw new Error(`Stream request failed: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalPayload = null;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const frames = buffer.split('\n\n');
            buffer = frames.pop() || '';

            for (const frame of frames) {
                if (!frame.trim()) continue;
                let eventType = 'message';
                let dataStr = '';

                frame.split('\n').forEach((line) => {
                    if (line.startsWith('event:')) {
                        eventType = line.replace('event:', '').trim();
                    } else if (line.startsWith('data:')) {
                        dataStr += line.replace('data:', '').trim();
                    }
                });

                if (!dataStr) continue;

                let payload = {};
                try {
                    payload = JSON.parse(dataStr);
                } catch {
                    payload = { message: dataStr };
                }

                if (eventType === 'status') {
                    const node = payload.node || 'processing';
                    updateTypingStatus(`MediGenius: ${node}...`);
                } else if (eventType === 'error') {
                    addMessage(payload.message || 'Stream error occurred.', 'assistant');
                    showToast('Error occurred', 'error');
                } else if (eventType === 'final') {
                    finalPayload = payload;
                }
            }
        }

        if (finalPayload && finalPayload.success) {
            addMessage(finalPayload.response, 'assistant', finalPayload.timestamp, finalPayload.source);
            showToast('Response received', 'success');
            loadChatSessions();
        } else if (!finalPayload) {
            addMessage('Sorry, no final response was returned.', 'assistant');
            showToast('No final response', 'error');
        } else {
            addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
            showToast('Error occurred', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Connection error. Please check your internet and try again.', 'assistant');
        showToast('Connection error', 'error');
    } finally {
        updateTypingStatus('MediGenius is thinking');
        hideTyping();
    }
}

// Add Message
function addMessage(content, type, timestamp = null, source = null, animate = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    if (animate) messageDiv.style.opacity = '0';
    
    const time = timestamp || new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    if (type === 'user') {
        messageDiv.innerHTML = `
            <div class="message-wrapper">
                <div class="message-avatar">
                    <i class="fas fa-user"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">
                        ${content}
                        <span class="message-time">${time}</span>
                    </div>
                </div>
            </div>
        `;
    } else {
        let footerHtml = '';
        if (source || type === 'assistant') {
            footerHtml = `
                <div class="message-footer">
                    ${source ? `
                        <span class="message-source">
                            <i class="fas fa-database"></i>
                            ${source}
                        </span>
                    ` : ''}
                    <div class="message-actions">
                        <button class="message-action" onclick="copyMessage(this)" title="Copy">
                            <i class="fas fa-copy"></i>
                        </button>
                    </div>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-wrapper">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">${content}</div>
                    <span class="message-time">${time}</span>
                    ${footerHtml}
                </div>
            </div>
        `;
    }
    
    messagesContainer.appendChild(messageDiv);
    
    if (animate) {
        setTimeout(() => {
            messageDiv.style.transition = 'opacity 0.4s ease';
            messageDiv.style.opacity = '1';
        }, 100);
    }
    
    smoothScrollToBottom();
    chatHistory.push({ content, type, timestamp: time, source });
}

// Copy Message
function copyMessage(btn) {
    const messageText = btn.closest('.message-content').querySelector('.message-text').textContent.trim();
    navigator.clipboard.writeText(messageText).then(() => {
        showToast('Message copied', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

// Typing Indicator
function showTyping() {
    isTyping = true;
    updateTypingStatus('MediGenius is thinking');
    typingIndicator.classList.add('active');
    smoothScrollToBottom();
}

function hideTyping() {
    isTyping = false;
    typingIndicator.classList.remove('active');
}

function updateTypingStatus(text) {
    if (typingText) {
        typingText.textContent = text;
    }
}

function smoothScrollToBottom() {
    chatArea.scrollTo({
        top: chatArea.scrollHeight,
        behavior: 'smooth'
    });
}

// Theme Toggle
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
    
    themeBtn.innerHTML = currentTheme === 'light' ? '<i class="fas fa-moon"></i>' : '<i class="fas fa-sun"></i>';
    showToast(`${currentTheme === 'dark' ? 'Dark' : 'Light'} mode activated`, 'success');
}

// Create New Chat
async function createNewChat() {
    try {
        const response = await fetch('/api/new-chat', {
            method: 'POST'
        });
        
        if (response.ok) {
            messagesContainer.innerHTML = '';
            welcomeScreen.classList.remove('hidden');
            chatHistory = [];
            currentSessionId = null;
            
            document.querySelectorAll('.chat-item').forEach(item => {
                item.classList.remove('active');
            });
            
            loadChatSessions();
            showToast('New chat created', 'success');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to create new chat', 'error');
    }
}

// Clear Chat
async function clearChat() {
    if (!confirm('Are you sure you want to clear this conversation?')) return;
    
    try {
        const response = await fetch('/api/clear', {
            method: 'POST'
        });
        
        if (response.ok) {
            messagesContainer.innerHTML = '';
            welcomeScreen.classList.remove('hidden');
            chatHistory = [];
            showToast('Conversation cleared', 'success');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to clear conversation', 'error');
    }
}

// Download Chat
function downloadChat() {
    if (chatHistory.length === 0) {
        showToast('No messages to download', 'error');
        return;
    }
    
    let content = 'MediGenius Chat Export\n';
    content += '='.repeat(50) + '\n\n';
    
    chatHistory.forEach((msg) => {
        content += `[${msg.timestamp}] ${msg.type === 'user' ? 'You' : 'MediGenius'}:\n`;
        content += msg.content + '\n';
        if (msg.source) {
            content += `Source: ${msg.source}\n`;
        }
        content += '\n';
    });
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medigenius-chat-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('Chat downloaded successfully', 'success');
}

// Toast Notifications
function showToast(message, type = 'success') {
    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'info': 'fa-info-circle'
    };
    
    const colors = {
        'success': 'linear-gradient(135deg, #10b981, #059669)',
        'error': 'linear-gradient(135deg, #ef4444, #dc2626)',
        'info': 'linear-gradient(135deg, #3b82f6, #2563eb)'
    };
    
    toast.style.background = colors[type];
    toast.innerHTML = `<i class="fas ${icons[type]}"></i><span>${message}</span>`;
    
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Make functions globally available
window.deleteSession = deleteSession;
window.copyMessage = copyMessage;
