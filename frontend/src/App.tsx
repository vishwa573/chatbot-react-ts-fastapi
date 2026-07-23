import { useState, FormEvent, useRef, useEffect } from 'react';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface Conversation {
  id: string;
  title: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Phase 4: Conversation State
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initial load: Fetch sidebar conversations
  const fetchConversations = async () => {
    try {
      const res = await fetch('http://localhost:8000/conversations');
      const data = await res.json();
      setConversations(data);
    } catch (error) {
      console.error("Failed to fetch conversations", error);
    }
  };

  useEffect(() => {
    fetchConversations();
  }, []);

  // When active conversation changes, fetch its message history
  useEffect(() => {
    if (!activeConversationId) {
      setMessages([]); // Clear chat for "New Chat"
      return;
    }

    const fetchMessages = async () => {
      try {
        const res = await fetch(`http://localhost:8000/conversations/${activeConversationId}/messages`);
        const data = await res.json();
        setMessages(data);
      } catch (error) {
        console.error("Failed to fetch messages", error);
      }
    };

    fetchMessages();
  }, [activeConversationId]);

  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          conversation_id: activeConversationId,
          content: userMessage.content 
        }),
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      if (!response.body) throw new Error('No response body');

      setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // 1. Create a temporary variable to hold the ID instead of setting state immediately
      let newConvId: string | null = null; 

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').trim();
            if (!dataStr) continue;
            
            const data = JSON.parse(dataStr);
            
            // 2. Capture the ID from the backend, but DO NOT trigger the useEffect yet
            if (!activeConversationId && data.conversation_id && !newConvId) {
              newConvId = data.conversation_id;
            }
            
            setMessages((prev) => {
              const newMessages = [...prev];
              const lastIndex = newMessages.length - 1;
              newMessages[lastIndex] = {
                ...newMessages[lastIndex],
                content: newMessages[lastIndex].content + data.content
              };
              return newMessages;
            });
          }
        }
      }

      // 3. NOW that the stream is completely done (and the backend has saved everything),
      // we can safely update the sidebar and active session.
      if (newConvId) {
        setActiveConversationId(newConvId);
        fetchConversations();
      }

    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error: Could not reach backend.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Section */}
      <aside className="sidebar">
        <button 
          className="new-chat-btn" 
          onClick={() => setActiveConversationId(null)}
        >
          + New Chat
        </button>
        <div className="conversation-list">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              className={`conv-item ${activeConversationId === conv.id ? 'active' : ''}`}
              onClick={() => setActiveConversationId(conv.id)}
            >
              {conv.title}
            </button>
          ))}
        </div>
      </aside>

      {/* Main Chat Section */}
      <main className="chat-container">
        <header className="chat-header">
          <h2>{activeConversationId ? 'Chat Session' : 'New Chat'}</h2>
        </header>

        <div className="messages-feed">
          {messages.length === 0 ? (
            <p className="empty-state">Start a conversation!</p>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`message-bubble ${
                  msg.role === 'user' ? 'user' : 'assistant'
                }`}
              >
                <strong>{msg.role === 'user' ? 'You' : 'AI'}:</strong>
                <p>{msg.content}</p>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className="chat-input-form" onSubmit={handleSendMessage}>
          <input
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? '...' : 'Send'}
          </button>
        </form>
      </main>
    </div>
  );
}

export default App;