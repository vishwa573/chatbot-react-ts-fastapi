import { useState, FormEvent, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
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
  // Phase 4.5: Dictionary State for Background Generation
  // Key = conversation_id, Value = array of messages
  const [chatsData, setChatsData] = useState<Record<string, Message[]>>({});
  
  const [input, setInput] = useState('');
  const [loadingChats, setLoadingChats] = useState<Record<string, boolean>>({});
  
  // Holds the network kill switches for every active generation
  const abortControllersRef = useRef<Record<string, AbortController>>({});  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<Record<string, string[]>>({});

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Auto-scroll whenever the active chat's data changes
  useEffect(() => {
    scrollToBottom();
  }, [chatsData, activeConversationId]);

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
  const handleDeleteConversation = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation(); // Prevents the click from selecting the chat
    
    try {
      const res = await fetch(`http://localhost:8000/conversations/${id}`, {
        method: 'DELETE',
      });
      
      if (res.ok) {
        // 1. Remove from sidebar list
        setConversations(prev => prev.filter(c => c.id !== id));
        
        // 2. Clear the main screen if the deleted chat was currently open
        if (activeConversationId === id) {
          setActiveConversationId(null);
        }
        
        // 3. Remove from our background dictionary to free up RAM
        setChatsData(prev => {
          const newData = { ...prev };
          delete newData[id];
          return newData;
        });
      }
    } catch (error) {
      console.error("Failed to delete conversation", error);
    }
  };
  useEffect(() => {
    fetchConversations();
  }, []);

  // Fetch messages ONLY if we haven't already loaded them into our dictionary
  useEffect(() => {
    if (!activeConversationId) return;

    if (!chatsData[activeConversationId]) {
      const fetchMessages = async () => {
        try {
          const res = await fetch(`http://localhost:8000/conversations/${activeConversationId}/messages`);
          const data = await res.json();
          setChatsData((prev) => ({ ...prev, [activeConversationId]: data }));
        } catch (error) {
          console.error("Failed to fetch messages", error);
        }
      };
      fetchMessages();
    }
  }, [activeConversationId, chatsData]);

  const handleStopGenerating = () => {
    const trackingId = activeConversationId || "new";
    if (abortControllersRef.current[trackingId]) {
      abortControllersRef.current[trackingId].abort();
    }
  };
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Only PDF files are supported for RAG indexing.');
      return;
    }

    setIsUploading(true);

    // If starting on a fresh "New Chat", generate a UUID now so the file and future messages share the same ID
    let currentConvId = activeConversationId;
    if (!currentConvId) {
      currentConvId = crypto.randomUUID();
      setActiveConversationId(currentConvId);
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('conversation_id', currentConvId);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to upload document');
      }

      const data = await response.json();

      // Track uploaded filename for UI display
      setUploadedFiles((prev) => ({
        ...prev,
        [currentConvId]: [...(prev[currentConvId] || []), data.filename],
      }));

      fetchConversations();
    } catch (error: any) {
      console.error('File upload error:', error);
      alert(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    
    // Use "new" as a temporary key if we don't have a UUID yet
    const trackingId = activeConversationId || "new";
    if (!input.trim() || loadingChats[trackingId]) return;

    const userText = input;
    setInput('');
    
    // 1. Lock this specific chat and create its kill switch
    setLoadingChats(prev => ({ ...prev, [trackingId]: true }));
    const abortController = new AbortController();
    abortControllersRef.current[trackingId] = abortController;

    const initialConvId = activeConversationId;

    if (initialConvId) {
      setChatsData((prev) => ({
        ...prev,
        [initialConvId]: [
          ...(prev[initialConvId] || []),
          { role: 'user', content: userText },
          { role: 'assistant', content: '' }
        ]
      }));
    }

    let streamConvId = initialConvId;

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          conversation_id: initialConvId,
          content: userText 
        }),
        signal: abortController.signal, // 2. Attach the kill switch
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

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
            
            if (!streamConvId && data.conversation_id) {
              streamConvId = data.conversation_id;
              
              // 3. Migrate the kill switch and loading state from "new" to the real UUID
              abortControllersRef.current[streamConvId as string] = abortControllersRef.current["new"];
              delete abortControllersRef.current["new"];
              
              setLoadingChats(prev => {
                const newState = { ...prev, [streamConvId as string]: true };
                delete newState["new"];
                return newState;
              });
              
              setChatsData((prev) => ({
                ...prev,
                [streamConvId as string]: [
                  { role: 'user', content: userText },
                  { role: 'assistant', content: data.content }
                ]
              }));
              
              setActiveConversationId((currentId) => 
                currentId === null ? (streamConvId as string) : currentId
              );
              
              fetchConversations();
              continue;
            }
            
            if (streamConvId) {
              setChatsData((prev) => {
                const chatMessages = prev[streamConvId as string] || [];
                const lastIndex = chatMessages.length - 1;
                
                const updatedMessages = [...chatMessages];
                updatedMessages[lastIndex] = {
                  ...updatedMessages[lastIndex],
                  content: updatedMessages[lastIndex].content + data.content
                };
                
                return { ...prev, [streamConvId as string]: updatedMessages };
              });
            }
          }
        }
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log(`Stream ${streamConvId || 'new'} stopped by user.`);
      } else {
        console.error('Failed to send message:', error);
      }
    } finally {
      // 4. Clean up the lock and the kill switch when done
      const finalId = streamConvId || trackingId;
      setLoadingChats(prev => ({ ...prev, [finalId as string]: false }));
      delete abortControllersRef.current[finalId as string];
    }
  };

  // Derive the messages to show on screen based on what is active
  const activeMessages = activeConversationId ? (chatsData[activeConversationId] || []) : [];

  return (
    <div className="app-container">
      <aside className="sidebar">
        <button 
          className="new-chat-btn" 
          onClick={() => setActiveConversationId(null)}
        >
          + New Chat
        </button>
        <div className="conversation-list">
          {conversations.map((conv) => (
            <div 
              key={conv.id}
              className={`conv-item-container ${activeConversationId === conv.id ? 'active' : ''}`}
              onClick={() => setActiveConversationId(conv.id)}
            >
              <button className="conv-item-title">
                {conv.title}
              </button>
              <button 
                className="delete-conv-btn" 
                onClick={(e) => handleDeleteConversation(e, conv.id)}
                title="Delete chat"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="chat-container">
        <header className="chat-header">
          <h2>{activeConversationId ? 'Chat Session' : 'New Chat'}</h2>
        </header>

        <div className="messages-feed">
          {activeMessages.length === 0 ? (
            <p className="empty-state">Start a conversation!</p>
          ) : (
            activeMessages.map((msg, index) => (
              <div
                key={index}
                className={`message-bubble ${
                  msg.role === 'user' ? 'user' : 'assistant'
                }`}
              >
                <strong>{msg.role === 'user' ? 'You' : 'AI'}:</strong>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Document Tags Indicator */}
        {activeConversationId && uploadedFiles[activeConversationId]?.length > 0 && (
          <div className="uploaded-docs-bar">
            <span>Indexed Documents:</span>
            {uploadedFiles[activeConversationId].map((name, idx) => (
              <span key={idx} className="doc-chip">
                📄 {name}
              </span>
            ))}
          </div>
        )}

        {/* Hidden File Input */}
        <input
          type="file"
          ref={fileInputRef}
          accept=".pdf"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />

        <form className="chat-input-form" onSubmit={handleSendMessage}>
          <button
            type="button"
            className="upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading || (loadingChats[activeConversationId || "new"] || false)}
            title="Upload PDF for RAG"
          >
            {isUploading ? '⏳' : '📎'}
          </button>

          <input
            type="text"
            placeholder={
              isUploading
                ? "Parsing & vectorizing document..."
                : "Type a message or ask about your uploaded PDF..."
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isUploading || (loadingChats[activeConversationId || "new"] || false)}
          />

          {loadingChats[activeConversationId || "new"] ? (
            <button type="button" className="stop-btn" onClick={handleStopGenerating}>
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={isUploading || !input.trim() || (loadingChats[activeConversationId || "new"] || false)}
            >
              Send
            </button>
          )}
        </form>
      </main>
    </div>
  );
}

export default App;