import React, { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  // Use sessionId but don't need to update it, so using just useState
  const sessionId = useState(crypto.randomUUID())[0];

  // Example prompts for the user
  const examplePrompts = [
    "How do I add funds to my account?",
    "What is the process for withdrawing money?",
    "How do quarterly settlements work?",
    "What are the charges for different services?"
  ];

  // Scroll to bottom of chat when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus on input when component mounts
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Function to convert message text to markdown with clickable links
  const renderMessageContent = (content) => {
    // Process the content for markdown and links
    const processedContent = processMarkdownAndLinks(content);
    return { __html: processedContent };
  };

  // Function to process markdown and make links clickable
  const processMarkdownAndLinks = (text) => {
    // Handle links - Convert URLs to clickable links
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    text = text.replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Handle bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Handle italic
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Handle headers
    text = text.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    text = text.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
    text = text.replace(/^# (.*?)$/gm, '<h1>$1</h1>');
    
    // Handle lists
    text = text.replace(/^- (.*?)$/gm, '<li>$1</li>');
    
    // Handle line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
  };

  // Make source URLs clickable
  const processSourceUrl = (sourceUrl) => {
    if (sourceUrl.startsWith('http')) {
      return (
        <a href={sourceUrl} target="_blank" rel="noopener noreferrer">
          {sourceUrl}
        </a>
      );
    }
    return sourceUrl;
  };

  // Process sources to remove duplicates
  const processSourcesForDisplay = (sources) => {
    if (!sources || !sources.length) return [];
    
    const uniqueSources = [];
    const seenUrls = new Set();
    
    sources.forEach(source => {
      // If it's a string, add it directly
      if (typeof source === 'string') {
        if (!seenUrls.has(source)) {
          seenUrls.add(source);
          uniqueSources.push(source);
        }
        return;
      }
      
      // If it's an object, extract the source URL
      const sourceUrl = source.source;
      if (sourceUrl && !seenUrls.has(sourceUrl)) {
        seenUrls.add(sourceUrl);
        
        // Create a formatted display string
        let displayText = `Information from ${sourceUrl}`;
        if (source.title) {
          displayText += ` - ${source.title}`;
        }
        
        uniqueSources.push(displayText);
      }
    });
    
    return uniqueSources;
  };

  const handleSendMessage = async (e) => {
    e?.preventDefault();
    
    if (!inputMessage.trim()) return;
    
    // Add user message to chat
    const userMessage = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    
    try {
      // Call backend API
      const response = await fetch(import.meta.env.VITE_API_URL + '/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: userMessage.content,
          session_id: sessionId
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        // Add bot message to chat
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          isInsuranceRelated: data.is_insurance_related
        }]);
      } else {
        // Add error message
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${data.error || 'Failed to get response from the server'}`,
          error: true
        }]);
      }
    } catch (error) {
      console.error('Error calling API:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error connecting to the server. Please try again later.',
        error: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExamplePrompt = (prompt) => {
    setInputMessage(prompt);
    inputRef.current?.focus();
  };

  return (
    <div className="chat-container">
      <main className="chat-area">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <h2>What can I help with?</h2>
            <div className="example-prompts">
              {examplePrompts.map((prompt, index) => (
                <button 
                  key={index} 
                  className="example-prompt-btn"
                  onClick={() => handleExamplePrompt(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((message, index) => (
              <div 
                key={index} 
                className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'} ${message.error ? 'error-message' : ''}`}
              >
                <div 
                  className="message-content" 
                  dangerouslySetInnerHTML={renderMessageContent(message.content)}
                />
                {message.sources && message.sources.length > 0 && (
                  <div className="sources">
                    <h4>Sources:</h4>
                    <ul>
                      {processSourcesForDisplay(message.sources).map((source, idx) => (
                        <li key={idx}>
                          {processSourceUrl(source)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="message assistant-message loading">
                <div className="loading-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>
      
      <footer className="chat-input-area">
        <form onSubmit={handleSendMessage} className="full-width-form">
          <div className="input-container">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask anything"
              ref={inputRef}
              disabled={isLoading}
              className="expanded-input"
            />
            <button 
              type="submit" 
              className="send-button" 
              disabled={!inputMessage.trim() || isLoading}
            >
              <span className="send-icon">➤</span>
            </button>
          </div>
        </form>
      </footer>
    </div>
  );
}

export default App
