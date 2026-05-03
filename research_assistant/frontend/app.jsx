const { useState, useEffect, useRef } = React;

function App() {
    const [sessions, setSessions] = useState([]);
    const [activeSession, setActiveSession] = useState(null);
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [activities, setActivities] = useState([]);
    
    const messagesEndRef = useRef(null);

    // Fetch all sessions on mount
    useEffect(() => {
        fetchSessions();
    }, []);

    // Scroll to bottom when messages or activities change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, activities]);

    const fetchSessions = async () => {
        try {
            const res = await fetch('/api/sessions');
            const data = await res.json();
            setSessions(data.sessions || []);
        } catch (e) {
            console.error("Failed to fetch sessions", e);
        }
    };

    const loadSession = async (threadId) => {
        setActiveSession(threadId);
        setActivities([]);
        setMessages([]);
        try {
            const res = await fetch(`/api/sessions/${threadId}`);
            const data = await res.json();
            
            const loadedMessages = [];
            if (data.query) {
                loadedMessages.push({ role: 'user', content: data.query });
            }
            if (data.final_answer) {
                loadedMessages.push({ role: 'assistant', content: data.final_answer });
            }
            setMessages(loadedMessages);
        } catch (e) {
            console.error("Failed to load session", e);
        }
    };

    const startNewSession = () => {
        setActiveSession(null);
        setMessages([]);
        setActivities([]);
        setInputValue('');
    };

    const deleteSession = async (e, threadId) => {
        e.stopPropagation();
        if(!confirm('Are you sure you want to delete this session?')) return;
        try {
            await fetch(`/api/sessions/${threadId}`, { method: 'DELETE' });
            if (activeSession === threadId) {
                startNewSession();
            }
            fetchSessions();
        } catch (error) {
            console.error("Failed to delete", error);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!inputValue.trim() || isLoading) return;

        const query = inputValue.trim();
        setInputValue('');
        setIsLoading(true);
        setActivities([]);
        
        // Optimistically add user message
        setMessages([{ role: 'user', content: query }]);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, thread_id: activeSession })
            });

            if (!response.body) throw new Error('ReadableStream not supported');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let done = false;
            
            let tempActivities = [];

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                done = readerDone;
                if (value) {
                    const chunk = decoder.decode(value, { stream: true });
                    const events = chunk.split('\n\n');
                    
                    for (let eventStr of events) {
                        if (!eventStr.trim()) continue;
                        
                        const lines = eventStr.split('\n');
                        const eventLine = lines.find(l => l.startsWith('event:'));
                        const dataLine = lines.find(l => l.startsWith('data:'));
                        
                        if (!eventLine || !dataLine) continue;
                        
                        const eventName = eventLine.replace('event:', '').trim();
                        const dataStr = dataLine.replace('data:', '').trim();
                        
                        if (eventName === 'thread_id') {
                            if (!activeSession) {
                                setActiveSession(dataStr);
                                fetchSessions(); // Refresh list to show new session
                            }
                        } else if (eventName === 'update') {
                            const data = JSON.parse(dataStr);
                            tempActivities.push(data);
                            setActivities([...tempActivities]);
                        } else if (eventName === 'final_answer') {
                            const data = JSON.parse(dataStr);
                            setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
                            tempActivities = []; // clear activities once we have answer
                            setActivities([]);
                        } else if (eventName === 'error') {
                            const data = JSON.parse(dataStr);
                            setMessages(prev => [...prev, { role: 'assistant', content: `**Error:** ${data.error}` }]);
                            tempActivities = [];
                            setActivities([]);
                        } else if (eventName === 'close') {
                            setIsLoading(false);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error in chat:', error);
            setMessages(prev => [...prev, { role: 'assistant', content: '**Error:** Failed to communicate with server.' }]);
            setIsLoading(false);
            setActivities([]);
        }
    };

    const renderMarkdown = (text) => {
        return { __html: marked.parse(text) };
    };

    return (
        <div className="app-container">
            {/* Sidebar */}
            <div className="sidebar">
                <h2>Research Sessions</h2>
                <button className="new-chat-btn" onClick={startNewSession}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="12" y1="5" x2="12" y2="19"></line>
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                    New Research
                </button>
                
                <div className="session-list">
                    {sessions.map(id => (
                        <div 
                            key={id} 
                            className={`session-item ${activeSession === id ? 'active' : ''}`}
                            onClick={() => loadSession(id)}
                        >
                            <span style={{overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>{id}</span>
                            <button className="delete-btn" onClick={(e) => deleteSession(e, id)}>✕</button>
                        </div>
                    ))}
                </div>
            </div>

            {/* Main Chat */}
            <div className="main-chat">
                <div className="chat-history">
                    {messages.length === 0 && !isLoading && (
                        <div className="empty-state">
                            <h1>Research Assistant</h1>
                            <p>Ask a complex question. The multi-agent system will search the web, summarize findings, and synthesize an answer.</p>
                        </div>
                    )}
                    
                    {messages.map((msg, i) => (
                        <div key={i} className={`message ${msg.role}`}>
                            <div className="message-role">{msg.role}</div>
                            <div className="message-content" dangerouslySetInnerHTML={renderMarkdown(msg.content)} />
                        </div>
                    ))}
                    
                    {/* Activity Stream */}
                    {activities.length > 0 && (
                        <div className="message assistant">
                            <div className="message-role">Agent Activity</div>
                            <div className="agent-activity">
                                {activities.map((act, i) => (
                                    <div key={i} className="activity-item">
                                        <span className="activity-agent">[{act.agent}]</span>
                                        <span className="activity-status">{act.status}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Form */}
                <div className="input-area">
                    <form className="input-container" onSubmit={handleSubmit}>
                        <input 
                            type="text" 
                            className="chat-input"
                            placeholder="Type your research query..." 
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            disabled={isLoading}
                        />
                        <button type="submit" className="send-btn" disabled={!inputValue.trim() || isLoading}>
                            <svg className="send-icon" viewBox="0 0 24 24">
                                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                            </svg>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
