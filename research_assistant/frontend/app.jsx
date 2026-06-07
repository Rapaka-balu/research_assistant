const { useState, useEffect, useRef } = React;

function App() {
    const [sessions, setSessions] = useState([]);
    const [activeSession, setActiveSession] = useState(null);
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [activities, setActivities] = useState([]);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    const hasMessages = messages.length > 0 || isLoading;

    // Fetch all sessions on mount
    useEffect(() => {
        fetchSessions();
    }, []);

    // Scroll to bottom when messages or activities change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, activities]);

    // Auto-focus input
    useEffect(() => {
        if (!isLoading && inputRef.current) {
            inputRef.current.focus();
        }
    }, [isLoading, hasMessages]);

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
                            }
                        } else if (eventName === 'update') {
                            const data = JSON.parse(dataStr);
                            tempActivities.push(data);
                            setActivities([...tempActivities]);
                        } else if (eventName === 'final_answer') {
                            const data = JSON.parse(dataStr);
                            setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
                            tempActivities = [];
                            setActivities([]);
                            // Refresh sessions to pick up the new session title
                            fetchSessions();
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

    // Truncate session title for sidebar display
    const truncateTitle = (title, maxLen = 36) => {
        if (!title) return 'Untitled';
        if (title.length <= maxLen) return title;
        return title.substring(0, maxLen) + '…';
    };

    return (
        <div className="app-container">
            {/* Sidebar */}
            <div className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
                <div className="sidebar-header">
                    <h2>Research Sessions</h2>
                    <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)} title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            {sidebarOpen ? (
                                <><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></>
                            ) : (
                                <><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></>
                            )}
                        </svg>
                    </button>
                </div>
                <button className="new-chat-btn" onClick={startNewSession}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="12" y1="5" x2="12" y2="19"></line>
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                    {sidebarOpen && 'New Research'}
                </button>
                
                {sidebarOpen && (
                    <div className="session-list">
                        {sessions.map(session => {
                            const id = typeof session === 'string' ? session : session.thread_id;
                            const title = typeof session === 'string' ? session : (session.title || session.thread_id);
                            return (
                                <div 
                                    key={id} 
                                    className={`session-item ${activeSession === id ? 'active' : ''}`}
                                    onClick={() => loadSession(id)}
                                    title={title}
                                >
                                    <div className="session-icon">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                                        </svg>
                                    </div>
                                    <span className="session-title">{truncateTitle(title)}</span>
                                    <button className="delete-btn" onClick={(e) => deleteSession(e, id)} title="Delete session">✕</button>
                                </div>
                            );
                        })}
                        {sessions.length === 0 && (
                            <div className="no-sessions">No sessions yet</div>
                        )}
                    </div>
                )}
            </div>

            {/* Sidebar toggle for collapsed state */}
            {!sidebarOpen && (
                <button className="sidebar-expand-btn" onClick={() => setSidebarOpen(true)} title="Expand sidebar">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/>
                    </svg>
                </button>
            )}

            {/* Main Chat */}
            <div className="main-chat">
                {!hasMessages ? (
                    /* ── Centered empty state with input ── */
                    <div className="centered-state">
                        <div className="hero-section">
                            <div className="hero-icon">
                                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="url(#grad1)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                    <defs>
                                        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" style={{stopColor:'#3b82f6', stopOpacity:1}} />
                                            <stop offset="100%" style={{stopColor:'#8b5cf6', stopOpacity:1}} />
                                        </linearGradient>
                                    </defs>
                                    <circle cx="11" cy="11" r="8"/>
                                    <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                                </svg>
                            </div>
                            <h1>Research Assistant</h1>
                            <p>Ask any question. Our multi-agent system will search, summarize, analyze, and deliver a cited answer.</p>
                        </div>
                        <form className="centered-input-container" onSubmit={handleSubmit}>
                            <input 
                                ref={inputRef}
                                type="text" 
                                className="chat-input"
                                placeholder="What would you like to research?" 
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                disabled={isLoading}
                                autoFocus
                            />
                            <button type="submit" className="send-btn" disabled={!inputValue.trim() || isLoading}>
                                <svg className="send-icon" viewBox="0 0 24 24">
                                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                                </svg>
                            </button>
                        </form>
                        <div className="suggestion-chips">
                            <button className="chip" onClick={() => setInputValue('What are the latest breakthroughs in quantum computing?')}>
                                🔬 Quantum computing breakthroughs
                            </button>
                            <button className="chip" onClick={() => setInputValue('Compare renewable energy sources for home use')}>
                                ⚡ Renewable energy comparison
                            </button>
                            <button className="chip" onClick={() => setInputValue('Explain the economic impact of AI on the job market')}>
                                🤖 AI's economic impact
                            </button>
                        </div>
                    </div>
                ) : (
                    /* ── Chat messages view ── */
                    <>
                        <div className="chat-history">
                            {messages.map((msg, i) => (
                                <div key={i} className={`message ${msg.role}`}>
                                    <div className="message-role">
                                        {msg.role === 'user' ? (
                                            <><span className="role-icon">👤</span> You</>
                                        ) : (
                                            <><span className="role-icon">🔬</span> Research Assistant</>
                                        )}
                                    </div>
                                    <div className="message-content" dangerouslySetInnerHTML={renderMarkdown(msg.content)} />
                                </div>
                            ))}
                            
                            {/* Activity Stream */}
                            {activities.length > 0 && (
                                <div className="message assistant">
                                    <div className="message-role"><span className="role-icon">⚙️</span> Agent Activity</div>
                                    <div className="agent-activity">
                                        {activities.map((act, i) => (
                                            <div key={i} className="activity-item">
                                                <span className="activity-agent">{act.agent}</span>
                                                <span className="activity-status">{act.status}</span>
                                            </div>
                                        ))}
                                        <div className="activity-spinner">
                                            <div className="spinner"></div>
                                            <span>Working...</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Bottom input when in chat mode */}
                        <div className="input-area">
                            <form className="input-container" onSubmit={handleSubmit}>
                                <input 
                                    ref={inputRef}
                                    type="text" 
                                    className="chat-input"
                                    placeholder="Ask a follow-up question..." 
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
                    </>
                )}
            </div>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
