// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import { LogOut, Stethoscope, Network, Activity, Settings, BarChart3, HeartPulse, User, Sun, Moon } from 'lucide-react';
import Auth from './components/Auth';
import Profile from './components/Profile';
import SymptomPredictor from './components/SymptomPredictor';
import GraphVisualizer from './components/GraphVisualizer';
import MABSimulator from './components/MABSimulator';
import Dashboard from './components/Dashboard';

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [username, setUsername] = useState(localStorage.getItem('username') || '');
  const [role, setRole] = useState(localStorage.getItem('role') || '');
  const [activeTab, setActiveTab] = useState('predictor');
  const [selectedDisease, setSelectedDisease] = useState('');
  const [checkingAuth, setCheckingAuth] = useState(false);
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  useEffect(() => {
    if (token) {
      verifyToken();
    }
  }, [token]);

  const verifyToken = async () => {
    setCheckingAuth(true);
    try {
      const response = await fetch('/api/auth/me', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) {
        handleLogout();
      } else {
        const data = await response.json();
        setUsername(data.username);
        setRole(data.role);
        localStorage.setItem('username', data.username);
        localStorage.setItem('role', data.role);
      }
    } catch (err) {
      console.error('Token validation failed:', err);
    } finally {
      setCheckingAuth(false);
    }
  };

  const handleLoginSuccess = (newToken, newUsername, newRole) => {
    setToken(newToken);
    setUsername(newUsername);
    setRole(newRole);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    localStorage.removeItem('role');
    setToken('');
    setUsername('');
    setRole('');
    setSelectedDisease('');
    setActiveTab('predictor');
  };

  if (!token) {
    return (
      <div className="container" style={{ paddingTop: '5rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.4rem', color: 'var(--text-heading)', letterSpacing: '-0.03em' }}>
            Cura
          </h1>
          <p style={{ color: 'var(--accent-amber)', fontSize: '0.85rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.12em' }}>
            Personalized Clinical Recommendation System
          </p>
        </div>
        <Auth onLoginSuccess={handleLoginSuccess} />
      </div>
    );
  }

  if (checkingAuth) {
    return (
      <div className="flex-center" style={{ minHeight: '100vh' }}>
        <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
          <div className="spin" style={{ display: 'inline-block', width: '20px', height: '20px', border: '2px solid var(--border-light)', borderTopColor: 'var(--accent-amber)', borderRadius: '50%', marginBottom: '1rem' }}></div>
          <div style={{ fontSize: '0.85rem' }}>Validating secure credentials...</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Navigation Header */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="logo" style={{ cursor: 'pointer' }} onClick={() => setActiveTab('predictor')}>
            <HeartPulse size={16} style={{ color: 'var(--accent-amber)' }} />
            <span>Cura</span>
          </div>

          <div className="nav-links">
            <span 
              className={`nav-link ${activeTab === 'predictor' ? 'active' : ''}`}
              onClick={() => setActiveTab('predictor')}
            >
              <Stethoscope size={13} />
              Symptom Predictor
            </span>
            <span 
              className={`nav-link ${activeTab === 'graph' ? 'active' : ''}`}
              onClick={() => setActiveTab('graph')}
            >
              <Network size={13} />
              Knowledge Graph
            </span>
            <span 
              className={`nav-link ${activeTab === 'bandit' ? 'active' : ''}`}
              onClick={() => setActiveTab('bandit')}
            >
              <Activity size={13} />
              Bandit Simulator
            </span>
            <span 
              className={`nav-link ${activeTab === 'profile' ? 'active' : ''}`}
              onClick={() => setActiveTab('profile')}
            >
              <Settings size={13} />
              Profile
            </span>

            {/* Restricted Analytics tab for Admin/Analyst */}
            {(role === 'Admin' || role === 'Analyst') && (
              <span 
                className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`}
                onClick={() => setActiveTab('dashboard')}
              >
                <BarChart3 size={13} />
                Analytics Console
              </span>
            )}
          </div>

          {/* User Details & Logout */}
          <div className="user-section">
            <div className="user-details">
              <User size={12} style={{ color: 'var(--text-secondary)' }} />
              <span style={{ color: 'var(--text-heading)', fontWeight: '500' }}>{username}</span>
              <span className={`role-badge ${role === 'Admin' ? 'badge-admin' : role === 'Analyst' ? 'badge-analyst' : 'badge-user'}`}>
                {role}
              </span>
            </div>

            {/* Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
              style={{
                background: 'var(--bg-pill)',
                border: '1px solid var(--border-light)',
                color: 'var(--text-primary)',
                padding: '0.35rem 0.5rem',
                borderRadius: '4px',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'var(--transition-fast)'
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-pill-hover)'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'var(--bg-pill)'}
            >
              {theme === 'dark' ? <Sun size={12} /> : <Moon size={12} />}
            </button>
            
            <button 
              onClick={handleLogout}
              style={{
                background: 'var(--bg-pill)',
                border: '1px solid var(--border-light)',
                color: 'var(--accent-red)',
                padding: '0.35rem 0.65rem',
                borderRadius: '4px',
                fontSize: '0.75rem',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                fontWeight: '500',
                transition: 'var(--transition-fast)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'var(--accent-red-light)';
                e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.2)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'var(--bg-pill)';
                e.currentTarget.style.borderColor = 'var(--border-light)';
              }}
            >
              <LogOut size={12} />
              Logout
            </button>
          </div>
        </div>
      </nav>

      {/* Main Body Content */}
      <main className="container" style={{ flex: 1 }}>
        {activeTab === 'predictor' && (
          <SymptomPredictor token={token} onDiseaseSelected={setSelectedDisease} />
        )}
        {activeTab === 'graph' && (
          <GraphVisualizer token={token} selectedDisease={selectedDisease} />
        )}
        {activeTab === 'bandit' && (
          <MABSimulator token={token} />
        )}
        {activeTab === 'profile' && (
          <Profile token={token} />
        )}
        {activeTab === 'dashboard' && (
          <Dashboard token={token} userRole={role} />
        )}
      </main>
    </div>
  );
}
