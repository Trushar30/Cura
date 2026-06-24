// frontend/src/components/Auth.jsx
import React, { useState } from 'react';
import { User as UserIcon, Lock, Award, KeyRound, UserPlus } from 'lucide-react';

export default function Auth({ onLoginSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('User');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (!username.trim() || !password.trim()) {
      setError('Please fill in all fields');
      return;
    }
    
    setLoading(true);
    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup';
    
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, role }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }
      
      // Store in localStorage
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('username', data.username);
      localStorage.setItem('role', data.role);
      
      // Callback to app
      onLoginSuccess(data.access_token, data.username, data.role);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-center" style={{ minHeight: 'calc(100vh - 220px)' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '380px', padding: '2.5rem 2rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
          <div style={{
            display: 'inline-flex',
            padding: '0.75rem',
            background: 'var(--accent-amber-light)',
            border: '1px solid rgba(217, 119, 6, 0.15)',
            borderRadius: '4px',
            color: 'var(--accent-amber)',
            marginBottom: '0.75rem'
          }}>
            {isLogin ? <KeyRound size={20} /> : <UserPlus size={20} />}
          </div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '600' }}>{isLogin ? 'Sign In' : 'Register Account'}</h2>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
            {isLogin ? 'Enter your clinical credentials' : 'Join our personalized health analytics network'}
          </p>
        </div>

        {error && (
          <div style={{
            background: 'var(--accent-red-light)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: '4px',
            color: 'var(--accent-red)',
            padding: '0.6rem 0.8rem',
            marginBottom: '1.25rem',
            fontSize: '0.8rem',
            fontWeight: '500'
          }}>
            ⚠️ {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.35rem', fontWeight: '500' }}>
              Username
            </label>
            <div style={{ position: 'relative' }}>
              <UserIcon size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                style={{ paddingLeft: '30px', fontSize: '0.85rem' }}
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.35rem', fontWeight: '500' }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{ paddingLeft: '30px', fontSize: '0.85rem' }}
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn-primary"
            disabled={loading}
            style={{ width: '100%', justifyContent: 'center', marginTop: '0.5rem', fontSize: '0.8rem' }}
          >
            {loading ? 'Processing...' : isLogin ? 'Sign In' : 'Register'}
          </button>
        </form>

        <div style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: '0.8rem' }}>
          <span style={{ color: 'var(--text-secondary)' }}>
            {isLogin ? "New user? " : "Already have an account? "}
          </span>
          <span
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
            }}
            style={{ color: 'var(--text-heading)', cursor: 'pointer', fontWeight: '600', textDecoration: 'underline' }}
          >
            {isLogin ? 'Register' : 'Sign In'}
          </span>
        </div>
      </div>
    </div>
  );
}
