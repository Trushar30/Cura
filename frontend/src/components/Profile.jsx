// frontend/src/components/Profile.jsx
import React, { useState, useEffect } from 'react';
import { UserCheck, Navigation, Settings } from 'lucide-react';

const CHRONIC_OPTIONS = ["Diabetes ", "Gerd", "Allergy", "Hypothyroidism", "Hypertension ", "Arthritis", "Migraine"];

export default function Profile({ token }) {
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('Male');
  const [location, setLocation] = useState('United States');
  const [selectedConditions, setSelectedConditions] = useState([]);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const response = await fetch('/api/profile', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setAge(data.age || '');
        setGender(data.gender || 'Male');
        setLocation(data.location || 'United States');
        
        if (data.chronic_conditions) {
          const list = data.chronic_conditions.split(',').map(s => s.trim()).filter(Boolean);
          setSelectedConditions(list);
        }
      }
    } catch (err) {
      console.error("Error loading profile:", err);
    }
  };

  const handleConditionChange = (cond) => {
    if (selectedConditions.includes(cond)) {
      setSelectedConditions(selectedConditions.filter(c => c !== cond));
    } else {
      setSelectedConditions([...selectedConditions, cond]);
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');
    
    const chronic_conditions = selectedConditions.join(', ');
    
    try {
      const response = await fetch('/api/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          age: age ? parseInt(age) : null,
          gender,
          location,
          chronic_conditions
        })
      });
      
      if (response.ok) {
        setMessage('✅ Profile updated successfully!');
      } else {
        throw new Error('Failed to update profile');
      }
    } catch (err) {
      setMessage(`❌ Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto' }}>
      <div className="glass-panel">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.75rem' }}>
          <Settings size={20} style={{ color: 'var(--accent-amber)' }} />
          <h2>Manage Health Profile & Preferences</h2>
        </div>

        {message && (
          <div style={{
            background: message.startsWith('✅') ? 'var(--accent-green-light)' : 'var(--accent-red-light)',
            border: message.startsWith('✅') ? '1px solid rgba(16, 185, 129, 0.2)' : '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: '4px',
            color: message.startsWith('✅') ? 'var(--accent-green)' : 'var(--accent-red)',
            padding: '0.65rem 0.85rem',
            marginBottom: '1.25rem',
            fontWeight: '500',
            fontSize: '0.85rem'
          }}>
            {message}
          </div>
        )}

        <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="grid-cols-2" style={{ gap: '1.25rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: '500' }}>
                Age
              </label>
              <input
                type="number"
                min="0"
                max="120"
                placeholder="e.g. 35"
                value={age}
                onChange={(e) => setAge(e.target.value)}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: '500' }}>
                Gender
              </label>
              <select value={gender} onChange={(e) => setGender(e.target.value)}>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: '500' }}>
              Region / Location
            </label>
            <div style={{ position: 'relative' }}>
              <Navigation size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="e.g. New York, United States"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                style={{ paddingLeft: '32px' }}
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: '500' }}>
              Chronic Conditions & Predispositions
            </label>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              Select chronic conditions below to filter and contextualize your clinical drug reviews:
            </p>
            
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {CHRONIC_OPTIONS.map((cond) => {
                const isSelected = selectedConditions.includes(cond);
                return (
                  <button
                    key={cond}
                    type="button"
                    onClick={() => handleConditionChange(cond)}
                    style={{
                      padding: '0.35rem 0.75rem',
                      borderRadius: '3px',
                      fontSize: '0.8rem',
                      cursor: 'pointer',
                      border: '1px solid',
                      borderColor: isSelected ? 'var(--text-secondary)' : 'var(--border-light)',
                      background: isSelected ? 'var(--bg-pill-selected)' : 'transparent',
                      color: isSelected ? 'var(--text-heading)' : 'var(--text-secondary)',
                      transition: 'var(--transition-fast)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.35rem'
                    }}
                  >
                    {isSelected && <span className="glow-pill"></span>}
                    {cond.trim()}
                  </button>
                );
              })}
            </div>
          </div>

          <div style={{ marginTop: '0.5rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button type="submit" className="btn-primary" disabled={loading} style={{ fontSize: '0.8rem' }}>
              <UserCheck size={14} />
              {loading ? 'Saving Changes...' : 'Save Profile'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
