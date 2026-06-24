// frontend/src/components/MABSimulator.jsx
import React, { useState, useEffect } from 'react';
import { RefreshCw, ThumbsUp, ThumbsDown, HelpCircle, Activity, Award } from 'lucide-react';

export default function MABSimulator({ token }) {
  const [arms, setArms] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchBanditRecommendations();
  }, []);

  const fetchBanditRecommendations = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/recommend/bandit/recommendations', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setArms(data);
      } else {
        throw new Error('Failed to load bandit arms');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (drugName, clicked) => {
    try {
      const response = await fetch('/api/recommend/bandit/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ drug_name: drugName, clicked })
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // Log action in local history
        const timestamp = new Date().toLocaleTimeString();
        setHistory(prev => [
          {
            time: timestamp,
            drug: drugName,
            feedback: clicked ? 'Positive (Reward)' : 'Negative (Penalty)',
            newAlpha: data.alpha,
            newBeta: data.beta
          },
          ...prev.slice(0, 5) // keep last 6 entries
        ]);

        // Refresh recommendations to see updated priors and stochastic draws
        await fetchBanditRecommendations();
      } else {
        throw new Error('Failed to submit feedback');
      }
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="glass-panel">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Activity size={20} style={{ color: 'var(--accent-amber)' }} />
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: '600' }}>Thompson Sampling RL Simulator</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                Multi-Armed Bandit (MAB) Online Reinforcement Learning Loop
              </p>
            </div>
          </div>
          <button 
            className="btn-secondary" 
            onClick={fetchBanditRecommendations} 
            disabled={loading}
            style={{ padding: '0.45rem 0.85rem', fontSize: '0.8rem' }}
          >
            <RefreshCw size={13} className={loading ? 'spin' : ''} />
            Re-Sample
          </button>
        </div>

        {error && (
          <div style={{ background: 'var(--accent-red-light)', border: '1px solid rgba(239, 68, 68, 0.2)', color: 'var(--accent-red)', padding: '0.65rem 0.85rem', borderRadius: '4px', marginBottom: '1.25rem', fontSize: '0.85rem' }}>
            ⚠️ {error}
          </div>
        )}

        <div className="grid-cols-2" style={{ gap: '1.5rem', alignItems: 'start' }}>
          {/* Description Card */}
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.55', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <p>
              This simulator demonstrates **Thompson Sampling**, a Bayesian approach to solving the exploration-exploitation dilemma. 
              The recommendation system maintains Beta distribution priors for each medication:
            </p>
            <ul style={{ paddingLeft: '1.1rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <li>
                <span style={{ color: 'var(--text-heading)', fontWeight: '500' }}>Alpha (α):</span> The count of positive feedback clicks plus 1.
              </li>
              <li>
                <span style={{ color: 'var(--text-heading)', fontWeight: '500' }}>Beta (β):</span> The count of negative feedback clicks plus 1.
              </li>
              <li>
                <span style={{ color: 'var(--text-heading)', fontWeight: '500' }}>Thompson Sample:</span> A stochastic draw from the Beta(α, β) distribution. The medication with the highest sampled value is recommended.
              </li>
            </ul>
            <div style={{
              background: 'var(--bg-pill)',
              border: '1px solid var(--border-light)',
              borderRadius: '4px',
              padding: '0.75rem',
              color: 'var(--text-secondary)',
              fontSize: '0.75rem',
              display: 'flex',
              gap: '0.5rem',
              alignItems: 'flex-start',
              marginTop: '0.5rem'
            }}>
              <HelpCircle size={15} style={{ flexShrink: 0, marginTop: '0.1rem', color: 'var(--accent-amber)' }} />
              <span>
                Click **Re-Sample** to draw new stochastic values. Notice how arms with higher positive feedback ratios are favored, while less-explored arms are occasionally chosen to resolve uncertainty.
              </span>
            </div>
          </div>

          {/* Interactive arms */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <h3 style={{ fontSize: '0.95rem', color: 'var(--text-heading)', fontWeight: '600', marginBottom: '0.25rem' }}>
              Top Stochastic Draws
            </h3>
            {loading && arms.length === 0 ? (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Loading simulator parameters...</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {arms.map((arm) => {
                  const successRate = arm.expected_reward * 100;
                  return (
                    <div 
                      key={arm.drug_name} 
                      style={{ 
                        padding: '0.85rem 1rem', 
                        display: 'flex', 
                        flexDirection: 'column', 
                        gap: '0.5rem',
                        background: 'var(--bg-tertiary)',
                        border: '1px solid var(--border-light)',
                        borderRadius: '4px'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: '600', color: 'var(--text-heading)', fontSize: '0.85rem' }}>{arm.drug_name}</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--accent-amber)' }}>
                          Sample: <b>{arm.score.toFixed(3)}</b>
                        </span>
                      </div>

                      {/* Stats */}
                      <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        <span>Helpful (α-1): <b style={{ color: 'var(--accent-green)' }}>{arm.alpha - 1}</b></span>
                        <span>Irrelevant (β-1): <b style={{ color: 'var(--accent-red)' }}>{arm.beta - 1}</b></span>
                        <span>Exp. Reward: <b>{successRate.toFixed(0)}%</b></span>
                      </div>

                      {/* Bar graph representing Beta mean expected reward */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '0.1rem' }}>
                        <div style={{ flex: 1, height: '4px', background: 'var(--bg-input)', borderRadius: '2px', overflow: 'hidden' }}>
                          <div 
                            style={{ 
                              width: `${successRate}%`, 
                              height: '100%', 
                              background: 'var(--text-secondary)',
                              transition: 'width 0.2s ease'
                            }}
                          ></div>
                        </div>
                        
                        {/* Action buttons */}
                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                          <button
                            onClick={() => handleFeedback(arm.drug_name, true)}
                            title="Helpful (Positive Reward)"
                            style={{ 
                              padding: '0.2rem 0.45rem', 
                              background: 'var(--bg-pill)', 
                              border: '1px solid var(--border-light)', 
                              borderRadius: '3px', 
                              color: 'var(--accent-green)', 
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.2rem',
                              fontSize: '0.7rem'
                            }}
                          >
                            <ThumbsUp size={10} /> Upvote
                          </button>
                          <button
                            onClick={() => handleFeedback(arm.drug_name, false)}
                            title="Irrelevant (Negative Penalty)"
                            style={{ 
                              padding: '0.2rem 0.45rem', 
                              background: 'var(--bg-pill)', 
                              border: '1px solid var(--border-light)', 
                              borderRadius: '3px', 
                              color: 'var(--accent-red)', 
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.2rem',
                              fontSize: '0.7rem'
                            }}
                          >
                            <ThumbsDown size={10} /> Downvote
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Simulator Logs */}
      <div className="glass-panel">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.75rem' }}>
          <Award size={16} style={{ color: 'var(--accent-amber)' }} />
          <h3 style={{ fontSize: '0.95rem', fontWeight: '600' }}>Simulation Feedback Log</h3>
        </div>

        {history.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center', padding: '1rem' }}>
            No interactive adjustments logged yet. Click Upvote or Downvote above.
          </p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.8rem' }}>
              <thead>
                <tr style={{ color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '0.5rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Time</th>
                  <th style={{ padding: '0.5rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Medication</th>
                  <th style={{ padding: '0.5rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Action</th>
                  <th style={{ padding: '0.5rem', textAlign: 'right', borderBottom: '1px solid var(--border-light)' }}>Updated Priors (α, β)</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i} style={{ color: 'var(--text-secondary)' }}>
                    <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-light)' }}>{h.time}</td>
                    <td style={{ padding: '0.5rem', color: 'var(--text-heading)', fontWeight: '500', borderBottom: '1px solid var(--border-light)' }}>{h.drug}</td>
                    <td style={{ padding: '0.5rem', color: h.feedback.includes('Reward') ? 'var(--accent-green)' : 'var(--accent-red)', borderBottom: '1px solid var(--border-light)' }}>
                      {h.feedback}
                    </td>
                    <td style={{ padding: '0.5rem', textAlign: 'right', fontWeight: '600', borderBottom: '1px solid var(--border-light)' }}>
                      ({h.newAlpha}, {h.newBeta})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
