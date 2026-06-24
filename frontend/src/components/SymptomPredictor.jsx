// frontend/src/components/SymptomPredictor.jsx
import React, { useState } from 'react';
import { Activity, Stethoscope, Search, HeartPulse, Check, Pill, ListRestart, ShieldCheck } from 'lucide-react';

const COMMON_SYMPTOMS = [
  "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
  "shivering", "chills", "joint_pain", "stomach_pain", "acidity", 
  "ulcers_on_tongue", "vomiting", "fatigue", "weight_gain", "anxiety", 
  "weight_loss", "restlessness", "lethargy", "cough", "high_fever", 
  "breathlessness", "sweating", "dehydration", "indigestion", "headache", 
  "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", 
  "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", 
  "chest_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs", 
  "excessive_hunger", "stiff_neck", "swelling_joints", "muscle_weakness", 
  "loss_of_balance", "depression", "irritability", "muscle_pain"
];

export default function SymptomPredictor({ token, onDiseaseSelected }) {
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [recomLoading, setRecomLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedbackMsg, setFeedbackMsg] = useState({});

  const toggleSymptom = (sym) => {
    if (selectedSymptoms.includes(sym)) {
      setSelectedSymptoms(selectedSymptoms.filter(s => s !== sym));
    } else {
      setSelectedSymptoms([...selectedSymptoms, sym]);
    }
  };

  const handlePredict = async () => {
    if (selectedSymptoms.length === 0) {
      setError('Please select at least one symptom');
      return;
    }
    
    setLoading(true);
    setPrediction(null);
    setRecommendations([]);
    setError('');
    
    try {
      const response = await fetch('/api/recommend/predict-disease', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ symptoms: selectedSymptoms }),
      });
      
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed');
      
      setPrediction(data);
      
      if (onDiseaseSelected) {
        onDiseaseSelected(data.matched_disease);
      }
      
      fetchDrugRecommendations(data.matched_disease);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchDrugRecommendations = async (diseaseName) => {
    setRecomLoading(true);
    try {
      const response = await fetch(`/api/recommend/recommendations?disease=${encodeURIComponent(diseaseName)}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (response.ok) {
        setRecommendations(data.recommendations);
      }
    } catch (err) {
      console.error("Failed to load drug recommendations", err);
    } finally {
      setRecomLoading(false);
    }
  };

  const handleBanditFeedback = async (drugName, clicked) => {
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
        setFeedbackMsg(prev => ({
          ...prev,
          [drugName]: clicked ? '👍 Helpful (logged)' : '👎 Irrelevant (logged)'
        }));
      }
    } catch (err) {
      console.error(err);
    }
  };

  const filteredSymptoms = COMMON_SYMPTOMS.filter(sym => 
    sym.replace('_', ' ').toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="glass-panel">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.75rem' }}>
          <Stethoscope size={20} style={{ color: 'var(--accent-amber)' }} />
          <h2>Clinical Symptom & Diagnosis Engine</h2>
        </div>

        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.25rem', fontSize: '0.85rem' }}>
          Select presenting symptoms below. The diagnostic engine maps input vectors against clinical profiles, predicts disease states, and computes comparative scoring for indicated treatments.
        </p>

        <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Search size={15} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search symptoms (e.g. fever, headache)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ paddingLeft: '32px', fontSize: '0.85rem', padding: '0.5rem 0.5rem 0.5rem 32px' }}
            />
          </div>
          <button
            className="btn-primary"
            onClick={handlePredict}
            disabled={loading || selectedSymptoms.length === 0}
            style={{ padding: '0.5rem 1rem', fontSize: '0.8rem' }}
          >
            <Activity size={14} className={loading ? 'spin' : ''} />
            {loading ? 'Analyzing...' : 'Analyze Symptoms'}
          </button>
        </div>

        {/* Selected List */}
        {selectedSymptoms.length > 0 && (
          <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border-light)', marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '600', letterSpacing: '0.02em' }}>SELECTED SYMPTOMS ({selectedSymptoms.length})</span>
              <button 
                onClick={() => setSelectedSymptoms([])}
                style={{ background: 'none', border: 'none', color: 'var(--accent-red)', fontSize: '0.75rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.2rem' }}
              >
                <ListRestart size={12} /> Reset
              </button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
              {selectedSymptoms.map(sym => (
                <span 
                  key={sym} 
                  onClick={() => toggleSymptom(sym)}
                  style={{
                    padding: '0.25rem 0.6rem',
                    background: 'var(--bg-pill)',
                    border: '1px solid var(--border-light)',
                    borderRadius: '3px',
                    fontSize: '0.75rem',
                    color: 'var(--text-heading)',
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.25rem'
                  }}
                >
                  {sym.replace('_', ' ')} &times;
                </span>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div style={{ background: 'var(--accent-red-light)', border: '1px solid rgba(239, 68, 68, 0.2)', color: 'var(--accent-red)', padding: '0.65rem 0.85rem', borderRadius: '4px', marginBottom: '1.25rem', fontSize: '0.85rem' }}>
            ⚠️ {error}
          </div>
        )}

        {/* Scrollable list of symptom tags */}
        <div style={{ maxHeight: '180px', overflowY: 'auto', display: 'flex', flexWrap: 'wrap', gap: '0.4rem', padding: '0.5rem', background: 'var(--bg-input)', borderRadius: '4px', border: '1px solid var(--border-light)' }}>
          {filteredSymptoms.map(sym => {
            const isSelected = selectedSymptoms.includes(sym);
            return (
              <button
                key={sym}
                onClick={() => toggleSymptom(sym)}
                style={{
                  padding: '0.35rem 0.65rem',
                  borderRadius: '3px',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  border: '1px solid',
                  borderColor: isSelected ? 'var(--text-secondary)' : 'var(--border-light)',
                  background: isSelected ? 'var(--bg-pill-selected)' : 'transparent',
                  color: isSelected ? 'var(--text-heading)' : 'var(--text-secondary)',
                  transition: 'var(--transition-fast)'
                }}
              >
                {sym.replace('_', ' ')}
              </button>
            );
          })}
        </div>
      </div>

      {/* Prediction Result details */}
      {prediction && (
        <div className="grid-cols-2" style={{ gap: '1.5rem' }}>
          {/* Diagnostic Card */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className={`role-badge ${prediction.risk_level === 'High' ? 'badge-admin' : prediction.risk_level === 'Medium' ? 'badge-analyst' : 'badge-user'}`}>
                {prediction.risk_level} Risk
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                Confidence Score: <b>{Math.round(prediction.match_score * 100)}%</b>
              </span>
            </div>
            
            <h3 style={{ fontSize: '1.25rem', marginTop: '0.25rem', color: 'var(--text-heading)' }}>
              Diagnosis: <span style={{ color: 'var(--accent-amber)' }}>{prediction.matched_disease}</span>
            </h3>
            
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>{prediction.description}</p>

            <div style={{ marginTop: '0.5rem' }}>
              <h4 style={{ fontSize: '0.8rem', color: 'var(--text-heading)', marginBottom: '0.4rem', fontWeight: '600' }}>Precautionary Directives:</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                {prediction.precautions.map((prec, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    <ShieldCheck size={14} style={{ color: 'var(--accent-green)', flexShrink: 0 }} />
                    {prec}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Diets & Quick Guidelines */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '0.95rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-heading)' }}>
                <HeartPulse size={16} style={{ color: 'var(--accent-amber)' }} /> Indicated Regimen & Diets
              </h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                {prediction.diets.map((diet, i) => (
                  <span key={i} style={{ padding: '0.25rem 0.55rem', background: 'var(--bg-pill)', border: '1px solid var(--border-light)', borderRadius: '3px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    {diet}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 style={{ fontSize: '0.95rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-heading)' }}>
                <Pill size={16} style={{ color: 'var(--accent-green)' }} /> Formulated Primary Medications
              </h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                {prediction.medications.map((med, i) => (
                  <span key={i} style={{ padding: '0.25rem 0.55rem', background: 'var(--bg-pill)', border: '1px solid var(--border-light)', borderRadius: '3px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    {med}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Benchmark Table */}
      {prediction && recommendations.length > 0 && (
        <div className="glass-panel">
          <h3 style={{ fontSize: '1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: '600' }}>
            <Pill size={16} style={{ color: 'var(--accent-amber)' }} /> Cross-Algorithm Recommendation Metrics
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1.25rem' }}>
            Comparison of medication scores normalized on a [1, 10] scale across different filters. Click buttons to record click feedback logs.
          </p>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ minWidth: '600px' }}>
              <thead>
                <tr style={{ color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Medication</th>
                  <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>SVD (Collab)</th>
                  <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>TF-IDF (Content)</th>
                  <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Hybrid (blended)</th>
                  <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>NCF (Deep L.)</th>
                  <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>NLP Sentiment</th>
                  <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Recency Decay</th>
                  <th style={{ padding: '0.6rem 0.75rem', textAlign: 'right', borderBottom: '1px solid var(--border-light)' }}>Action Feedback</th>
                </tr>
              </thead>
              <tbody>
                {recommendations.map((rec) => (
                  <tr key={rec.drug_name} style={{ color: 'var(--text-secondary)' }}>
                    <td style={{ padding: '0.75rem', fontWeight: '600', color: 'var(--text-heading)', borderBottom: '1px solid var(--border-light)' }}>{rec.drug_name}</td>
                    <td style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-light)' }}>{rec.svd_rating.toFixed(1)}</td>
                    <td style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-light)' }}>{rec.content_score.toFixed(1)}</td>
                    <td style={{ padding: '0.75rem', color: 'var(--accent-amber)', fontWeight: '500', borderBottom: '1px solid var(--border-light)' }}>{rec.hybrid_score.toFixed(1)}</td>
                    <td style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-light)' }}>{rec.ncf_rating.toFixed(1)}</td>
                    <td style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-light)' }}>{rec.sentiment_score.toFixed(1)}</td>
                    <td style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-light)' }}>{rec.recency_rating.toFixed(1)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', borderBottom: '1px solid var(--border-light)' }}>
                      {feedbackMsg[rec.drug_name] ? (
                        <span style={{ fontSize: '0.75rem', color: 'var(--accent-amber)', fontWeight: '600' }}>
                          {feedbackMsg[rec.drug_name]}
                        </span>
                      ) : (
                        <div style={{ display: 'inline-flex', gap: '0.25rem' }}>
                          <button
                            onClick={() => handleBanditFeedback(rec.drug_name, true)}
                            style={{ padding: '0.2rem 0.4rem', background: 'var(--bg-pill)', border: '1px solid var(--border-light)', borderRadius: '3px', color: 'var(--accent-green)', cursor: 'pointer', fontSize: '0.7rem' }}
                          >
                            Helpful
                          </button>
                          <button
                            onClick={() => handleBanditFeedback(rec.drug_name, false)}
                            style={{ padding: '0.2rem 0.4rem', background: 'var(--bg-pill)', border: '1px solid var(--border-light)', borderRadius: '3px', color: 'var(--accent-red)', cursor: 'pointer', fontSize: '0.7rem' }}
                          >
                            Irrelevant
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
