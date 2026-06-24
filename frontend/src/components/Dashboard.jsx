// frontend/src/components/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { ShieldAlert, BarChart3, Database, Users, Activity, FileText, CheckCircle, Trash2, Award } from 'lucide-react';

export default function Dashboard({ token, userRole }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // Admin-specific state
  const [usersList, setUsersList] = useState([]);
  const [adminMsg, setAdminMsg] = useState('');

  useEffect(() => {
    fetchDashboardData();
    if (userRole === 'Admin') {
      fetchUsersList();
    }
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/analytics/dashboard', {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      } else if (response.status === 403) {
        setError('Access Denied: You do not possess the required credentials (Admin/Analyst) to view the analytics dashboard.');
      } else {
        throw new Error('Failed to load dashboard analytics');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchUsersList = async () => {
    try {
      const response = await fetch('/api/analytics/users', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setUsersList(data);
      }
    } catch (err) {
      console.error('Failed to load user list:', err);
    }
  };

  const handleRoleChange = async (userId, newRole) => {
    setAdminMsg('');
    try {
      const response = await fetch(`/api/analytics/users/${userId}/role`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ role: newRole })
      });
      const data = await response.json();
      if (response.ok) {
        setAdminMsg(`✅ Role updated for ${data.username} to ${data.new_role}`);
        fetchUsersList();
        fetchDashboardData();
      } else {
        setAdminMsg(`❌ Error: ${data.detail || 'Failed to update role'}`);
      }
    } catch (err) {
      setAdminMsg(`❌ Error: ${err.message}`);
    }
  };

  const handleFlushLogs = async () => {
    if (!window.confirm("Warning: Are you absolutely sure you want to clear all user activity logs? This action is irreversible.")) {
      return;
    }
    setAdminMsg('');
    try {
      const response = await fetch('/api/analytics/logs', {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (response.ok) {
        setAdminMsg(`✅ ${data.message}`);
        fetchDashboardData();
      } else {
        setAdminMsg(`❌ Error: ${data.detail || 'Failed to clear logs'}`);
      }
    } catch (err) {
      setAdminMsg(`❌ Error: ${err.message}`);
    }
  };

  if (userRole !== 'Admin' && userRole !== 'Analyst') {
    return (
      <div className="flex-center" style={{ minHeight: '350px' }}>
        <div className="glass-panel" style={{ maxWidth: '460px', textAlign: 'center', padding: '2.5rem 2rem' }}>
          <div style={{ color: 'var(--accent-red)', marginBottom: '0.85rem' }}>
            <ShieldAlert size={40} style={{ display: 'inline-block' }} />
          </div>
          <h2 style={{ color: 'var(--text-heading)', marginBottom: '0.5rem', fontSize: '1.2rem' }}>Restricted Analytics Portal</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.5' }}>
            The analytics console is exclusively accessible by **Administrators** and **Medical Analysts**. 
            Your current account type is logged as: <b style={{ color: 'var(--accent-amber)' }}>{userRole}</b>.
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <div className="spin" style={{ display: 'inline-block', width: '20px', height: '20px', border: '2px solid var(--border-light)', borderTopColor: 'var(--accent-amber)', borderRadius: '50%', marginBottom: '1rem' }}></div>
        <div style={{ fontSize: '0.85rem' }}>Aggregating clinical activity metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel" style={{ border: '1px solid rgba(239, 68, 68, 0.15)', background: 'var(--accent-red-light)', padding: '2rem', textAlign: 'center' }}>
        <ShieldAlert size={32} style={{ color: 'var(--accent-red)', marginBottom: '0.5rem', display: 'inline-block' }} />
        <h3 style={{ color: 'var(--text-heading)', fontSize: '1.1rem' }}>Analytics Connection Error</h3>
        <p style={{ color: 'var(--accent-red)', fontSize: '0.85rem', marginTop: '0.5rem' }}>{error}</p>
      </div>
    );
  }

  const maxDiagnoses = stats?.top_diagnoses?.length > 0 
    ? Math.max(...stats.top_diagnoses.map(d => d.count)) 
    : 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Overview Cards */}
      <div className="grid-cols-4">
        {/* KPI 1 */}
        <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ padding: '0.5rem', background: 'var(--bg-pill)', border: '1px solid var(--border-light)', borderRadius: '4px', color: 'var(--text-secondary)' }}>
            <Users size={18} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '500' }}>TOTAL USERS</div>
            <div style={{ fontSize: '1.35rem', fontWeight: '600', color: 'var(--text-heading)', marginTop: '0.1rem' }}>
              {stats?.users?.total || 0}
            </div>
          </div>
        </div>

        {/* KPI 2 */}
        <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ padding: '0.5rem', background: 'var(--bg-pill)', border: '1px solid var(--border-light)', borderRadius: '4px', color: 'var(--text-secondary)' }}>
            <Activity size={18} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '500' }}>LOGGED EVENTS</div>
            <div style={{ fontSize: '1.35rem', fontWeight: '600', color: 'var(--text-heading)', marginTop: '0.1rem' }}>
              {stats?.activities?.total || 0}
            </div>
          </div>
        </div>

        {/* KPI 3 */}
        <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ padding: '0.5rem', background: 'var(--bg-pill)', border: '1px solid var(--border-light)', borderRadius: '4px', color: 'var(--text-secondary)' }}>
            <BarChart3 size={18} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '500' }}>PATIENT AVG AGE</div>
            <div style={{ fontSize: '1.35rem', fontWeight: '600', color: 'var(--text-heading)', marginTop: '0.1rem' }}>
              {stats?.users?.avg_age || '35.0'} yrs
            </div>
          </div>
        </div>

        {/* KPI 4 */}
        <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ padding: '0.5rem', background: 'var(--bg-pill)', border: '1px solid var(--border-light)', borderRadius: '4px', color: 'var(--text-secondary)' }}>
            <Database size={18} />
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '500' }}>DATA SERVICE</div>
            <div style={{ fontSize: '1.35rem', fontWeight: '600', color: 'var(--text-heading)', marginTop: '0.1rem' }}>
              Active (SQLite)
            </div>
          </div>
        </div>
      </div>

      <div className="grid-cols-2" style={{ gap: '1.5rem' }}>
        {/* Top Diagnoses bar chart */}
        <div className="glass-panel">
          <h3 style={{ fontSize: '0.95rem', color: 'var(--text-heading)', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.5rem', fontWeight: '600' }}>
            Top Predicted Disease Classifications
          </h3>
          {stats?.top_diagnoses?.length === 0 ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              No diagnoses logged yet.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {stats?.top_diagnoses?.map((item) => {
                const percent = (item.count / maxDiagnoses) * 100;
                return (
                  <div key={item.disease} style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                      <span style={{ color: 'var(--text-heading)', fontWeight: '500' }}>{item.disease}</span>
                      <span style={{ color: 'var(--text-secondary)' }}><b>{item.count}</b> queries</span>
                    </div>
                    <div style={{ height: '4px', background: 'var(--bg-input)', borderRadius: '2px', overflow: 'hidden' }}>
                      <div 
                        style={{ 
                          width: `${percent}%`, 
                          height: '100%', 
                          background: 'var(--text-secondary)',
                          borderRadius: '2px'
                        }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Roles & Activity Breakdown */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '0.95rem', color: 'var(--text-heading)', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.5rem', fontWeight: '600' }}>
            User Role Allocation
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {stats?.users?.roles && Object.entries(stats.users.roles).map(([role, count]) => (
              <div key={role} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-tertiary)', padding: '0.5rem 0.85rem', borderRadius: '4px', border: '1px solid var(--border-light)' }}>
                <span className={`role-badge ${role === 'Admin' ? 'badge-admin' : role === 'Analyst' ? 'badge-analyst' : 'badge-user'}`}>
                  {role}
                </span>
                <span style={{ color: 'var(--text-heading)', fontWeight: '500', fontSize: '0.8rem' }}>{count} Accounts</span>
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: '0.95rem', color: 'var(--text-heading)', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.5rem', fontWeight: '600', marginTop: '0.5rem' }}>
            Activity Logs Summary
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            {stats?.activities?.breakdown && Object.entries(stats.activities.breakdown).map(([act, count]) => (
              <span 
                key={act} 
                style={{ 
                  padding: '0.25rem 0.55rem', 
                  background: 'var(--bg-pill)', 
                  border: '1px solid var(--border-light)', 
                  borderRadius: '3px', 
                  fontSize: '0.75rem', 
                  color: 'var(--text-secondary)' 
                }}
              >
                {act}: <b>{count}</b>
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Model Benchmark Matrix */}
      <div className="glass-panel">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.5rem' }}>
          <CheckCircle size={16} style={{ color: 'var(--accent-green)' }} />
          <h3 style={{ fontSize: '0.95rem', fontWeight: '600' }}>Offline Model Accuracy Benchmarks</h3>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1rem' }}>
          Validation results (Root Mean Squared Error & Mean Absolute Error) computed offline against test splits. Lower error indicates higher predictive fidelity.
        </p>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ color: 'var(--text-secondary)' }}>
                <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Algorithm</th>
                <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>RMSE</th>
                <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>MAE</th>
                <th style={{ padding: '0.6rem 0.75rem', textAlign: 'right', borderBottom: '1px solid var(--border-light)' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {stats?.model_comparison?.map((row, idx) => (
                <tr key={idx} style={{ color: 'var(--text-secondary)' }}>
                  <td style={{ padding: '0.75rem', fontWeight: '500', color: 'var(--text-heading)', borderBottom: '1px solid var(--border-light)' }}>
                    {row.Model}
                  </td>
                  <td style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-light)' }}>
                    {typeof row.RMSE === 'number' ? row.RMSE.toFixed(4) : row.RMSE}
                  </td>
                  <td style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-light)' }}>
                    {typeof row.MAE === 'number' ? row.MAE.toFixed(4) : row.MAE}
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--accent-green)', fontWeight: '500', borderBottom: '1px solid var(--border-light)' }}>
                    {row.Status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Audit Logs table */}
      <div className="glass-panel">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.5rem' }}>
          <FileText size={16} style={{ color: 'var(--accent-amber)' }} />
          <h3 style={{ fontSize: '0.95rem', fontWeight: '600' }}>Clinical Audit Trail</h3>
        </div>
        
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ color: 'var(--text-secondary)' }}>
                <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Timestamp</th>
                <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Operator</th>
                <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Action</th>
                <th style={{ padding: '0.6rem 0.75rem', textAlign: 'left', borderBottom: '1px solid var(--border-light)' }}>Details</th>
              </tr>
            </thead>
            <tbody>
              {stats?.recent_logs?.map((log) => (
                <tr key={log.id} style={{ color: 'var(--text-secondary)' }}>
                  <td style={{ padding: '0.75rem', whiteSpace: 'nowrap', borderBottom: '1px solid var(--border-light)' }}>{log.timestamp}</td>
                  <td style={{ padding: '0.75rem', color: 'var(--text-heading)', fontWeight: '500', borderBottom: '1px solid var(--border-light)' }}>{log.username}</td>
                  <td style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-light)' }}>
                    <span style={{
                      padding: '0.15rem 0.35rem',
                      borderRadius: '3px',
                      fontSize: '0.75rem',
                      fontWeight: '600',
                      background: 'var(--bg-pill)',
                      border: '1px solid var(--border-light)',
                      color: 'var(--text-primary)'
                    }}>
                      {log.activity_type}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem', maxWidth: '350px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', borderBottom: '1px solid var(--border-light)' }} title={log.details}>
                    {log.details}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Admin Panel Controls: Only rendered for Admin role */}
      {userRole === 'Admin' && (
        <div className="glass-panel" style={{ border: '1px solid rgba(217, 119, 6, 0.15)', background: 'var(--accent-amber-light)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', borderBottom: '1px solid rgba(217, 119, 6, 0.15)', paddingBottom: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Award size={18} style={{ color: 'var(--accent-amber)' }} />
              <h3 style={{ fontSize: '0.95rem', fontWeight: '600', color: 'var(--text-heading)' }}>Administrative System Management</h3>
            </div>
            <button 
              className="btn-secondary" 
              onClick={handleFlushLogs}
              style={{ 
                borderColor: 'var(--accent-red)', 
                color: 'var(--accent-red)',
                padding: '0.35rem 0.75rem',
                fontSize: '0.75rem',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem'
              }}
            >
              <Trash2 size={12} /> Clear System Logs
            </button>
          </div>

          {adminMsg && (
            <div style={{
              background: 'rgba(0,0,0,0.2)',
              border: '1px solid var(--border-light)',
              borderRadius: '4px',
              color: '#ffffff',
              padding: '0.5rem 0.75rem',
              marginBottom: '1rem',
              fontSize: '0.8rem',
              fontWeight: '500'
            }}>
              {adminMsg}
            </div>
          )}

          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1rem' }}>
            Below is the listing of active system accounts. As an Administrator, you can promote or demote user access credentials instantly:
          </p>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.8rem' }}>
              <thead>
                <tr style={{ color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '0.5rem', textAlign: 'left', borderBottom: '1px solid rgba(217, 119, 6, 0.15)' }}>User ID</th>
                  <th style={{ padding: '0.5rem', textAlign: 'left', borderBottom: '1px solid rgba(217, 119, 6, 0.15)' }}>Username</th>
                  <th style={{ padding: '0.5rem', textAlign: 'left', borderBottom: '1px solid rgba(217, 119, 6, 0.15)' }}>Current Role</th>
                  <th style={{ padding: '0.5rem', textAlign: 'right', borderBottom: '1px solid rgba(217, 119, 6, 0.15)' }}>Assign System Role</th>
                </tr>
              </thead>
              <tbody>
                {usersList.map((user) => (
                  <tr key={user.id} style={{ color: 'var(--text-secondary)' }}>
                    <td style={{ padding: '0.65rem 0.5rem', borderBottom: '1px solid rgba(217, 119, 6, 0.05)' }}>#{user.id}</td>
                    <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-heading)', fontWeight: '500', borderBottom: '1px solid rgba(217, 119, 6, 0.05)' }}>{user.username}</td>
                    <td style={{ padding: '0.65rem 0.5rem', borderBottom: '1px solid rgba(217, 119, 6, 0.05)' }}>
                      <span className={`role-badge ${user.role === 'Admin' ? 'badge-admin' : user.role === 'Analyst' ? 'badge-analyst' : 'badge-user'}`}>
                        {user.role}
                      </span>
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', textAlign: 'right', borderBottom: '1px solid rgba(217, 119, 6, 0.05)' }}>
                      <select 
                        value={user.role} 
                        onChange={(e) => handleRoleChange(user.id, e.target.value)}
                        style={{ width: '120px', padding: '0.2rem 0.4rem', fontSize: '0.75rem', display: 'inline-block' }}
                      >
                        <option value="User">User</option>
                        <option value="Analyst">Analyst</option>
                        <option value="Admin">Admin</option>
                      </select>
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
