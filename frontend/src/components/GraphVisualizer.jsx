// frontend/src/components/GraphVisualizer.jsx
import React, { useState, useEffect } from 'react';
import { Network, Info } from 'lucide-react';

export default function GraphVisualizer({ token, selectedDisease }) {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);
  const [hoveredNode, setHoveredNode] = useState(null);

  useEffect(() => {
    fetchSubgraph();
  }, [selectedDisease]);

  const fetchSubgraph = async () => {
    setLoading(true);
    const diseaseParam = selectedDisease ? `?disease=${encodeURIComponent(selectedDisease)}` : '';
    try {
      const response = await fetch(`/api/graph/subgraph${diseaseParam}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setGraphData(data);
      }
    } catch (err) {
      console.error("Error loading subgraph", err);
    } finally {
      setLoading(false);
    }
  };

  // Dimensions of the SVG container
  const width = 600;
  const height = 400;
  const cx = width / 2;
  const cy = height / 2;

  // Position nodes radially around the center (where the disease node sits)
  const nodes = [...graphData.nodes];
  const links = [...graphData.links];
  
  const diseaseNode = nodes.find(n => n.type === 'Disease') || nodes[0];
  const otherNodes = nodes.filter(n => n !== diseaseNode);
  
  // Assign coordinates
  const nodePositions = {};
  if (diseaseNode) {
    nodePositions[diseaseNode.id] = { x: cx, y: cy };
  }
  
  otherNodes.forEach((node, idx) => {
    const isSymptom = node.type === 'Symptom';
    const radius = isSymptom ? 100 : 160;
    const angle = (idx * 2 * Math.PI) / otherNodes.length;
    
    nodePositions[node.id] = {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle)
    };
  });

  const getNodeColor = (type) => {
    switch (type) {
      case 'Disease': return '#ef4444'; // Soft crimson
      case 'Symptom': return '#f59e0b'; // Amber
      case 'Medication': return '#3b82f6'; // Muted blue
      case 'Diet': return '#10b981'; // Sage green
      case 'Precaution': return '#8b5cf6'; // Muted purple
      default: return '#71717a';
    }
  };

  return (
    <div className="glass-panel" style={{ width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.75rem' }}>
        <Network size={20} style={{ color: 'var(--accent-amber)' }} />
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '600' }}>Disease Knowledge Graph</h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
            Visualizing clinical relationships for: <b style={{ color: 'var(--text-heading)' }}>{selectedDisease || 'Fungal Infection'}</b>
          </p>
        </div>
      </div>

      <div className="graph-layout">
        {/* SVG Drawing Canvas */}
        <div style={{ background: 'var(--bg-input)', borderRadius: '6px', border: '1px solid var(--border-light)', display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '0.5rem', position: 'relative', overflow: 'hidden' }}>
          {loading ? (
            <div style={{ height: '360px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Compiling network graph...</span>
            </div>
          ) : nodes.length === 0 ? (
            <div style={{ height: '360px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No graph data loaded. Analyze symptoms first.</span>
            </div>
          ) : (
            <svg width="100%" height="360px" viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
              {/* Draw Link Edges */}
              {links.map((link, idx) => {
                const start = nodePositions[link.source];
                const end = nodePositions[link.target];
                if (!start || !end) return null;
                return (
                  <line
                    key={idx}
                    x1={start.x}
                    y1={start.y}
                    x2={end.x}
                    y2={end.y}
                    stroke="var(--graph-edge)"
                    strokeWidth="1"
                    strokeDasharray={link.relation === 'has_symptom' ? '0' : '3 3'}
                  />
                );
              })}

              {/* Draw Nodes */}
              {nodes.map((node) => {
                const pos = nodePositions[node.id];
                if (!pos) return null;
                const isCenter = node.type === 'Disease';
                const color = getNodeColor(node.type);
                const isHovered = hoveredNode?.id === node.id;
                
                return (
                  <g
                    key={node.id}
                    transform={`translate(${pos.x}, ${pos.y})`}
                    onMouseEnter={() => setHoveredNode(node)}
                    onMouseLeave={() => setHoveredNode(null)}
                    style={{ cursor: 'pointer' }}
                  >
                    <circle
                      r={isCenter ? 16 : isHovered ? 10 : 7}
                      fill={color}
                      stroke="var(--bg-primary)"
                      strokeWidth={isHovered ? 2 : 1.5}
                      style={{
                        transition: 'r 0.15s ease, stroke-width 0.15s ease'
                      }}
                    />
                    
                    {/* Compact Label */}
                    <text
                      y={isCenter ? -24 : 16}
                      textAnchor="middle"
                      fill={isHovered ? 'var(--text-heading)' : 'var(--text-secondary)'}
                      fontSize={isCenter ? '10px' : '8px'}
                      fontWeight={isCenter || isHovered ? '600' : 'normal'}
                      style={{
                        transition: 'fill 0.15s ease, font-weight 0.15s ease',
                        pointerEvents: 'none'
                      }}
                    >
                      {node.id.length > 15 ? `${node.id.substring(0, 13)}...` : node.id}
                    </text>
                  </g>
                );
              })}
            </svg>
          )}

          {/* Node Overlay Info */}
          {hoveredNode && (
            <div style={{
              position: 'absolute',
              bottom: '12px',
              left: '12px',
              background: 'var(--bg-secondary)',
              border: `1px solid var(--border-light)`,
              borderRadius: '4px',
              padding: '0.4rem 0.65rem',
              maxWidth: '220px',
              fontSize: '0.75rem',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
            }}>
              <div style={{ fontWeight: '600', color: 'var(--text-heading)', marginBottom: '0.15rem' }}>{hoveredNode.id}</div>
              <div style={{ color: getNodeColor(hoveredNode.type), fontWeight: '600', textTransform: 'uppercase', fontSize: '0.65rem' }}>
                Type: {hoveredNode.type}
              </div>
            </div>
          )}
        </div>

        {/* Legend Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          <h4 style={{ fontSize: '0.85rem', color: 'var(--text-heading)', borderBottom: '1px solid var(--border-light)', paddingBottom: '0.4rem', fontWeight: '600' }}>Node Classes</h4>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.8rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ef4444', display: 'inline-block' }}></span>
              <span>Disease</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#f59e0b', display: 'inline-block' }}></span>
              <span>Symptom</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#3b82f6', display: 'inline-block' }}></span>
              <span>Medication</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', display: 'inline-block' }}></span>
              <span>Diet</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#8b5cf6', display: 'inline-block' }}></span>
              <span>Precaution</span>
            </div>
          </div>

          <div style={{
            background: 'var(--bg-pill)',
            padding: '0.6rem',
            borderRadius: '4px',
            border: '1px solid var(--border-light)',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            lineHeight: '1.45',
            marginTop: '0.5rem'
          }}>
            <Info size={13} style={{ color: 'var(--accent-amber)', marginBottom: '0.2rem', display: 'block' }} />
            Hover over nodes to inspect relationship paths inside the local Knowledge Graph neighborhood.
          </div>
        </div>
      </div>
    </div>
  );
}
