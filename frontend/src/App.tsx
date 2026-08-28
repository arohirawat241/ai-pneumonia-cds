import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { Container, Typography, Box, Paper, Button, CircularProgress, Alert, TextField, Tabs, Tab } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';
import LogoutIcon from '@mui/icons-material/Logout';

interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities: { NORMAL: number; PNEUMONIA: number };
  heatmap_base64: string;
  requires_review: boolean; // <-- NEW
}

interface Case {
  id: number;
  filename: string;
  prediction: string;
  confidence: number;
  prob_normal: number;
  prob_pneumonia: number;
  created_at: string;
  image_path: string;
}

function CaseHistory({ token }: { token: string | null }) {
  const [cases, setCases] = useState<Case[]>([]);

  useEffect(() => {
    if (!token) return;
    fetch('http://localhost:8000/api/cases', {
      headers: { 'Authorization': `Bearer ${token}` }
    })
      .then(res => res.json())
      .then(data => setCases(data))
      .catch(err => console.error('Error fetching cases:', err));
  }, [token]);

  if (cases.length === 0) return null;

  return (
    <Paper sx={{ p: 3, mt: 4 }}>
      <Typography variant="h5" gutterBottom>Recent Cases</Typography>
      <Box sx={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #ddd' }}>
              <th style={{ padding: '10px', textAlign: 'left' }}>ID</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Prediction</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Confidence</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Image</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr key={c.id} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '10px' }}>#{c.id}</td>
                <td style={{ padding: '10px', color: c.prediction === 'PNEUMONIA' ? '#f44336' : '#4caf50', fontWeight: 'bold' }}>
                  {c.prediction}
                </td>
                <td style={{ padding: '10px' }}>{c.confidence}%</td>
                <td style={{ padding: '10px' }}>
                  <img 
                    src={`http://localhost:8000${c.image_path.replace('backend', '')}`} 
                    alt="X-ray" 
                    style={{ width: '60px', height: '60px', objectFit: 'cover', borderRadius: '4px' }} 
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Box>
    </Paper>
  );
}

function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    if (savedToken) setToken(savedToken);
  }, []);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthLoading(true);
    setAuthError('');
    
    try {
      const endpoint = authMode === 'login' ? '/api/login' : '/api/register';
      const response = await axios.post(`http://localhost:8000${endpoint}`, { username, password });
      
      if (authMode === 'login') {
        localStorage.setItem('token', response.data.access_token);
        setToken(response.data.access_token);
      } else {
        alert('Registration successful! Please log in.');
        setAuthMode('login');
      }
    } catch (err: any) {
      setAuthError(err.response?.data?.detail || 'Authentication failed');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const { getRootProps, getInputProps } = useDropzone({
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      const selectedFile = acceptedFiles[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  });

  const handlePredict = async () => {
    if (!file || !token) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post('http://localhost:8000/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        },
      });
      setResult(response.data);
    } catch (err: any) {
      if (err.response?.status === 401) {
        handleLogout();
        setError('Session expired. Please log in again.');
      } else {
        setError('Failed to get prediction. Make sure the backend is running.');
      }
    } finally {
      setLoading(false);
    }
  };

  if (!token) {
    return (
      <Container maxWidth="sm">
        <Box sx={{ mt: 10, textAlign: 'center' }}>
          <Typography variant="h4" gutterBottom> Pneumonia Detection</Typography>
          <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
            AI-Assisted Clinical Decision Support System
          </Typography>
          
          <Paper sx={{ p: 4 }}>
            <Tabs value={authMode} onChange={(_, newValue) => { setAuthMode(newValue); setAuthError(''); }} centered sx={{ mb: 3 }}>
              <Tab label="Login" value="login" />
              <Tab label="Register" value="register" />
            </Tabs>

            <form onSubmit={handleAuth}>
              <TextField fullWidth label="Username" margin="normal" value={username} onChange={(e) => setUsername(e.target.value)} required />
              <TextField fullWidth label="Password" type="password" margin="normal" value={password} onChange={(e) => setPassword(e.target.value)} required />
              {authError && <Alert severity="error" sx={{ mt: 2 }}>{authError}</Alert>}
              <Button type="submit" variant="contained" fullWidth size="large" sx={{ mt: 3 }} disabled={authLoading}>
                {authLoading ? <CircularProgress size={24} /> : (authMode === 'login' ? 'Login' : 'Create Account')}
              </Button>
            </form>
          </Paper>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="md">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 3, mb: 3 }}>
        <Typography variant="h4">🫁 Pneumonia Detection</Typography>
        <Button variant="outlined" color="error" startIcon={<LogoutIcon />} onClick={handleLogout}>Logout</Button>
      </Box>

      <Paper sx={{ p: 3, mb: 3 }}>
        <div {...getRootProps()} style={{ border: '2px dashed #1976d2', borderRadius: '8px', padding: '40px', textAlign: 'center', cursor: 'pointer', backgroundColor: preview ? '#f5f5f5' : 'white' }}>
          <input {...getInputProps()} />
          {preview ? (
            <Box>
              <img src={preview} alt="Preview" style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '8px' }} />
              <Typography variant="body2" sx={{ mt: 2 }}>Click or drag to replace image</Typography>
            </Box>
          ) : (
            <Box>
              <CloudUploadIcon sx={{ fontSize: 64, color: '#1976d2', mb: 2 }} />
              <Typography variant="h6">Drop X-Ray Image Here</Typography>
              <Typography variant="body2" color="textSecondary">or click to browse (JPG, PNG)</Typography>
            </Box>
          )}
        </div>
      </Paper>

      {file && (
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <Button variant="contained" size="large" onClick={handlePredict} disabled={loading} startIcon={loading ? <CircularProgress size={20} /> : <ImageIcon />}>
            {loading ? 'Analyzing...' : 'Analyze X-Ray'}
          </Button>
        </Box>
      )}

      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

      {result && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom align="center">Prediction Results</Typography>
          <Alert severity={result.prediction === 'PNEUMONIA' ? 'warning' : 'success'} sx={{ mb: 3 }}>
            <Typography variant="h6" align="center">{result.prediction}</Typography>
            <Typography variant="body2" align="center">Confidence: {result.confidence}%</Typography>
          </Alert>

          {/* --- NEW: UNCERTAINTY ESTIMATION BANNER --- */}
          {result.requires_review && (
            <Alert severity="warning" sx={{ mb: 3, border: '2px solid #ed6c02' }}>
              <Typography variant="h6" align="center">⚠️ Requires Human Review</Typography>
              <Typography variant="body2" align="center">
                AI confidence is below 80%. A radiologist should verify this scan.
              </Typography>
            </Alert>
          )}
          {/* ------------------------------------------ */}

          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap', mb: 3 }}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" gutterBottom>Original X-Ray</Typography>
              <img src={preview!} alt="Original" style={{ width: '300px', height: '300px', objectFit: 'cover', borderRadius: '8px', border: '2px solid #e0e0e0' }} />
            </Box>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" gutterBottom>AI Attention (Grad-CAM)</Typography>
              <img src={result.heatmap_base64} alt="Grad-CAM" style={{ width: '300px', height: '300px', objectFit: 'cover', borderRadius: '8px', border: '2px solid #f44336' }} />
            </Box>
          </Box>

          <Box sx={{ maxWidth: '400px', margin: '0 auto' }}>
            <Typography variant="subtitle2" gutterBottom>Class Probabilities:</Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2">Normal: {result.probabilities.NORMAL}%</Typography>
              <Box sx={{ width: '100%', backgroundColor: '#e0e0e0', height: '8px', borderRadius: '4px', mt: 1 }}>
                <Box sx={{ width: `${result.probabilities.NORMAL}%`, backgroundColor: '#4caf50', height: '100%', borderRadius: '4px' }} />
              </Box>
            </Box>
            <Box>
              <Typography variant="body2">Pneumonia: {result.probabilities.PNEUMONIA}%</Typography>
              <Box sx={{ width: '100%', backgroundColor: '#e0e0e0', height: '8px', borderRadius: '4px', mt: 1 }}>
                <Box sx={{ width: `${result.probabilities.PNEUMONIA}%`, backgroundColor: '#f44336', height: '100%', borderRadius: '4px' }} />
              </Box>
            </Box>
          </Box>
        </Paper>
      )}

      <CaseHistory token={token} />
    </Container>
  );
}

export default App;