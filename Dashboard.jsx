import React, { useState, useEffect, useRef } from 'react';

export default function Dashboard() {
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [employees, setEmployees] = useState([]);
  const [logs, setLogs] = useState([]);
  const [newEmp, setNewEmp] = useState({ employee_id: '', name: '', department: '' });
  const [verificationStatus, setVerificationStatus] = useState(null);
  const videoRef = useRef(null);

  // --- 1. Authentication Handler ---
  const handleLogin = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    try {
      const res = await fetch('http://localhost:8000/token', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setToken(data.access_token);
        localStorage.setItem('token', data.access_token);
      } else {
        alert('Invalid login credentials');
      }
    } catch (err) {
      console.error('Login Error:', err);
      alert('Unable to connect to backend. Is FastAPI running on port 8000?');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken('');
  };

  // --- 2. HTML5 WebRTC Webcam Stream ---
  useEffect(() => {
    if (token && videoRef.current) {
      navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480 } })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((err) => console.error('Camera Access Error:', err));
    }
  }, [token]);

  // --- 3. Data Fetching (Employees & Logs from Neon) ---
  const fetchEmployees = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/employees');
      if (res.ok) {
        const data = await res.json();
        setEmployees(data);
      }
    } catch (err) {
      console.error('Fetch Employees Error:', err);
    }
  };

  const fetchLogs = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/logs');
      if (res.ok) {
        const data = await res.json();
        setLogs(data);
      }
    } catch (err) {
      console.error('Fetch Logs Error:', err);
    }
  };

  useEffect(() => {
    if (token) {
      fetchEmployees();
      fetchLogs();
      const interval = setInterval(fetchLogs, 5000); // Auto-refresh logs every 5 seconds
      return () => clearInterval(interval);
    }
  }, [token]);

  // --- 4. Employee Registration Handler ---
  const handleAddEmployee = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch('http://localhost:8000/api/v1/employees', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newEmp),
      });
      if (res.ok) {
        setNewEmp({ employee_id: '', name: '', department: '' });
        fetchEmployees();
      }
    } catch (err) {
      console.error('Add Employee Error:', err);
    }
  };

  // --- 5. Manual Frame Trigger Verification ---
  const handleVerifyCurrentFrame = async () => {
    if (!videoRef.current) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      try {
        const res = await fetch('http://localhost:8000/api/v1/verify', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        setVerificationStatus(data);
        fetchLogs();
      } catch (err) {
        console.error('Verification Error:', err);
      }
    }, 'image/jpeg');
  };

  // --- Unauthenticated View: Login Screen ---
  if (!token) {
    return (
      <div style={{ maxWidth: '400px', margin: '4rem auto', fontFamily: 'sans-serif', padding: '2rem', border: '1px solid #ddd', borderRadius: '8px' }}>
        <h2>🔐 Admin Authentication</h2>
        <form onSubmit={handleLogin}>
          <div style={{ marginBottom: '1rem' }}>
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              style={{ width: '100%', padding: '8px', marginTop: '4px' }}
              placeholder="admin"
              required
            />
          </div>
          <div style={{ marginBottom: '1rem' }}>
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{ width: '100%', padding: '8px', marginTop: '4px' }}
              placeholder="admin123"
              required
            />
          </div>
          <button type="submit" style={{ width: '100%', padding: '10px', background: '#0070f3', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
            Login to Dashboard
          </button>
        </form>
      </div>
    );
  }

  // --- Authenticated View: Admin Dashboard ---
  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif', backgroundColor: '#f9fafb', minHeight: '100vh' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '1rem' }}>
        <h1>🏢 Enterprise Face Attendance Dashboard</h1>
        <button onClick={handleLogout} style={{ padding: '8px 16px', background: '#ef4444', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
          Logout
        </button>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        {/* Left Column: Camera Stream & Manual Trigger */}
        <div style={{ background: '#fff', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <h3>📷 Live WebRTC Camera Stream</h3>
          <video ref={videoRef} autoPlay playsInline width="100%" style={{ borderRadius: '6px', background: '#000', marginBottom: '1rem' }} />
          <button
            onClick={handleVerifyCurrentFrame}
            style={{ width: '100%', padding: '12px', background: '#10b981', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
          >
            Verify Frame & Match Vector
          </button>

          {verificationStatus && (
            <div style={{ marginTop: '1rem', padding: '1rem', borderRadius: '4px', background: verificationStatus.status === 'success' ? '#ecfdf5' : '#fef2f2' }}>
              <strong>Status:</strong> {verificationStatus.status}<br />
              {verificationStatus.person_name && <><strong>Recognized:</strong> {verificationStatus.person_name}<br /></>}
              {verificationStatus.message && <><strong>Message:</strong> {verificationStatus.message}</>}
            </div>
          )}
        </div>

        {/* Right Column: Employee Registry & Neon PostgreSQL Logs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Employee Registry */}
          <div style={{ background: '#fff', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h3>👥 Employee Registration</h3>
            <form onSubmit={handleAddEmployee} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr auto', gap: '8px', marginBottom: '1rem' }}>
              <input
                type="text"
                placeholder="Emp ID"
                value={newEmp.employee_id}
                onChange={(e) => setNewEmp({ ...newEmp, employee_id: e.target.value })}
                required
              />
              <input
                type="text"
                placeholder="Name"
                value={newEmp.name}
                onChange={(e) => setNewEmp({ ...newEmp, name: e.target.value })}
                required
              />
              <input
                type="text"
                placeholder="Department"
                value={newEmp.department}
                onChange={(e) => setNewEmp({ ...newEmp, department: e.target.value })}
                required
              />
              <button type="submit" style={{ padding: '6px 12px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
                Add
              </button>
            </form>

            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                  <th style={{ padding: '8px' }}>ID</th>
                  <th style={{ padding: '8px' }}>Name</th>
                  <th style={{ padding: '8px' }}>Department</th>
                </tr>
              </thead>
              <tbody>
                {employees.map((emp) => (
                  <tr key={emp.employee_id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '8px' }}>{emp.employee_id}</td>
                    <td style={{ padding: '8px' }}>{emp.name}</td>
                    <td style={{ padding: '8px' }}>{emp.department}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Activity Logs */}
          <div style={{ background: '#fff', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h3>📋 Real-Time Neon Database Logs</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                  <th style={{ padding: '8px' }}>Person</th>
                  <th style={{ padding: '8px' }}>Confidence</th>
                  <th style={{ padding: '8px' }}>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, index) => (
                  <tr key={index} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '8px' }}>{log.PersonID}</td>
                    <td style={{ padding: '8px' }}>{typeof log.Confidence === 'number' ? log.Confidence.toFixed(2) : log.Confidence}</td>
                    <td style={{ padding: '8px' }}>{log.Timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}