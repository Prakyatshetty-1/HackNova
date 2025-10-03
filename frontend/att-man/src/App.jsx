import { SignedIn, SignedOut, SignUp, SignIn } from '@clerk/clerk-react'
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom'
import './App.css'
import './styles/Auth.css'
import Dashboard from './pages/Dashboard'
import Uploads from './pages/Uploads'
import Reports from './pages/Reports'
import Anomalies from './pages/Anomalies'

// Login Page Component
function LoginPage({ onNavigate }) {
  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <h1 className="auth-title">Welcome Back</h1>
          <p className="auth-subtitle">Sign in to access your dashboard</p>
        </div>
        
        <div className="auth-card">
          <SignIn 
            routing="hash"
            signUpUrl="/signup"
          />
          
          <div className="auth-footer">
            <p className="auth-footer-text">
              Don't have an account?{' '}
              <button
                onClick={() => onNavigate('/signup')}
                className="auth-link"
              >
                Sign up
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Signup Page Component
function SignupPage({ onNavigate }) {
  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <h1 className="auth-title">Get Started</h1>
          <p className="auth-subtitle">Create your account to continue</p>
        </div>
        
        <div className="auth-card">
          <SignUp 
            routing="hash"
            signInUrl="/login"
          />
          
          <div className="auth-footer">
            <p className="auth-footer-text">
              Already have an account?{' '}
              <button
                onClick={() => onNavigate('/login')}
                className="auth-link"
              >
                Sign in
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Landing Page Component
function LandingPage({ onNavigate }) {
  return (
    <div className="landing-page">
      <nav className="landing-nav">
        <h1 className="landing-logo">MyApp</h1>
        <div className="landing-nav-buttons">
          <button
            onClick={() => onNavigate('/login')}
            className="landing-btn landing-btn-text"
          >
            Sign In
          </button>
          <button
            onClick={() => onNavigate('/signup')}
            className="landing-btn landing-btn-primary"
          >
            Sign Up
          </button>
        </div>
      </nav>
      
      <div className="landing-content">
        <h2 className="landing-title">
          Welcome to Your Dashboard
        </h2>
        <p className="landing-description">
          Manage your uploads, reports, and anomalies all in one place.
        </p>
        
        <div className="landing-cta">
          <button
            onClick={() => onNavigate('/signup')}
            className="landing-btn landing-btn-large landing-btn-primary"
          >
            Get Started
          </button>
          <button
            onClick={() => onNavigate('/login')}
            className="landing-btn landing-btn-large landing-btn-secondary"
          >
            Sign In
          </button>
        </div>
        
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">
              <svg className="feature-svg" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <h3 className="feature-title">Easy Uploads</h3>
            <p className="feature-description">Upload and manage your files seamlessly</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <svg className="feature-svg" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="feature-title">Detailed Reports</h3>
            <p className="feature-description">Generate comprehensive analytics and insights</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <svg className="feature-svg" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 className="feature-title">Anomaly Detection</h3>
            <p className="feature-description">Identify and track unusual patterns instantly</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Protected Route Component
function ProtectedRoute({ children }) {
  return (
    <>
      <SignedIn>{children}</SignedIn>
      <SignedOut>
        <Navigate to="/login" replace />
      </SignedOut>
    </>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route 
          path="/landing" 
          element={
            <SignedOut>
              <LandingPage onNavigate={(path) => window.location.href = path} />
            </SignedOut>
          } 
        />
        <Route 
          path="/login" 
          element={
            <SignedOut>
              <LoginPage onNavigate={(path) => window.location.href = path} />
            </SignedOut>
          } 
        />
        <Route 
          path="/signup" 
          element={
            <SignedOut>
              <SignupPage onNavigate={(path) => window.location.href = path} />
            </SignedOut>
          } 
        />
        
        {/* Protected Routes */}
        <Route 
          path="/" 
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/upload" 
          element={
            <ProtectedRoute>
              <Uploads />
            </ProtectedRoute>
          }
        />
        <Route 
          path="/reports" 
          element={
            <ProtectedRoute>
              <Reports />
            </ProtectedRoute>
          }
        />
        <Route 
          path="/anomalies" 
          element={
            <ProtectedRoute>
              <Anomalies />
            </ProtectedRoute>
          }
        />
      </Routes>
    </Router>
  )
}

export default App