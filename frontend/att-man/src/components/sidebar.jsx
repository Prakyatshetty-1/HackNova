import { 
  Upload, 
  FileText, 
  AlertTriangle, 
  Settings,
  LogOut,
  LayoutDashboard
} from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useClerk } from '@clerk/clerk-react';
import '../styles/Dashboard.css';

export default function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { signOut } = useClerk();
  
  const menuItems = [
    { icon: LayoutDashboard, label: 'Dashboard', path: '/' },
    { icon: Upload, label: 'Upload', path: '/upload' },
    { icon: FileText, label: 'Reports', path: '/reports' },
    { icon: AlertTriangle, label: 'Anomalies', path: '/anomalies' },
    { icon: Settings, label: 'Settings', path: '/settings' },
  ];

  const isActive = (path) => location.pathname === path;

  const handleLogout = async () => {
    await signOut();
    navigate('/login');
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2 className="sidebar-logo">AttendanceAI</h2>
      </div>
      <nav className="sidebar-nav">
        {menuItems.map((item, index) => (
          <Link 
            key={index} 
            to={item.path}
            className={`sidebar-item ${isActive(item.path) ? 'active' : ''}`}
          >
            <item.icon size={20} />
            <span>{item.label}</span>
          </Link>
        ))}
      </nav>
      <div className="sidebar-footer">
        <button 
          onClick={handleLogout} 
          className="sidebar-item logout-button"
        >
          <LogOut size={20} />
          <span>Logout</span>
        </button>
      </div>
    </aside>
  );
}