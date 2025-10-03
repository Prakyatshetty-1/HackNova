import React from 'react';
import { 
  Upload, 
  FileText, 
  AlertTriangle, 
  UserX, 
  CheckCircle2,
  Menu,
  LayoutDashboard,
  Settings,
  LogOut,
  Clock
} from "lucide-react";
import Sidebar from '../components/sidebar';
import '../styles/Dashboard.css';


// StatsCard Component
const StatsCard = ({ title, icon: Icon, value, description, descriptionClass, iconClass }) => {
  return (
    <div className="stats-card">
      <div className="stats-card-header">
        <span className="stats-card-title">{title}</span>
        <div className={`stats-icon ${iconClass || ''}`}>
          <Icon size={20} />
        </div>
      </div>
      <div className="stats-card-content">
        <div className="stats-value">{value}</div>
        <p className={`stats-description ${descriptionClass || ''}`}>
          {description}
        </p>
      </div>
    </div>
  );
};

// QuickActionCard Component
const QuickActionCard = ({ 
  icon: Icon, 
  title, 
  subtitle, 
  description, 
  link, 
  buttonText, 
  buttonVariant,
  iconBg 
}) => {
  return (
    <div className="quick-action-card">
      <div className={`quick-action-icon ${iconBg}`}>
        <Icon size={24} />
      </div>
      <div className="quick-action-content">
        <div className="quick-action-header">
          <h3 className="quick-action-title">{title}</h3>
          <span className="quick-action-subtitle">{subtitle}</span>
        </div>
        <p className="quick-action-description">{description}</p>
        <a href={link} className={`quick-action-button ${buttonVariant}`}>
          {buttonText}
        </a>
      </div>
    </div>
  );
};

// RecentActivity Component
const RecentActivity = () => {
  const activities = [
    {
      action: "Attendance sheet processed",
      subject: "Computer Science - Year 3",
      time: "5 minutes ago",
      status: "success"
    },
    {
      action: "Anomaly detected",
      subject: "Mathematics - Year 2",
      time: "12 minutes ago",
      status: "warning"
    },
    {
      action: "Report generated",
      subject: "Engineering Department",
      time: "1 hour ago",
      status: "info"
    },
    {
      action: "Attendance sheet processed",
      subject: "Physics - Year 1",
      time: "2 hours ago",
      status: "success"
    }
  ];

  return (
    <div className="recent-activity">
      <h2 className="recent-activity-title">Recent Activity</h2>
      <div className="activity-list">
        {activities.map((activity, index) => (
          <div key={index} className="activity-item">
            <div className={`activity-status ${activity.status}`}></div>
            <div className="activity-content">
              <div className="activity-main">
                <span className="activity-action">{activity.action}</span>
                <span className="activity-subject">{activity.subject}</span>
              </div>
              <div className="activity-time">
                <Clock size={14} />
                {activity.time}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Main Dashboard Component
export default function DashboardPage() {
  

  const quickActions = [
    {
      icon: Upload,
      title: "Upload Attendance",
      subtitle: "Process new sheets",
      description: "Upload scanned attendance sheets in image or PDF format for automated processing.",
      link: "/upload",
      buttonText: "Upload Sheets",
      buttonVariant: "primary",
      iconBg: "primary"
    },
    {
      icon: FileText,
      title: "View Reports",
      subtitle: "Subject & department data",
      description: "Access subject-wise and department-level attendance reports with detailed analytics.",
      link: "/reports",
      buttonText: "View Reports",
      buttonVariant: "outline",
      iconBg: "accent"
    },
    {
      icon: AlertTriangle,
      title: "Review Anomalies",
      subtitle: "Check flagged entries",
      description: "Review duplicate entries, suspicious signatures, and inconsistent markings.",
      link: "#/anomalies",
      buttonText: "Review Now",
      buttonVariant: "outline",
      iconBg: "warning"
    }
  ];

  return (
    <div className="dashboard-layout">
      <Sidebar />
      <main className="dashboard-main">
        <div className="dashboard-container">
          <div className="dashboard-header">
            <h1 className="dashboard-title">Dashboard</h1>
            <p className="dashboard-description">Overview of attendance management system</p>
          </div>

       

          <div className="quick-actions-grid">
            {quickActions.map((action, index) => (
              <QuickActionCard key={index} {...action} />
            ))}
          </div>

          <RecentActivity />
        </div>
      </main>
    </div>
  );
}