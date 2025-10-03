import { AlertTriangle, CheckCircle2, X, Eye, Copy, Users } from "lucide-react"
import { useState } from "react"
import Sidebar from "../components/sidebar"
import "../styles/Anomalies.css"

const mockAnomalies = [
  {
    id: "1",
    type: "duplicate",
    severity: "high",
    subject: "Data Structures",
    rollNumber: "CE2021045",
    studentName: "Rahul Sharma",
    date: "2024-03-08",
    description: "Duplicate entry detected - marked present in two different lecture slots",
    status: "pending",
  },
  {
    id: "2",
    type: "signature",
    severity: "high",
    subject: "Computer Networks",
    rollNumber: "CE2021089",
    studentName: "Priya Patel",
    date: "2024-03-07",
    description: "Signature mismatch - current signature differs significantly from previous records",
    status: "pending",
  },
  {
    id: "3",
    type: "inconsistent",
    severity: "medium",
    subject: "Operating Systems",
    rollNumber: "CE2021123",
    studentName: "Amit Kumar",
    date: "2024-03-09",
    description: "Inconsistent marking - unclear presence indicator (neither P nor A)",
    status: "pending",
  },
  {
    id: "4",
    type: "invalid",
    severity: "low",
    subject: "Database Management",
    rollNumber: "CE2021156",
    studentName: "Sneha Gupta",
    date: "2024-03-06",
    description: "Invalid roll number format detected in attendance sheet",
    status: "resolved",
  },
  {
    id: "5",
    type: "duplicate",
    severity: "high",
    subject: "Software Engineering",
    rollNumber: "CE2021078",
    studentName: "Vikram Singh",
    date: "2024-03-10",
    description: "Multiple entries for same student in single lecture",
    status: "pending",
  },
]

// Card Components
const Card = ({ children, className = "" }) => (
  <div className={`card ${className}`}>{children}</div>
)

const CardHeader = ({ children, className = "" }) => (
  <div className={`card-header ${className}`}>{children}</div>
)

const CardTitle = ({ children, className = "" }) => (
  <h3 className={`card-title ${className}`}>{children}</h3>
)

const CardDescription = ({ children, className = "" }) => (
  <p className={`card-description ${className}`}>{children}</p>
)

const CardContent = ({ children, className = "" }) => (
  <div className={`card-content ${className}`}>{children}</div>
)

// Button Component
const Button = ({ children, variant = "default", size = "default", className = "", onClick }) => (
  <button className={`btn btn-${variant} btn-${size} ${className}`} onClick={onClick}>
    {children}
  </button>
)

// Badge Component
const Badge = ({ children, variant = "default", className = "" }) => (
  <span className={`badge badge-${variant} ${className}`}>{children}</span>
)

export default function AnomaliesPage() {
  const [anomalies, setAnomalies] = useState(mockAnomalies)
  const [filter, setFilter] = useState("all")

  const filteredAnomalies = anomalies.filter((a) => (filter === "all" ? true : a.status === filter))

  const getTypeIcon = (type) => {
    switch (type) {
      case "duplicate":
        return <Copy className="icon-sm" />
      case "signature":
        return <Users className="icon-sm" />
      case "inconsistent":
        return <AlertTriangle className="icon-sm" />
      case "invalid":
        return <X className="icon-sm" />
      default:
        return <AlertTriangle className="icon-sm" />
    }
  }

  const getTypeColor = (type) => {
    switch (type) {
      case "duplicate":
        return "type-destructive"
      case "signature":
        return "type-warning"
      case "inconsistent":
        return "type-primary"
      case "invalid":
        return "type-muted"
      default:
        return "type-muted"
    }
  }

  const getSeverityColor = (severity) => {
    switch (severity) {
      case "high":
        return "destructive"
      case "medium":
        return "secondary"
      case "low":
        return "outline"
      default:
        return "secondary"
    }
  }

  const handleResolve = (id) => {
    setAnomalies((prev) => prev.map((a) => (a.id === id ? { ...a, status: "resolved" } : a)))
  }

  const handleDismiss = (id) => {
    setAnomalies((prev) => prev.map((a) => (a.id === id ? { ...a, status: "dismissed" } : a)))
  }

  const pendingCount = anomalies.filter((a) => a.status === "pending").length
  const resolvedCount = anomalies.filter((a) => a.status === "resolved").length
  const dismissedCount = anomalies.filter((a) => a.status === "dismissed").length
  const highSeverityCount = anomalies.filter((a) => a.severity === "high" && a.status === "pending").length

  return (
    <div className="anomalies-container">
      <Sidebar />
      <main className="anomalies-main">
        <div className="anomalies-content">
          <div className="page-header">
            <h1 className="page-title">Anomaly Detection</h1>
            <p className="page-description">Review and resolve flagged attendance entries</p>
          </div>

          {/* Stats Cards */}
          <div className="stats-grid">
            <Card>
              <CardHeader className="compact">
                <CardTitle className="small">Pending Review</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="stat-value text-warning">{pendingCount}</div>
                <p className="stat-label">Requires attention</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="compact">
                <CardTitle className="small">High Severity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="stat-value text-destructive">{highSeverityCount}</div>
                <p className="stat-label">Critical issues</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="compact">
                <CardTitle className="small">Resolved</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="stat-value text-success">{resolvedCount}</div>
                <p className="stat-label">Successfully handled</p>
              </CardContent>
            </Card>
          </div>

          {/* Filters */}
          <Card className="filter-card">
            <CardContent className="filter-content">
              <div className="filter-buttons">
                <Button
                  variant={filter === "all" ? "default" : "outline"}
                  onClick={() => setFilter("all")}
                  className={filter !== "all" ? "transparent" : ""}
                >
                  All ({anomalies.length})
                </Button>
                <Button
                  variant={filter === "pending" ? "default" : "outline"}
                  onClick={() => setFilter("pending")}
                  className={filter !== "pending" ? "transparent" : ""}
                >
                  Pending ({pendingCount})
                </Button>
                <Button
                  variant={filter === "resolved" ? "default" : "outline"}
                  onClick={() => setFilter("resolved")}
                  className={filter !== "resolved" ? "transparent" : ""}
                >
                  Resolved ({resolvedCount})
                </Button>
                <Button
                  variant={filter === "dismissed" ? "default" : "outline"}
                  onClick={() => setFilter("dismissed")}
                  className={filter !== "dismissed" ? "transparent" : ""}
                >
                  Dismissed ({dismissedCount})
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Anomalies List */}
          <div className="anomalies-list">
            {filteredAnomalies.map((anomaly) => (
              <Card key={anomaly.id} className="anomaly-card">
                <CardContent className="anomaly-content">
                  <div className="anomaly-inner">
                    <div className="anomaly-left">
                      <div className={`anomaly-icon ${getTypeColor(anomaly.type)}`}>
                        {getTypeIcon(anomaly.type)}
                      </div>
                      <div className="anomaly-info">
                        <div className="anomaly-header">
                          <h3 className="anomaly-subject">{anomaly.subject}</h3>
                          <Badge variant={getSeverityColor(anomaly.severity)}>{anomaly.severity}</Badge>
                          <Badge variant="outline" className="type-badge">
                            {anomaly.type}
                          </Badge>
                        </div>
                        <p className="anomaly-description">{anomaly.description}</p>
                        <div className="anomaly-meta">
                          <span>
                            <span className="meta-label">Roll No:</span> {anomaly.rollNumber}
                          </span>
                          <span>
                            <span className="meta-label">Student:</span> {anomaly.studentName}
                          </span>
                          <span>
                            <span className="meta-label">Date:</span> {anomaly.date}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="anomaly-actions">
                      {anomaly.status === "pending" ? (
                        <>
                          <Button variant="outline" size="sm" className="transparent">
                            <Eye className="icon-xs" />
                            View
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleResolve(anomaly.id)}
                            className="transparent"
                          >
                            <CheckCircle2 className="icon-xs" />
                            Resolve
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleDismiss(anomaly.id)}
                            className="transparent"
                          >
                            <X className="icon-xs" />
                            Dismiss
                          </Button>
                        </>
                      ) : (
                        <Badge
                          variant={anomaly.status === "resolved" ? "default" : "secondary"}
                          className={anomaly.status === "resolved" ? "badge-resolved" : ""}
                        >
                          {anomaly.status === "resolved" ? (
                            <>
                              <CheckCircle2 className="icon-xs" />
                              Resolved
                            </>
                          ) : (
                            "Dismissed"
                          )}
                        </Badge>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Anomaly Types Guide */}
          <Card className="guide-card">
            <CardHeader>
              <CardTitle>Anomaly Types</CardTitle>
              <CardDescription>Understanding different types of detected anomalies</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="guide-grid">
                <div className="guide-item">
                  <div className="guide-item-content">
                    <div className="guide-icon type-destructive">
                      <Copy className="icon-sm" />
                    </div>
                    <div>
                      <p className="guide-title">Duplicate Entry</p>
                      <p className="guide-description">Same student marked multiple times</p>
                    </div>
                  </div>
                </div>
                <div className="guide-item">
                  <div className="guide-item-content">
                    <div className="guide-icon type-warning">
                      <Users className="icon-sm" />
                    </div>
                    <div>
                      <p className="guide-title">Signature Mismatch</p>
                      <p className="guide-description">Suspicious or inconsistent signatures</p>
                    </div>
                  </div>
                </div>
                <div className="guide-item">
                  <div className="guide-item-content">
                    <div className="guide-icon type-primary">
                      <AlertTriangle className="icon-sm" />
                    </div>
                    <div>
                      <p className="guide-title">Inconsistent Marking</p>
                      <p className="guide-description">Unclear or ambiguous presence indicators</p>
                    </div>
                  </div>
                </div>
                <div className="guide-item">
                  <div className="guide-item-content">
                    <div className="guide-icon type-muted">
                      <X className="icon-sm" />
                    </div>
                    <div>
                      <p className="guide-title">Invalid Data</p>
                      <p className="guide-description">Incorrect roll numbers or missing information</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}