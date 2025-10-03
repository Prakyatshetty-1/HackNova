import { Download, Search, FileText, TrendingUp, TrendingDown, Calendar, Filter, X, Loader2 } from "lucide-react"
import { useState, useMemo, useEffect } from "react"
import Sidebar from "../components/sidebar"
import "../styles/Reports.css"
import { getAllSubjects, getSubjectsBySemester } from "../data/data"

// Card Components
const Card = ({ children, className = "", onClick }) => (
  <div className={`card ${className}`} onClick={onClick}>{children}</div>
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
const Button = ({ children, variant = "default", size = "default", className = "", onClick, disabled = false }) => (
  <button className={`btn btn-${variant} btn-${size} ${className}`} onClick={onClick} disabled={disabled}>
    {children}
  </button>
)

// Badge Component
const Badge = ({ children, variant = "default", className = "" }) => (
  <span className={`badge badge-${variant} ${className}`}>{children}</span>
)

// Input Component
const Input = ({ className = "", ...props }) => (
  <input className={`input ${className}`} {...props} />
)

// Alert Component
const Alert = ({ children, variant = "default", className = "" }) => (
  <div className={`alert alert-${variant} ${className}`}>{children}</div>
)

export default function ReportsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedReport, setSelectedReport] = useState(null)
  const [showFilters, setShowFilters] = useState(false)
  
  // Filter states
  const [selectedSemester, setSelectedSemester] = useState("")
  const [selectedClass, setSelectedClass] = useState("")
  const [selectedSubject, setSelectedSubject] = useState("")

  // Data states
  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const API_BASE_URL = "http://localhost:5000/api"

  // Fetch reports from Supabase
  useEffect(() => {
    fetchReports()
  }, [])

  const fetchReports = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await fetch(`${API_BASE_URL}/get-uploads`)
      const data = await response.json()
      
      if (data.success) {
        setReports(data.data)
      } else {
        setError(data.error || "Failed to fetch reports")
      }
    } catch (err) {
      console.error("Error fetching reports:", err)
      setError("Failed to connect to server")
    } finally {
      setLoading(false)
    }
  }

  // Get available subjects based on selected semester
  const availableSubjects = useMemo(() => {
    if (selectedSemester) {
      return getSubjectsBySemester(selectedSemester)
    }
    return getAllSubjects()
  }, [selectedSemester])

  // Filter reports based on all criteria
  const filteredReports = useMemo(() => {
    return reports.filter((report) => {
      const matchesSearch = report.subject?.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesSemester = !selectedSemester || report.semester === selectedSemester
      const matchesClass = !selectedClass || report.class === selectedClass
      const matchesSubject = !selectedSubject || report.subject?.toLowerCase().includes(selectedSubject.toLowerCase())
      
      return matchesSearch && matchesSemester && matchesClass && matchesSubject
    })
  }, [reports, searchQuery, selectedSemester, selectedClass, selectedSubject])

  // Clear all filters
  const clearFilters = () => {
    setSelectedSemester("")
    setSelectedClass("")
    setSelectedSubject("")
    setSearchQuery("")
  }

  // Check if any filter is active
  const hasActiveFilters = selectedSemester || selectedClass || selectedSubject || searchQuery

  // Download file from Supabase
  const handleDownload = (url, filename) => {
    if (url) {
      // Open Supabase URL in new tab
      window.open(url, '_blank')
    } else {
      alert("File URL not available")
    }
  }

  // Calculate statistics
  const calculateAvgAttendance = (summary) => {
    if (!summary || typeof summary !== 'object') return 0
    
    const dates = Object.keys(summary)
    if (dates.length === 0) return 0
    
    const totalPercentage = dates.reduce((sum, date) => {
      return sum + (parseFloat(summary[date]?.attendance_percentage) || 0)
    }, 0)
    
    return (totalPercentage / dates.length).toFixed(1)
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    })
  }

  const getAttendanceColor = (percentage) => {
    if (percentage >= 85) return "text-success"
    if (percentage >= 75) return "text-warning"
    return "text-destructive"
  }

  const totalReports = filteredReports.length
  const avgAttendance = filteredReports.length > 0 
    ? (filteredReports.reduce((acc, r) => acc + parseFloat(calculateAvgAttendance(r.summary)), 0) / filteredReports.length).toFixed(1)
    : 0

  return (
    <div className="reports-container">
      <Sidebar />
      <main className="reports-main">
        <div className="reports-content">
          <div className="page-header">
            <h1 className="page-title">Reports</h1>
            <p className="page-description">View and download attendance reports from Supabase</p>
          </div>

          {/* Error Alert */}
          {error && (
            <Alert variant="error" className="mb-4">
              {error}
              <Button variant="ghost" size="sm" onClick={fetchReports} className="ml-2">
                Retry
              </Button>
            </Alert>
          )}

          {/* Search and Filters */}
          <Card className="search-card">
            <CardContent className="search-content">
              <div className="search-bar-container">
                <div className="search-input-wrapper">
                  <Search className="search-icon" />
                  <Input
                    placeholder="Search subjects..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="search-input"
                  />
                </div>
                <Button 
                  variant={showFilters ? "default" : "outline"} 
                  className="filter-btn"
                  onClick={() => setShowFilters(!showFilters)}
                >
                  <Filter className="btn-icon" />
                  Filters
                  {hasActiveFilters && <Badge className="filter-badge">{
                    [selectedSemester, selectedClass, selectedSubject].filter(Boolean).length
                  }</Badge>}
                </Button>
                {hasActiveFilters && (
                  <Button variant="ghost" onClick={clearFilters}>
                    <X className="btn-icon" />
                    Clear
                  </Button>
                )}
                <Button onClick={fetchReports} disabled={loading}>
                  {loading ? <Loader2 className="btn-icon animate-spin" /> : <Download className="btn-icon" />}
                  Refresh
                </Button>
              </div>

              {/* Filter Panel */}
              {showFilters && (
                <div className="filter-panel">
                  <div className="filter-grid">
                    <div className="form-group">
                      <label className="form-label">Semester</label>
                      <select
                        className="form-select"
                        value={selectedSemester}
                        onChange={(e) => {
                          setSelectedSemester(e.target.value)
                          setSelectedSubject("")
                        }}
                      >
                        <option value="">All Semesters</option>
                        <option value="Sem 1">Sem 1</option>
                        <option value="Sem 2">Sem 2</option>
                        <option value="Sem 3">Sem 3</option>
                        <option value="Sem 4">Sem 4</option>
                        <option value="Sem 5">Sem 5</option>
                        <option value="Sem 6">Sem 6</option>
                        <option value="Sem 7">Sem 7</option>
                        <option value="Sem 8">Sem 8</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Class</label>
                      <select
                        className="form-select"
                        value={selectedClass}
                        onChange={(e) => setSelectedClass(e.target.value)}
                      >
                        <option value="">All Classes</option>
                        <option value="A">A</option>
                        <option value="B">B</option>
                        <option value="C">C</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Subject</label>
                      <select
                        className="form-select"
                        value={selectedSubject}
                        onChange={(e) => setSelectedSubject(e.target.value)}
                      >
                        <option value="">
                          {selectedSemester 
                            ? `All Subjects (${availableSubjects.length})`
                            : "All Subjects"}
                        </option>
                        {availableSubjects.map((subject) => (
                          <option key={subject.value} value={subject.value}>
                            {subject.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Loading State */}
          {loading ? (
            <Card className="empty-state">
              <CardContent>
                <div className="empty-state-content">
                  <Loader2 className="empty-icon animate-spin" />
                  <h3 className="empty-title">Loading reports...</h3>
                  <p className="empty-description">Fetching data from Supabase</p>
                </div>
              </CardContent>
            </Card>
          ) : filteredReports.length === 0 ? (
            <Card className="empty-state">
              <CardContent>
                <div className="empty-state-content">
                  <FileText className="empty-icon" />
                  <h3 className="empty-title">No reports found</h3>
                  <p className="empty-description">
                    {hasActiveFilters 
                      ? "Try adjusting your filters or search query"
                      : "Upload some attendance files to see reports here"}
                  </p>
                  {hasActiveFilters && (
                    <Button onClick={clearFilters} className="empty-action">
                      Clear all filters
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="reports-list">
              {filteredReports.map((report) => {
                const avgAttendanceValue = calculateAvgAttendance(report.summary)
                
                return (
                  <Card
                    key={report.id}
                    className="report-card"
                  >
                    <CardContent className="report-content">
                      <div className="report-inner">
                        <div className="report-left">
                          <div className="report-icon">
                            <FileText className="icon" />
                          </div>
                          <div className="report-info">
                            <div className="report-header">
                              <h3 className="report-subject">{report.subject}</h3>
                            </div>
                            <div className="report-meta">
                              <span>{report.semester}</span>
                              <span>Class {report.class}</span>
                              <span>{formatDate(report.upload_timestamp)}</span>
                            </div>
                            <div className="report-filename">
                              <small>{report.original_filename}</small>
                            </div>
                          </div>
                        </div>
                        <div className="report-right">
                          <div className="attendance-display">
                            <div className={`attendance-value ${getAttendanceColor(avgAttendanceValue)}`}>
                              {avgAttendanceValue}%
                            </div>
                            <p className="attendance-label">Avg. Attendance</p>
                          </div>
                          <div className="button-group">
                            {report.csv_url && (
                              <Button 
                                variant="outline" 
                                size="sm" 
                                className="download-btn"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleDownload(report.csv_url, report.csv_filename)
                                }}
                              >
                                <Download className="btn-icon-small" />
                                CSV
                              </Button>
                            )}
                            {report.pdf_url && (
                              <Button 
                                variant="outline" 
                                size="sm" 
                                className="download-btn"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleDownload(report.pdf_url, report.pdf_filename)
                                }}
                              >
                                <Download className="btn-icon-small" />
                                PDF
                              </Button>
                            )}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}

          {/* Department Summary */}
          {!loading && filteredReports.length > 0 && (
            <Card className="department-card">
              <CardHeader>
                <CardTitle>Summary Statistics</CardTitle>
                <CardDescription>
                  {hasActiveFilters 
                    ? "Statistics for filtered results" 
                    : "Overall statistics from database"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="summary-items">
                  <div className="summary-item">
                    <div>
                      <p className="summary-title">Total Reports</p>
                      <p className="summary-subtitle">
                        {hasActiveFilters ? "In filtered results" : "Stored in Supabase"}
                      </p>
                    </div>
                    <div className="summary-value">{totalReports}</div>
                  </div>
                  <div className="summary-item">
                    <div>
                      <p className="summary-title">Average Attendance</p>
                      <p className="summary-subtitle">Across all reports</p>
                    </div>
                    <div className="summary-value-group">
                      <div className={`summary-value ${getAttendanceColor(avgAttendance)}`}>
                        {avgAttendance}%
                      </div>
                      <Badge className={avgAttendance >= 75 ? "badge-success" : "badge-destructive"}>
                        {avgAttendance >= 75 ? "Good" : "Needs Attention"}
                      </Badge>
                    </div>
                  </div>
                  <div className="summary-item no-border">
                    <div>
                      <p className="summary-title">Unique Subjects</p>
                      <p className="summary-subtitle">Different subjects tracked</p>
                    </div>
                    <div className="summary-value">
                      {new Set(filteredReports.map(r => r.subject)).size}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}