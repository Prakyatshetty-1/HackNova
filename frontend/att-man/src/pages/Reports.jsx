import { Download, Search, FileText, TrendingUp, TrendingDown, Calendar, Filter, X } from "lucide-react"
import { useState, useMemo } from "react"
import Sidebar from "../components/sidebar"
import "../styles/Reports.css"
import { getAllSubjects, getSubjectsBySemester } from "../data/data"

const mockReports = [
  {
    id: "1",
    subject: "Data Structure",
    semester: "Sem 3",
    class: "A",
    totalStudents: 120,
    totalLectures: 45,
    avgAttendance: 87.5,
    lastUpdated: "2024-03-10",
    trend: "up",
  },
  {
    id: "2",
    subject: "Computer Network",
    semester: "Sem 5",
    class: "B",
    totalStudents: 115,
    totalLectures: 42,
    avgAttendance: 82.3,
    lastUpdated: "2024-03-09",
    trend: "down",
  },
  {
    id: "3",
    subject: "Operating System",
    semester: "Sem 4",
    class: "A",
    totalStudents: 118,
    totalLectures: 48,
    avgAttendance: 91.2,
    lastUpdated: "2024-03-10",
    trend: "up",
  },
  {
    id: "4",
    subject: "Database Management System",
    semester: "Sem 4",
    class: "C",
    totalStudents: 122,
    totalLectures: 40,
    avgAttendance: 78.9,
    lastUpdated: "2024-03-08",
    trend: "down",
  },
  {
    id: "5",
    subject: "Software Engineering",
    semester: "Sem 5",
    class: "A",
    totalStudents: 110,
    totalLectures: 44,
    avgAttendance: 85.6,
    lastUpdated: "2024-03-10",
    trend: "up",
  },
  {
    id: "6",
    subject: "Digital Logic & Computer Architecture",
    semester: "Sem 3",
    class: "B",
    totalStudents: 125,
    totalLectures: 46,
    avgAttendance: 89.4,
    lastUpdated: "2024-03-09",
    trend: "up",
  },
  {
    id: "7",
    subject: "Machine Learning",
    semester: "Sem 7",
    class: "A",
    totalStudents: 98,
    totalLectures: 38,
    avgAttendance: 92.1,
    lastUpdated: "2024-03-10",
    trend: "up",
  },
  {
    id: "8",
    subject: "Artificial Intelligence",
    semester: "Sem 6",
    class: "C",
    totalStudents: 105,
    totalLectures: 41,
    avgAttendance: 88.7,
    lastUpdated: "2024-03-09",
    trend: "up",
  },
]

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
const Button = ({ children, variant = "default", size = "default", className = "", onClick }) => (
  <button className={`btn btn-${variant} btn-${size} ${className}`} onClick={onClick}>
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

export default function ReportsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedReport, setSelectedReport] = useState(null)
  const [showFilters, setShowFilters] = useState(false)
  
  // Filter states
  const [selectedSemester, setSelectedSemester] = useState("")
  const [selectedClass, setSelectedClass] = useState("")
  const [selectedSubject, setSelectedSubject] = useState("")

  // Get available subjects based on selected semester
  const availableSubjects = useMemo(() => {
    if (selectedSemester) {
      return getSubjectsBySemester(selectedSemester)
    }
    return getAllSubjects()
  }, [selectedSemester])

  // Filter reports based on all criteria
  const filteredReports = useMemo(() => {
    return mockReports.filter((report) => {
      const matchesSearch = report.subject.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesSemester = !selectedSemester || report.semester === selectedSemester
      const matchesClass = !selectedClass || report.class === selectedClass
      const matchesSubject = !selectedSubject || report.subject.toLowerCase().includes(selectedSubject.toLowerCase())
      
      return matchesSearch && matchesSemester && matchesClass && matchesSubject
    })
  }, [searchQuery, selectedSemester, selectedClass, selectedSubject])

  // Clear all filters
  const clearFilters = () => {
    setSelectedSemester("")
    setSelectedClass("")
    setSelectedSubject("")
    setSearchQuery("")
  }

  // Check if any filter is active
  const hasActiveFilters = selectedSemester || selectedClass || selectedSubject || searchQuery

  const getAttendanceColor = (percentage) => {
    if (percentage >= 85) return "text-success"
    if (percentage >= 75) return "text-warning"
    return "text-destructive"
  }

  const totalSubjects = filteredReports.length
  const avgAttendance = filteredReports.length > 0 
    ? (filteredReports.reduce((acc, r) => acc + r.avgAttendance, 0) / filteredReports.length).toFixed(1)
    : 0
  const totalStudents = filteredReports.reduce((acc, r) => acc + r.totalStudents, 0)
  const totalLectures = filteredReports.reduce((acc, r) => acc + r.totalLectures, 0)

  return (
    <div className="reports-container">
      <Sidebar />
      <main className="reports-main">
        <div className="reports-content">
          <div className="page-header">
            <h1 className="page-title">Reports</h1>
            <p className="page-description">View and download attendance reports</p>
          </div>

          {/* Summary Cards */}
          

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
                <Button>
                  <Download className="btn-icon" />
                  Export
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
                          setSelectedSubject("") // Reset subject when semester changes
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

          {/* Reports List */}
          {filteredReports.length === 0 ? (
            <Card className="empty-state">
              <CardContent>
                <div className="empty-state-content">
                  <FileText className="empty-icon" />
                  <h3 className="empty-title">No reports found</h3>
                  <p className="empty-description">
                    Try adjusting your filters or search query
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
              {filteredReports.map((report) => (
                <Card
                  key={report.id}
                  className="report-card"
                  onClick={() => setSelectedReport(report)}
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
                            {report.trend === "up" ? (
                              <TrendingUp className="trend-icon trend-up" />
                            ) : (
                              <TrendingDown className="trend-icon trend-down" />
                            )}
                          </div>
                          <div className="report-meta">
                            <span>{report.semester}</span>
                            <span>Class {report.class}</span>
                            <span>{report.totalStudents} students</span>
                            <span>{report.totalLectures} lectures</span>
                          </div>
                        </div>
                      </div>
                      <div className="report-right">
                        <div className="attendance-display">
                          <div className={`attendance-value ${getAttendanceColor(report.avgAttendance)}`}>
                            {report.avgAttendance}%
                          </div>
                          <p className="attendance-label">Avg. Attendance</p>
                        </div>
                        <Button variant="outline" size="sm" className="download-btn">
                          <Download className="btn-icon-small" />
                          Download
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {/* Department Summary */}
          <Card className="department-card">
            <CardHeader>
              <CardTitle>Department Summary</CardTitle>
              <CardDescription>
                {hasActiveFilters 
                  ? "Statistics for filtered results" 
                  : "Overall attendance statistics for Computer Engineering"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="summary-items">
                <div className="summary-item">
                  <div>
                    <p className="summary-title">Total Lectures Conducted</p>
                    <p className="summary-subtitle">
                      {hasActiveFilters ? "In filtered results" : "Across all subjects"}
                    </p>
                  </div>
                  <div className="summary-value">{totalLectures}</div>
                </div>
                <div className="summary-item">
                  <div>
                    <p className="summary-title">Students Above 75%</p>
                    <p className="summary-subtitle">Meeting attendance requirement</p>
                  </div>
                  <div className="summary-value-group">
                    <div className="summary-value text-success">2,547</div>
                    <Badge className="badge-success">89.5%</Badge>
                  </div>
                </div>
                <div className="summary-item no-border">
                  <div>
                    <p className="summary-title">Students Below 75%</p>
                    <p className="summary-subtitle">Require attention</p>
                  </div>
                  <div className="summary-value-group">
                    <div className="summary-value text-destructive">300</div>
                    <Badge variant="destructive">10.5%</Badge>
                  </div>
                </div>
              </div>
              <div className="department-action">
                <Button className="full-width">
                  <Download className="btn-icon" />
                  {hasActiveFilters ? "Download Filtered Report" : "Download Department Report"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}