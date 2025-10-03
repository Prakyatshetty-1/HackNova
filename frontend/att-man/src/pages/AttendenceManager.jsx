"use client"

import { useState } from "react"
import "../styles/AttendenceManager.css"

export default function AttendanceManager() {
  const [files, setFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [selectedYear, setSelectedYear] = useState("")
  const [selectedClass, setSelectedClass] = useState("")

  const handleFileUpload = (uploadedFiles) => {
    if (!uploadedFiles || uploadedFiles.length === 0) return

    const newFiles = Array.from(uploadedFiles).map((file) => ({
      id: Math.random().toString(36).substring(7),
      name: file.name,
      size: file.size,
      uploadDate: new Date(),
      file: file,
      year: selectedYear,
      class: selectedClass,
    }))

    setFiles((prev) => [...prev, ...newFiles])
    alert(`${newFiles.length} file(s) uploaded successfully`)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFileUpload(e.dataTransfer.files)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDownload = (attendanceFile) => {
    const url = URL.createObjectURL(attendanceFile.file)
    const a = document.createElement("a")
    a.href = url
    a.download = attendanceFile.name
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    alert(`Downloading ${attendanceFile.name}`)
  }

  const handleDelete = (id) => {
    setFiles((prev) => prev.filter((file) => file.id !== id))
    alert("File removed successfully")
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i]
  }

  const formatDate = (date) => {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date)
  }

  return (
    <div className="attendance-container">
      {/* Upload Section */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">STEP 1: SELECT YEAR & CLASS</h2>
          <p className="card-description">Choose the year and class before uploading attendance sheets</p>
        </div>
        <div className="card-content">
          <div className="dropdown-container">
            <div className="dropdown-group">
              <label htmlFor="year-select" className="dropdown-label">
                Year
              </label>
              <select
                id="year-select"
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
                className="dropdown-select"
              >
                <option value="">Select Year</option>
                <option value="FE">FE (First Year)</option>
                <option value="SE">SE (Second Year)</option>
                <option value="TE">TE (Third Year)</option>
                <option value="BE">BE (Final Year)</option>
              </select>
            </div>

            <div className="dropdown-group">
              <label htmlFor="class-select" className="dropdown-label">
                Class
              </label>
              <select
                id="class-select"
                value={selectedClass}
                onChange={(e) => setSelectedClass(e.target.value)}
                className="dropdown-select"
              >
                <option value="">Select Class</option>
                <option value="A">Class A</option>
                <option value="B">Class B</option>
                <option value="C">Class C</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">STEP 2: UPLOAD FILE</h2>
          <p className="card-description">Upload your attendance sheets as images or PDF files</p>
        </div>
        <div className="card-content">
          <div className="upload-section">
            <label htmlFor="file-upload" className="upload-box">
              <div className="upload-icon">📤</div>
              <span className="upload-text">Upload file</span>
              <input
                id="file-upload"
                type="file"
                className="file-input"
                multiple
                accept="image/*,application/pdf"
                onChange={(e) => handleFileUpload(e.target.files)}
              />
            </label>
          </div>

          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`drag-drop-zone ${isDragging ? "dragging" : ""}`}
          >
            <p className="drag-drop-text">Drag & Drop a file here</p>
          </div>
        </div>
      </div>

      {/* Files List Section */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">STEP 3: MANAGE ATTENDANCE SHEETS</h2>
          <p className="card-description">View and download your uploaded attendance sheets</p>
        </div>
        <div className="card-content">
          {files.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📄</div>
              <p className="empty-text">No attendance sheets uploaded yet</p>
              <p className="empty-subtext">Upload your first file to get started</p>
            </div>
          ) : (
            <div className="files-list">
              {files.map((file) => (
                <div key={file.id} className="file-item">
                  <div className="file-info">
                    <div className="file-icon">📄</div>
                    <div className="file-details">
                      <p className="file-name">{file.name}</p>
                      <p className="file-meta">
                        {file.year && file.class && (
                          <span className="file-badge">
                            {file.year} - Class {file.class}
                          </span>
                        )}
                        {formatFileSize(file.size)} • Uploaded {formatDate(file.uploadDate)}
                      </p>
                    </div>
                  </div>
                  <div className="file-actions">
                    <button onClick={() => handleDownload(file)} className="btn-download">
                      ⬇️ Download
                    </button>
                    <button onClick={() => handleDelete(file.id)} className="btn-delete">
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
