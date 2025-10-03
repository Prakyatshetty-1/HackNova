"use client";

import React, { useState, useCallback, useMemo } from "react";
import Sidebar from "../components/sidebar";
import {
  Upload,
  FileImage,
  FileText,
  X,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import "../styles/Uploads.css";
import { getAllSubjects, getSubjectsBySemester } from "../data/data";

// ============= Component Definitions =============

// Card Components
const Card = ({ children, className = "" }) => (
  <div className={`card ${className}`}>{children}</div>
);

const CardHeader = ({ children, className = "" }) => (
  <div className={`card-header ${className}`}>{children}</div>
);

const CardTitle = ({ children, className = "" }) => (
  <h3 className={`card-title ${className}`}>{children}</h3>
);

const CardDescription = ({ children, className = "" }) => (
  <p className={`card-description ${className}`}>{children}</p>
);

const CardContent = ({ children, className = "" }) => (
  <div className={`card-content ${className}`}>{children}</div>
);

// Button Component
const Button = ({
  children,
  variant = "default",
  size = "default",
  className = "",
  onClick,
  ...props
}) => (
  <button
    className={`btn btn-${variant} btn-${size} ${className}`}
    onClick={onClick}
    {...props}
  >
    {children}
  </button>
);

// Badge Component
const Badge = ({ children, variant = "default", className = "" }) => (
  <span className={`badge badge-${variant} ${className}`}>{children}</span>
);

// ============= Main Component =============

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedYear, setSelectedYear] = useState("");
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedSubject, setSelectedSubject] = useState("");

  // Get filtered subjects based on selected semester
  const availableSubjects = useMemo(() => {
    if (selectedYear) {
      return getSubjectsBySemester(selectedYear);
    }
    return getAllSubjects();
  }, [selectedYear]);

  // Reset subject when semester changes
  const handleSemesterChange = (e) => {
    const newSemester = e.target.value;
    setSelectedYear(newSemester);
    setSelectedSubject(""); // Reset subject selection
  };

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    processFiles(droppedFiles);
  }, []);

  const handleFileInput = useCallback((e) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      processFiles(selectedFiles);
    }
  }, []);

  const processFiles = (fileList) => {
    const newFiles = fileList.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      status: "pending",
      progress: 0,
    }));
    setFiles((prev) => [...prev, ...newFiles]);

    newFiles.forEach((file) => {
      setTimeout(() => {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id ? { ...f, status: "processing", progress: 50 } : f
          )
        );
      }, 500);

      setTimeout(() => {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id ? { ...f, status: "completed", progress: 100 } : f
          )
        );
      }, 2000);
    });
  };

  const removeFile = (id) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        <div className="container">
          <div className="page-header">
            <h1 className="page-title">Upload Attendance</h1>
            <p className="page-description">
              Upload scanned attendance sheets for processing
            </p>
          </div>

          {/* Selection Panel */}
          <Card className="selection-panel">
            <CardHeader>
              <CardTitle>Select Details</CardTitle>
              <CardDescription>
                Choose academic year, class, and subject before uploading
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="selection-grid">
                <div className="form-group">
                  <label className="form-label">Academic Year</label>
                  <select
                    className="form-select"
                    value={selectedYear}
                    onChange={handleSemesterChange}
                  >
                    <option value="">Select Semester</option>
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
                    <option value="">Select Class</option>
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
                      {selectedYear 
                        ? `Select Subject (${availableSubjects.length} available)`
                        : "Select Subject (All subjects)"}
                    </option>
                    {availableSubjects.map((subject) => (
                      <option key={subject.value} value={subject.value}>
                        {subject.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Upload Area */}
          <Card className="upload-card">
            <CardHeader>
              <CardTitle>Upload Files</CardTitle>
              <CardDescription>
                Drag and drop or click to select attendance sheets (PDF, JPG,
                PNG)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`upload-area ${isDragging ? "dragging" : ""}`}
              >
                <div className="upload-icon-wrapper">
                  <Upload className="upload-icon" />
                </div>
                <h3 className="upload-title">Drop files here</h3>
                <p className="upload-subtitle">or click to browse</p>
                <input
                  type="file"
                  multiple
                  accept=".pdf,.jpg,.jpeg,.png"
                  onChange={handleFileInput}
                  className="upload-input"
                />
                <Button variant="outline" className="upload-button">
                  Select Files
                </Button>
                <p className="upload-info">
                  Supported formats: PDF, JPG, PNG (Max 10MB each)
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Uploaded Files List */}
          {files.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Uploaded Files ({files.length})</CardTitle>
                <CardDescription>
                  Processing status of uploaded attendance sheets
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="files-list">
                  {files.map((file) => (
                    <div key={file.id} className="file-item">
                      <div className="file-item-content">
                        <div className="file-icon-wrapper">
                          {file.type.includes("pdf") ? (
                            <FileText className="file-icon" />
                          ) : (
                            <FileImage className="file-icon" />
                          )}
                        </div>
                        <div className="file-info">
                          <p className="file-name">{file.name}</p>
                          <p className="file-size">
                            {formatFileSize(file.size)}
                          </p>
                        </div>
                        <div className="file-status">
                          {file.status === "pending" && (
                            <Badge
                              variant="secondary"
                              className="status-pending"
                            >
                              Pending
                            </Badge>
                          )}
                          {file.status === "processing" && (
                            <Badge
                              variant="secondary"
                              className="status-processing"
                            >
                              <Loader2 className="spinner" />
                              Processing
                            </Badge>
                          )}
                          {file.status === "completed" && (
                            <Badge className="status-completed">
                              <CheckCircle2 className="status-icon" />
                              Completed
                            </Badge>
                          )}
                          {file.status === "error" && (
                            <Badge variant="destructive">Error</Badge>
                          )}
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => removeFile(file.id)}
                            className="remove-btn"
                          >
                            <X className="remove-icon" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                {files.some((f) => f.status === "completed") && (
                  <div className="file-actions">
                    <Button variant="outline" onClick={() => setFiles([])}>
                      Clear All
                    </Button>
                    <Button>View Processed Reports</Button>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Processing Guidelines */}
          <Card className="guidelines-card">
            <CardHeader>
              <CardTitle>Processing Guidelines</CardTitle>
              <CardDescription>
                Best practices for accurate attendance extraction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="guidelines-grid">
                <div className="guideline-section">
                  <h4 className="guideline-title recommended">
                    <CheckCircle2 className="guideline-icon" />
                    Recommended
                  </h4>
                  <ul className="guideline-list">
                    <li>• High-resolution scans (300 DPI or higher)</li>
                    <li>• Clear, well-lit images</li>
                    <li>• Straight alignment of sheets</li>
                    <li>• Complete table borders visible</li>
                  </ul>
                </div>
                <div className="guideline-section">
                  <h4 className="guideline-title avoid">
                    <X className="guideline-icon" />
                    Avoid
                  </h4>
                  <ul className="guideline-list">
                    <li>• Blurry or low-quality images</li>
                    <li>• Shadows or glare on sheets</li>
                    <li>• Rotated or skewed scans</li>
                    <li>• Partial or cropped tables</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}