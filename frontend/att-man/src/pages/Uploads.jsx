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
  Download,
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
  disabled = false,
  ...props
}) => (
  <button
    className={`btn btn-${variant} btn-${size} ${className}`}
    onClick={onClick}
    disabled={disabled}
    {...props}
  >
    {children}
  </button>
);

// Badge Component
const Badge = ({ children, variant = "default", className = "" }) => (
  <span className={`badge badge-${variant} ${className}`}>{children}</span>
);

// Alert Component
const Alert = ({ children, variant = "default", className = "" }) => (
  <div className={`alert alert-${variant} ${className}`}>{children}</div>
);

// ============= Main Component =============

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedYear, setSelectedYear] = useState("");
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedSubject, setSelectedSubject] = useState("");
  const [alertMessage, setAlertMessage] = useState(null);

  const API_BASE_URL = "http://localhost:5000/api";

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

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    processFiles(droppedFiles);
  };

  const handleFileInput = (e) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      processFiles(selectedFiles);
    }
  };

  const processFiles = async (fileList) => {
    // Validate selections
    if (!selectedYear || !selectedClass || !selectedSubject) {
      setAlertMessage({
        type: "error",
        message: "Please select Academic Year, Class, and Subject before uploading files.",
      });
      setTimeout(() => setAlertMessage(null), 5000);
      return;
    }

    const newFiles = fileList.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      file: file, // Store the actual file object
      status: "pending",
      progress: 0,
      downloadUrl: null,
      pdfDownloadUrl: null,
      summary: null,
      preview: null,
      recordId: null,
    }));

    setFiles((prev) => [...prev, ...newFiles]);

    // Process each file
    for (const fileData of newFiles) {
      await uploadToAPI(fileData);
    }
  };

  const uploadToAPI = async (fileData) => {
    try {
      // Update status to processing
      setFiles((prev) =>
        prev.map((f) =>
          f.id === fileData.id ? { ...f, status: "processing", progress: 50 } : f
        )
      );

      // Create FormData
      const formData = new FormData();
      formData.append("file", fileData.file);
      formData.append("selectedYear", selectedYear);
      formData.append("selectedClass", selectedClass);
      formData.append("selectedSubject", selectedSubject);

      // Send to Flask API (which now uploads to Supabase)
      const response = await fetch(`${API_BASE_URL}/process-attendance`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        // Update file status with Supabase URLs
        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileData.id
              ? {
                  ...f,
                  status: "completed",
                  progress: 100,
                  downloadUrl: result.csv_url || `${API_BASE_URL}/download/${result.filename}`,
                  pdfDownloadUrl: result.pdf_url || (result.pdf_filename ? `${API_BASE_URL}/download/${result.pdf_filename}` : null),
                  summary: result.summary,
                  preview: result.preview,
                  filename: result.filename,
                  pdfFilename: result.pdf_filename,
                  recordId: result.record_id, // Store Supabase record ID
                  csvUrl: result.csv_url,
                  pdfUrl: result.pdf_url,
                }
              : f
          )
        );

        const successMsg = result.csv_url 
          ? `Successfully processed and stored ${fileData.name} in Supabase` 
          : `Successfully processed ${fileData.name} (stored locally)`;
        
        setAlertMessage({
          type: "success",
          message: successMsg,
        });
        setTimeout(() => setAlertMessage(null), 5000);
      } else {
        throw new Error(result.error || "Processing failed");
      }
    } catch (error) {
      console.error("Upload error:", error);
      setFiles((prev) =>
        prev.map((f) =>
          f.id === fileData.id
            ? { ...f, status: "error", progress: 0, errorMessage: error.message }
            : f
        )
      );

      setAlertMessage({
        type: "error",
        message: `Failed to process ${fileData.name}: ${error.message}`,
      });
      setTimeout(() => setAlertMessage(null), 5000);
    }
  };

  const removeFile = async (id, recordId) => {
    // If file has a Supabase record ID, delete from Supabase
    if (recordId) {
      try {
        const response = await fetch(`${API_BASE_URL}/delete-upload/${recordId}`, {
          method: 'DELETE',
        });
        const data = await response.json();
        
        if (data.success) {
          setAlertMessage({
            type: "success",
            message: "File deleted from Supabase successfully",
          });
          setTimeout(() => setAlertMessage(null), 3000);
        } else {
          setAlertMessage({
            type: "error",
            message: "Failed to delete from Supabase",
          });
          setTimeout(() => setAlertMessage(null), 3000);
          return; // Don't remove from UI if deletion failed
        }
      } catch (error) {
        console.error("Delete error:", error);
        setAlertMessage({
          type: "error",
          message: `Failed to delete: ${error.message}`,
        });
        setTimeout(() => setAlertMessage(null), 3000);
        return; // Don't remove from UI if deletion failed
      }
    }
    
    // Remove from local state
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const downloadFile = (downloadUrl, filename) => {
    // If it's a Supabase URL, open in new tab
    if (downloadUrl && downloadUrl.includes('supabase')) {
      window.open(downloadUrl, '_blank');
    } else {
      // Local download
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
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
              Upload scanned attendance sheets for processing and storage in Supabase
            </p>
          </div>

          {/* Alert Message */}
          {alertMessage && (
            <Alert variant={alertMessage.type} className="mb-4">
              {alertMessage.message}
            </Alert>
          )}

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
                  <label className="form-label">Academic Year *</label>
                  <select
                    className="form-select"
                    value={selectedYear}
                    onChange={handleSemesterChange}
                    required
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
                  <label className="form-label">Class *</label>
                  <select
                    className="form-select"
                    value={selectedClass}
                    onChange={(e) => setSelectedClass(e.target.value)}
                    required
                  >
                    <option value="">Select Class</option>
                    <option value="A">A</option>
                    <option value="B">B</option>
                    <option value="C">C</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Subject *</label>
                  <select
                    className="form-select"
                    value={selectedSubject}
                    onChange={(e) => setSelectedSubject(e.target.value)}
                    required
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
                Drag and drop or click to select attendance sheets (PDF, JPG, PNG)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`upload-area ${isDragging ? "dragging" : ""}`}
                onClick={() => document.getElementById('file-input').click()}
                style={{ cursor: 'pointer' }}
              >
                <div className="upload-icon-wrapper">
                  <Upload className="upload-icon" />
                </div>
                <h3 className="upload-title">Drop files here</h3>
                <p className="upload-subtitle">or click to browse</p>
                <input
                  id="file-input"
                  type="file"
                  multiple
                  accept="image/*,.pdf"
                  onChange={handleFileInput}
                  style={{ display: 'none' }}
                />
                <Button
                  variant="outline"
                  className="upload-button"
                  disabled={!selectedYear || !selectedClass || !selectedSubject}
                  onClick={(e) => {
                    e.stopPropagation();
                    document.getElementById('file-input').click();
                  }}
                >
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
                          {file.recordId && (
                            <p className="file-meta">
                              <small>✓ Stored in Supabase (ID: {file.recordId})</small>
                            </p>
                          )}
                          {file.errorMessage && (
                            <p className="file-error">{file.errorMessage}</p>
                          )}
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
                            <>
                              <Badge className="status-completed">
                                <CheckCircle2 className="status-icon" />
                                Completed
                              </Badge>
                              <div className="button-group">
                                {file.downloadUrl && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() =>
                                      downloadFile(file.downloadUrl, file.filename)
                                    }
                                    className="download-btn"
                                    title={file.csvUrl ? "Download from Supabase" : "Download locally"}
                                  >
                                    <Download className="download-icon" />
                                    CSV
                                  </Button>
                                )}
                                {file.pdfDownloadUrl && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() =>
                                      downloadFile(file.pdfDownloadUrl, file.pdfFilename)
                                    }
                                    className="download-btn pdf-btn"
                                    title={file.pdfUrl ? "Download from Supabase" : "Download locally"}
                                  >
                                    <Download className="download-icon" />
                                    PDF Report
                                  </Button>
                                )}
                              </div>
                            </>
                          )}
                          {file.status === "error" && (
                            <Badge variant="destructive">Error</Badge>
                          )}
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => removeFile(file.id, file.recordId)}
                            className="remove-btn"
                            title={file.recordId ? "Delete from Supabase" : "Remove from list"}
                          >
                            <X className="remove-icon" />
                          </Button>
                        </div>
                      </div>
                      
                      {/* Show preview for completed files */}
                      {file.status === "completed" && file.preview && (
                        <div className="file-preview">
                          <h4>CSV Preview:</h4>
                          <pre>{file.preview.join("\n")}</pre>
                        </div>
                      )}

                      {/* Show summary if available */}
                      {file.status === "completed" && file.summary && Object.keys(file.summary).length > 0 && (
                        <div className="file-summary">
                          <h4>Attendance Summary:</h4>
                          {Object.entries(file.summary).map(([date, stats]) => (
                            <div key={date} className="summary-item">
                              <strong>{date}:</strong> Present: {stats.present}/{stats.total}, 
                              Absent: {stats.absent}/{stats.total} 
                              ({stats.attendance_percentage}%)
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                {files.some((f) => f.status === "completed") && (
                  <div className="file-actions">
                    <Button variant="outline" onClick={() => setFiles([])}>
                      Clear All
                    </Button>
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
                    <li>• Files automatically stored in Supabase</li>
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