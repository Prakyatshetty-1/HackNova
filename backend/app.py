from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import google.generativeai as genai
import pandas as pd
import re
import os
from dotenv import load_dotenv
from datetime import datetime
import io
from supabase import create_client, Client

# Import the report generation functions
from generate_attendance_report import compute_attendance, save_report, save_summary, plot_distribution, save_pdf_report

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Storage bucket name
STORAGE_BUCKET = "attendance-files"

PROMPT = """You are analyzing an attendance sheet image. Extract the data and return it in PERFECT CSV format.

RULES:
- Mark "Absent" if cell contains: "A", "Absent", or any absence indicator
- Mark "Present" otherwise
- First column: Roll Number
- Second column: Student Id
- Third column: Name
- Subsequent columns: Dates (format: DD-MM-YYYY)
- Header row: "Roll Number,Student Id,Name" followed by dates

IMPORTANT: Return ONLY the CSV data, no explanations.

Format example:
Roll Number,Student Id,Name,01-01-2024,02-01-2024,03-01-2024
1,STU001,John Doe,Present,Absent,Present
2,STU002,Jane Smith,Absent,Present,Present
"""

def clean_and_save_csv(text_response, filename):
    lines = text_response.strip().split('\n')
    csv_lines = []
    
    for line in lines:
        line = line.strip()
        if ',' in line and any(keyword in line.upper() for keyword in ['STUDENT', 'PRESENT', 'ABSENT', 'P', 'A', 'ROLL', 'NAME', 'ID']):
            csv_lines.append(line)
        elif re.match(r'^[^,]*,[^,]*([,][^,]*)+$', line):
            csv_lines.append(line)
    
    if not csv_lines:
        csv_lines = ["Roll Number,Student Id,Name,Status", "No data,No data,No data extracted,Check image"]
    
    csv_content = '\n'.join(csv_lines)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        f.write(csv_content)
    
    return csv_content

def generate_attendance_summary(df):
    summary = {}
    date_columns = [col for col in df.columns if col not in ['Roll Number', 'Student Id', 'Name', 'Student']]
    
    for date_col in date_columns:
        if date_col in df.columns:
            present_count = (df[date_col].str.upper() == 'PRESENT').sum()
            absent_count = (df[date_col].str.upper() == 'ABSENT').sum()
            total = len(df)
            summary[date_col] = {
                'present': int(present_count),
                'absent': int(absent_count),
                'total': int(total),
                'attendance_percentage': round((present_count / total * 100), 2) if total > 0 else 0
            }
    
    return summary

def generate_pdf_report(csv_filepath, output_dir, defaulter_threshold=75.0):
    """Generate PDF report by directly calling the functions"""
    try:
        # Read the CSV
        try:
            df = pd.read_csv(csv_filepath)
        except Exception:
            df = pd.read_csv(csv_filepath, encoding="latin1")
        
        # Generate report
        report_df, summary = compute_attendance(df, defaulter_threshold=defaulter_threshold)
        
        # Save all outputs
        csv_out = save_report(report_df, output_dir)
        sum_out = save_summary(summary, report_df, output_dir)
        plot_out = plot_distribution(report_df, output_dir)
        pdf_out = save_pdf_report(report_df, summary, output_dir, chart_path=plot_out)
        
        print(f"✅ Report generated: {pdf_out}")
        return pdf_out
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def upload_to_supabase(file_path, storage_path, content_type):
    """Upload file to Supabase Storage"""
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        response = supabase.storage.from_(STORAGE_BUCKET).upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": content_type}
        )
        
        # Get public URL
        public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
        
        return public_url, storage_path
    except Exception as e:
        print(f"❌ Supabase upload failed: {e}")
        raise e

def save_metadata_to_supabase(metadata):
    """Save upload metadata to Supabase database"""
    try:
        response = supabase.table('attendance_uploads').insert(metadata).execute()
        return response.data[0]['id'] if response.data else None
    except Exception as e:
        print(f"❌ Metadata save failed: {e}")
        raise e

@app.route('/api/process-attendance', methods=['POST'])
def process_attendance():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        selected_year = request.form.get('selectedYear', '')
        selected_class = request.form.get('selectedClass', '')
        selected_subject = request.form.get('selectedSubject', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not all([selected_year, selected_class, selected_subject]):
            return jsonify({'error': 'Missing required fields: year, class, or subject'}), 400
        
        # Read file content
        file_content = file.read()
        image_b64 = base64.b64encode(file_content).decode('utf-8')
        
        # Determine mime type
        file_extension = file.filename.lower().split('.')[-1]
        mime_type_map = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'pdf': 'application/pdf'
        }
        mime_type = mime_type_map.get(file_extension, 'image/png')
        
        # Use Gemini Vision model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content([
            PROMPT,
            {
                "mime_type": mime_type,
                "data": base64.b64decode(image_b64),
            }
        ])
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"attendance_{selected_year}_{selected_class}_{selected_subject}_{timestamp}".replace(' ', '_')
        csv_filename = f"{base_filename}.csv"
        csv_filepath = os.path.join('uploads', csv_filename)
        
        # Create uploads directory
        os.makedirs('uploads', exist_ok=True)
        
        # Clean and save CSV locally first
        csv_content = clean_and_save_csv(response.text, csv_filepath)
        
        # Generate summary
        summary = {}
        pdf_filepath = None
        pdf_filename = None
        
        try:
            df = pd.read_csv(csv_filepath)
            summary = generate_attendance_summary(df)
            
            # Generate PDF report
            report_dir = os.path.join('uploads', f'report_{base_filename}')
            pdf_path = generate_pdf_report(csv_filepath, report_dir)
            
            if pdf_path and os.path.exists(pdf_path):
                pdf_filename = f"report_{base_filename}.pdf"
                # Copy PDF to uploads folder
                import shutil
                pdf_filepath = os.path.join('uploads', pdf_filename)
                shutil.copy(pdf_path, pdf_filepath)
                print(f"✅ PDF copied to: {pdf_filepath}")
            else:
                print("❌ PDF generation returned None or file doesn't exist")
        except Exception as e:
            print(f"❌ Summary/PDF generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Upload to Supabase Storage
        folder_path = f"{selected_year}/{selected_class}/{selected_subject}"
        csv_storage_path = f"{folder_path}/{csv_filename}"
        
        try:
            # Upload CSV to Supabase
            csv_url, csv_path = upload_to_supabase(
                csv_filepath, 
                csv_storage_path, 
                "text/csv"
            )
            print(f"✅ CSV uploaded to Supabase: {csv_url}")
            
            # Upload PDF to Supabase if available
            pdf_url = None
            pdf_path_storage = None
            if pdf_filepath and os.path.exists(pdf_filepath):
                pdf_storage_path = f"{folder_path}/{pdf_filename}"
                pdf_url, pdf_path_storage = upload_to_supabase(
                    pdf_filepath, 
                    pdf_storage_path, 
                    "application/pdf"
                )
                print(f"✅ PDF uploaded to Supabase: {pdf_url}")
            
            # Save metadata to Supabase database
            metadata = {
                'semester': selected_year,
                'class': selected_class,
                'subject': selected_subject,
                'original_filename': file.filename,
                'csv_filename': csv_filename,
                'pdf_filename': pdf_filename,
                'csv_url': csv_url,
                'pdf_url': pdf_url,
                'csv_path': csv_path,
                'pdf_path': pdf_path_storage,
                'summary': summary,
                'upload_timestamp': datetime.now().isoformat(),
                'processed': True
            }
            
            record_id = save_metadata_to_supabase(metadata)
            print(f"✅ Metadata saved to Supabase with ID: {record_id}")
            
            return jsonify({
                'success': True,
                'message': 'Attendance processed and uploaded to Supabase successfully',
                'filename': csv_filename,
                'pdf_filename': pdf_filename,
                'csv_url': csv_url,
                'pdf_url': pdf_url,
                'summary': summary,
                'preview': csv_content.split('\n')[:6],
                'record_id': record_id
            })
            
        except Exception as supabase_error:
            print(f"❌ Supabase upload failed: {supabase_error}")
            # Still return local files if Supabase fails
            return jsonify({
                'success': True,
                'message': 'Attendance processed (Supabase upload failed, files saved locally)',
                'filename': csv_filename,
                'pdf_filename': pdf_filename,
                'filepath': csv_filepath,
                'summary': summary,
                'preview': csv_content.split('\n')[:6],
                'supabase_error': str(supabase_error)
            })
    
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        filepath = os.path.join('uploads', filename)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/get-uploads', methods=['GET'])
def get_uploads():
    """Retrieve uploads from Supabase based on filters"""
    try:
        semester = request.args.get('semester')
        class_name = request.args.get('class')
        subject = request.args.get('subject')
        
        # Build query
        query = supabase.table('attendance_uploads').select('*')
        
        if semester:
            query = query.eq('semester', semester)
        if class_name:
            query = query.eq('class', class_name)
        if subject:
            query = query.eq('subject', subject)
        
        response = query.order('upload_timestamp', desc=True).execute()
        
        return jsonify({
            'success': True,
            'data': response.data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/delete-upload/<int:record_id>', methods=['DELETE'])
def delete_upload(record_id):
    """Delete upload from Supabase storage and database"""
    try:
        # Get record to find file paths
        record = supabase.table('attendance_uploads').select('*').eq('id', record_id).execute()
        
        if not record.data:
            return jsonify({'success': False, 'error': 'Record not found'}), 404
        
        file_record = record.data[0]
        
        # Delete files from storage
        try:
            if file_record.get('csv_path'):
                supabase.storage.from_(STORAGE_BUCKET).remove([file_record['csv_path']])
                print(f"✅ Deleted CSV from storage: {file_record['csv_path']}")
        except Exception as e:
            print(f"⚠️ CSV deletion warning: {e}")
        
        try:
            if file_record.get('pdf_path'):
                supabase.storage.from_(STORAGE_BUCKET).remove([file_record['pdf_path']])
                print(f"✅ Deleted PDF from storage: {file_record['pdf_path']}")
        except Exception as e:
            print(f"⚠️ PDF deletion warning: {e}")
        
        # Delete database record
        supabase.table('attendance_uploads').delete().eq('id', record_id).execute()
        print(f"✅ Deleted record from database: {record_id}")
        
        return jsonify({'success': True, 'message': 'Upload deleted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)