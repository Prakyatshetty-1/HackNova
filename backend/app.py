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

# Import the report generation functions
from generate_attendance_report import compute_attendance, save_report, save_summary, plot_distribution, save_pdf_report

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

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
        
        # Clean and save CSV
        csv_content = clean_and_save_csv(response.text, csv_filepath)
        
        # Generate summary
        summary = {}
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
                final_pdf_path = os.path.join('uploads', pdf_filename)
                shutil.copy(pdf_path, final_pdf_path)
                print(f"✅ PDF copied to: {final_pdf_path}")
            else:
                print("❌ PDF generation returned None or file doesn't exist")
        except Exception as e:
            print(f"❌ Summary/PDF generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return jsonify({
            'success': True,
            'message': 'Attendance processed successfully',
            'filename': csv_filename,
            'pdf_filename': pdf_filename,
            'filepath': csv_filepath,
            'summary': summary,
            'preview': csv_content.split('\n')[:6]
        })
    
    except Exception as e:
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

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)