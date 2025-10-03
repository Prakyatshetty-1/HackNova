"""
attendance_extractor_clean.py

Usage:
    python attendance_extractor_clean.py <input_image_or_pdf> [debug_output_dir]

Outputs:
    attendance_output.csv   (cell-level; includes bbox_x,bbox_y,bbox_w,bbox_h)
    attendance_matrix.csv   (student_id, roll, name, Lec_1, Lec_2, ...)
"""
import os
import sys
import cv2
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
import tempfile

# ----------------- CONFIG -----------------
# Set this to your tesseract exe if not on PATH (Windows default)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = None  # e.g. r"C:\poppler-23.04.0\Library\bin" if using PDF on Windows

RED_RATIO_THRESHOLD = 0.05   # fraction of red pixels to mark Absent
DARK_RATIO_THRESHOLD = 0.02  # fraction of dark/ink pixels to mark Present
MIN_CELL_AREA = 350          # ignore very small contours as cells
HEADER_CHECK_ROWS = 6        # number of top rows to search for header keywords
DEBUG_SAVE_CELLS = True      # set to False to disable saving cell crops

# ------------ helpers --------------
def pdf_to_images(pdf_path, dpi=200):
    kwargs = {"dpi": dpi}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH
    return convert_from_path(pdf_path, **kwargs)

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def red_mask_bgr(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 40]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 70, 40]); upper2 = np.array([179, 255, 255])
    m1 = cv2.inRange(hsv, lower1, upper1); m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)

def dark_mask_bgr(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]; val = hsv[:, :, 2]
    nonwhite = ((sat > 30) | (val < 200)).astype('uint8') * 255
    red = red_mask_bgr(img_bgr)
    dark = cv2.bitwise_and(nonwhite, cv2.bitwise_not(red))
    kernel = np.ones((2,2), np.uint8)
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

def ocr_cell_text(img_bgr, psm=7, whitelist=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    if h < 30 or w < 30:
        gray = cv2.resize(gray, (max(32,w*2), max(32,h*2)), interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = f'--psm {psm}'
    if whitelist:
        config += f' -c tessedit_char_whitelist={whitelist}'
    try:
        return pytesseract.image_to_string(th, config=config).strip()
    except Exception:
        return ""

# ---------- Table extraction ----------
def detect_table_cells_grid(img_bgr):
    """Attempt robust grid detection via morphological horizontal+vertical line detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to handle lighting
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)
    th_inv = 255 - th
    h,w = th_inv.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w//25), 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h//40)))
    horiz = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert  = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    grid = cv2.add(horiz, vert)
    grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,wc,hc = cv2.boundingRect(c)
        if wc*hc >= MIN_CELL_AREA:
            boxes.append((x,y,wc,hc))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def detect_table_cells_cc(img_bgr):
    """Fallback: connected components on inverted binary to find text/cell boxes."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = 255 - th
    # Dilate to join characters into cell blobs
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (15,7))
    dil = cv2.dilate(th_inv, kern, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,wc,hc = cv2.boundingRect(c)
        if wc*hc >= MIN_CELL_AREA:
            boxes.append((x,y,wc,hc))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def detect_table_cells(img_bgr):
    # try grid first
    boxes = detect_table_cells_grid(img_bgr)
    if len(boxes) < 5:
        # fallback to connected components
        boxes = detect_table_cells_cc(img_bgr)
    # if still too few, do a relaxed connected components pass
    if len(boxes) < 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray,0,255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(255-th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x,y,wc,hc = cv2.boundingRect(c)
            if wc*hc >= MIN_CELL_AREA//2:
                boxes.append((x,y,wc,hc))
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def group_boxes_into_rows(boxes, y_tol=14):
    # boxes is list of (x,y,w,h)
    rows = []
    for (x,y,w,h) in boxes:
        placed = False
        for row in rows:
            if abs(row['y'] - y) <= y_tol:
                row['boxes'].append((x,y,w,h))
                placed = True
                break
        if not placed:
            rows.append({'y': y, 'boxes': [(x,y,w,h)]})
    # sort boxes left-to-right inside rows
    for r in rows:
        r['boxes'] = sorted(r['boxes'], key=lambda B: B[0])
    rows = sorted(rows, key=lambda R: R['y'])
    return rows

# ---------- High-level processing ----------
def process_image(img_bgr, debug_dir=None, page_index=0):
    boxes = detect_table_cells(img_bgr)
    rows = group_boxes_into_rows(boxes, y_tol=max(10, img_bgr.shape[0]//300))

    table = []
    all_cells = []  # flat list for CSV
    # Build table structure (rows -> cell dicts)
    for ridx, r in enumerate(rows):
        cells = []
        for cidx, (x,y,w,h) in enumerate(r['boxes']):
            crop = img_bgr[y:y+h, x:x+w]
            total_pixels = max(1, crop.shape[0]*crop.shape[1])
            rmask = red_mask_bgr(crop)
            red_count = int(np.count_nonzero(rmask)); red_ratio = red_count/total_pixels
            dmask = dark_mask_bgr(crop)
            dark_count = int(np.count_nonzero(dmask)); dark_ratio = dark_count/total_pixels

            # small OCR for header-like cells (fast) - keep small to avoid slowness
            ocr_small = ocr_cell_text(crop, psm=7)

            # quick status guess
            status = 'Unknown'
            if red_ratio >= RED_RATIO_THRESHOLD:
                status = 'Absent'
            elif dark_ratio >= DARK_RATIO_THRESHOLD:
                status = 'Present'
            else:
                # try single char fallback
                ocr_char = ocr_cell_text(crop, psm=10, whitelist='PpAa')
                if ocr_char.strip().lower() == 'p':
                    status = 'Present'
                elif 'a' == ocr_char.strip().lower() or 'ab' in ocr_char.lower():
                    status = 'Absent'

            cell = {'row': ridx, 'col': cidx, 'bbox': (int(x),int(y),int(w),int(h)), 'crop': crop,
                    'red_ratio': round(red_ratio,4), 'dark_ratio': round(dark_ratio,4),
                    'ocr': ocr_small, 'status': status}
            cells.append(cell)
            all_cells.append({
                'page': page_index,
                'row': ridx,
                'col': cidx,
                'status': status,
                'ocr': ocr_small,
                'red_ratio': round(red_ratio,4),
                'dark_ratio': round(dark_ratio,4),
                'bbox_x': int(x), 'bbox_y': int(y), 'bbox_w': int(w), 'bbox_h': int(h)
            })
        table.append(cells)

    # If nothing detected, return empty gracefully
    if len(table) == 0:
        return {'header_row':0,'roll_col':0,'studentid_col':None,'name_col':1,'lecture_cols':[],'students':[]}, all_cells

    # ----- Identify header row -----
    header_idx = 0
    header_keywords = ['roll', 'roll no', 'rollno', 'student id', 'student', 'name', 'date', 'subject', 'lecture']
    best_score = -1
    # try stronger OCR for top rows (use multiple PSMs)
    for r in range(min(HEADER_CHECK_ROWS, len(table))):
        score = 0
        for cell in table[r]:
            txt = (cell.get('ocr') or '').lower()
            if not txt:
                # try stronger OCR for header detection
                try_txt = ocr_cell_text(cell['crop'], psm=6)
                if not try_txt:
                    try_txt = ocr_cell_text(cell['crop'], psm=3)
                txt = try_txt.lower()
            for kw in header_keywords:
                if kw in txt:
                    score += 1
        if score > best_score:
            best_score = score
            header_idx = r

    header_cells = table[header_idx] if header_idx < len(table) else []

    # ----- Detect column indices for roll, student_id, name -----
    roll_col = None
    studentid_col = None
    name_col = None
    for cidx, cell in enumerate(header_cells):
        txt = (cell.get('ocr') or '').lower()
        if not txt:
            txt = ocr_cell_text(cell['crop'], psm=6).lower()
        if any(k in txt for k in ['roll', 'roll no', 'rollno']) and roll_col is None:
            roll_col = cidx
        elif any(k in txt for k in ['student id', 'studentid', 'id', 'stid']) and studentid_col is None:
            studentid_col = cidx
        elif any(k in txt for k in ['name', 'student']) and name_col is None:
            name_col = cidx

    # Fallback guesses
    if roll_col is None:
        roll_col = 0
    if studentid_col is None:
        studentid_col = roll_col + 1 if (roll_col + 1) < len(header_cells) else None
    if name_col is None:
        guess = roll_col + 1 if studentid_col is None else (studentid_col + 1)
        if guess is not None and guess < len(header_cells):
            name_col = guess
        else:
            name_col = roll_col + 1 if (roll_col + 1) < len(header_cells) else roll_col

    # Compute max columns from table shape
    max_cols = max((len(r) for r in table), default=0)
    lecture_cols = [c for c in range(max_cols) if c not in (roll_col, studentid_col, name_col)]
    lecture_cols = sorted(lecture_cols)

    # Optional: save header crops for debugging
    if debug_dir and DEBUG_SAVE_CELLS:
        os.makedirs(debug_dir, exist_ok=True)
        for r_idx, row_cells in enumerate(table[:HEADER_CHECK_ROWS]):
            for cell in row_cells:
                x,y,w,h = cell['bbox']
                fn = os.path.join(debug_dir, f'header_r{r_idx}_c{cell["col"]}.png')
                cv2.imwrite(fn, cell['crop'])

    # ----- Build student rows ----- (rows after header_idx)
    students = []
    for ridx in range(header_idx+1, len(table)):
        row_cells = table[ridx]
        def get_cell(col):
            for c in row_cells:
                if c['col'] == col:
                    return c
            return None
        # OCR roll, student id, name (only these)
        roll_txt = ''
        sid_txt = ''
        name_txt = ''
        c_roll = get_cell(roll_col)
        if c_roll:
            roll_txt = ocr_cell_text(c_roll['crop'], psm=7)
            if not roll_txt:
                roll_txt = ocr_cell_text(c_roll['crop'], psm=6)
        c_sid = get_cell(studentid_col) if studentid_col is not None else None
        if c_sid:
            sid_txt = ocr_cell_text(c_sid['crop'], psm=7)
        c_name = get_cell(name_col)
        if c_name:
            name_txt = ocr_cell_text(c_name['crop'], psm=7)
            if not name_txt:
                name_txt = ocr_cell_text(c_name['crop'], psm=6)

        lecture_status = {}
        for lc in lecture_cols:
            c = get_cell(lc)
            label = f"Lec_{lc}"
            if c is None:
                lecture_status[label] = 'Unknown'
                continue
            crop = c['crop']; total_pixels = max(1, crop.shape[0]*crop.shape[1])
            rmask = red_mask_bgr(crop); red_count = int(np.count_nonzero(rmask)); red_ratio = red_count/total_pixels
            if red_ratio >= RED_RATIO_THRESHOLD:
                lecture_status[label] = 'Absent'
                if debug_dir and DEBUG_SAVE_CELLS:
                    cv2.imwrite(os.path.join(debug_dir, f'row{ridx}_col{lc}_ABSENT_RED.png'), crop)
                continue
            dmask = dark_mask_bgr(crop); dark_count = int(np.count_nonzero(dmask)); dark_ratio = dark_count/total_pixels
            char_p = ''
            if dark_ratio < DARK_RATIO_THRESHOLD:
                char_p = ocr_cell_text(crop, psm=10, whitelist='Pp')
            if dark_ratio >= DARK_RATIO_THRESHOLD or (char_p.strip().lower() == 'p'):
                lecture_status[label] = 'Present'
                if debug_dir and DEBUG_SAVE_CELLS:
                    cv2.imwrite(os.path.join(debug_dir, f'row{ridx}_col{lc}_PRESENT.png'), crop)
            else:
                txt_small = ocr_cell_text(crop, psm=7).lower()
                if 'ab' in txt_small or txt_small.strip() == 'a':
                    lecture_status[label] = 'Absent'
                elif txt_small.strip() == 'p' or 'pres' in txt_small:
                    lecture_status[label] = 'Present'
                else:
                    lecture_status[label] = 'Unknown'
        students.append({'row_index':ridx,'roll':roll_txt,'student_id':sid_txt,'name':name_txt,'lectures':lecture_status})

    result = {
        'header_row': header_idx,
        'roll_col': roll_col,
        'studentid_col': studentid_col,
        'name_col': name_col,
        'lecture_cols': lecture_cols,
        'students': students
    }
    return result, all_cells

# ---------- Aggregate & Save from in-memory result OR from cell-CSV ----------
def _ocr_crop_text(img, bbox, psm=7, whitelist=None):
    x,y,w,h = map(int, bbox)
    h_img, w_img = img.shape[:2]
    x = max(0, min(x, w_img-1)); y = max(0, min(y, h_img-1))
    w = max(1, min(w, w_img - x)); h = max(1, min(h, h_img - y))
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    H,W = gray.shape
    if H < 30 or W < 30:
        gray = cv2.resize(gray, (W*2, H*2), interpolation=cv2.INTER_LINEAR)
    _,th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = f'--psm {psm}'
    if whitelist:
        config += f' -c tessedit_char_whitelist={whitelist}'
    try:
        txt = pytesseract.image_to_string(th, config=config)
        return txt.strip()
    except Exception:
        return ""

def save_attendance_matrix(result=None,
                           cells_csv='attendance_output.csv',
                           image_path='attendance_sheet.png',
                           out_csv='attendance_matrix.csv'):
    """
    If `result` provided -> use it. Otherwise read cells_csv and image_path to aggregate.
    """
    # 1) If we have `result` in memory
    if result is not None and isinstance(result, dict) and result.get('students'):
        lec_cols = [f"Lec_{c}" for c in result['lecture_cols']]
        rows = []
        for s in result['students']:
            row = {'student_id': s.get('student_id',''), 'roll': s.get('roll',''), 'name': s.get('name','')}
            for lec in lec_cols:
                row[lec] = s['lectures'].get(lec, 'Unknown')
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"Saved attendance matrix to {out_csv} ({len(df)} rows, {len(lec_cols)} lectures).")
        return df

    # 2) read CSV and aggregate
    if not os.path.exists(cells_csv):
        raise FileNotFoundError(f"cells CSV not found: {cells_csv}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    cells = pd.read_csv(cells_csv, dtype={'page':int,'row':int,'col':int})
    # ensure bbox columns exist
    required_bbox = {'bbox_x','bbox_y','bbox_w','bbox_h'}
    if not required_bbox.issubset(set(cells.columns)):
        raise ValueError(f"CSV must contain bbox_x,bbox_y,bbox_w,bbox_h columns")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image: {image_path}")

    # group by (page,row)
    grouped = {}
    for _, r in cells.iterrows():
        key = (int(r['page']), int(r['row']))
        grouped.setdefault(key, []).append(r)
    grouped_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    for k in list(grouped.keys()):
        grouped[k] = sorted(grouped[k], key=lambda rr: int(rr['col']))

    # find header
    header_keywords = ['roll','roll no','student id','name','date','subject','lecture']
    top_keys = grouped_keys[:HEADER_CHECK_ROWS] if grouped_keys else []
    best_score = -1; header_key = None
    for key in top_keys:
        score = 0
        for cell in grouped[key]:
            txt = str(cell.get('ocr','') or '').lower()
            for kw in header_keywords:
                if kw in txt:
                    score += 1
        if score > best_score:
            best_score = score; header_key = key
    if header_key is None:
        header_key = top_keys[0] if top_keys else (grouped_keys[0] if grouped_keys else (0,0))
    print("Guessed header row:", header_key)

    header_cells = grouped.get(header_key, [])
    roll_col = None; sid_col = None; name_col = None
    for cell in header_cells:
        txt = str(cell.get('ocr','') or '').lower()
        cidx = int(cell['col'])
        if not txt:
            # try OCR directly on bbox (stronger)
            bbox = (cell['bbox_x'], cell['bbox_y'], cell['bbox_w'], cell['bbox_h'])
            txt = _ocr_crop_text(img, bbox, psm=6).lower()
        if any(k in txt for k in ['roll','roll no','rollno']) and roll_col is None:
            roll_col = cidx
        if any(k in txt for k in ['student id','studentid','id']) and sid_col is None:
            sid_col = cidx
        if any(k in txt for k in ['name','student']) and name_col is None:
            name_col = cidx

    all_cols = sorted({int(rc['col']) for rows in grouped.values() for rc in rows})
    max_cols = max(all_cols)+1 if all_cols else 0
    if roll_col is None:
        roll_col = 0
    if sid_col is None:
        sid_col = roll_col+1 if (roll_col+1) < max_cols else None
    if name_col is None:
        guess = roll_col+2 if (roll_col+2) < max_cols else (roll_col+1 if (roll_col+1) < max_cols else roll_col)
        name_col = guess

    print("Using columns -> roll:", roll_col, "student_id:", sid_col, "name:", name_col)

    lecture_cols = [c for c in all_cols if c not in (roll_col, sid_col, name_col)]
    lecture_cols = sorted(lecture_cols)
    if not lecture_cols:
        # fallback: assume columns after name_col are lectures
        lecture_cols = [c for c in range((name_col or roll_col)+1, max_cols)]

    print("Detected lecture columns:", lecture_cols)

    # build students
    students = []
    for key in grouped_keys:
        if key[1] <= header_key[1]:
            continue
        row_cells = grouped[key]
        bycol = {int(rc['col']): rc for rc in row_cells}
        def read_cell_text(col):
            rc = bycol.get(col)
            if rc is None:
                return ""
            saved = str(rc.get('ocr','') or '').strip()
            if saved:
                return saved
            bbox = (rc['bbox_x'], rc['bbox_y'], rc['bbox_w'], rc['bbox_h'])
            return _ocr_crop_text(img, bbox, psm=7)
        roll_txt = read_cell_text(roll_col)
        sid_txt = read_cell_text(sid_col) if sid_col is not None else ''
        name_txt = read_cell_text(name_col)

        lec_status = {}
        for lc in lecture_cols:
            rc = bycol.get(lc)
            label = f"Lec_{lc}"
            if rc is None:
                lec_status[label] = 'Unknown'; continue
            status = str(rc.get('status','') or '').strip().lower()
            if status in ('present','absent'):
                lec_status[label] = status.capitalize(); continue
            red = float(rc.get('red_ratio',0.0))
            dark = float(rc.get('dark_ratio',0.0))
            if red >= RED_RATIO_THRESHOLD:
                lec_status[label] = 'Absent'
            elif dark >= DARK_RATIO_THRESHOLD:
                lec_status[label] = 'Present'
            else:
                bbox = (rc['bbox_x'], rc['bbox_y'], rc['bbox_w'], rc['bbox_h'])
                txt = _ocr_crop_text(img, bbox, psm=7).lower()
                if 'ab' in txt or txt.strip() == 'a':
                    lec_status[label] = 'Absent'
                elif txt.strip() == 'p' or 'pres' in txt:
                    lec_status[label] = 'Present'
                else:
                    lec_status[label] = 'Unknown'
        students.append({'student_id': sid_txt, 'roll': roll_txt, 'name': name_txt, **lec_status})

    out_df = pd.DataFrame(students)
    out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print("Saved final attendance matrix to", out_csv)
    return out_df

# --------- CLI ----------
def main(input_path, debug_dir=None):
    name, ext = os.path.splitext(input_path)
    pages = []
    temp_image_path = None
    if ext.lower() == '.pdf':
        pages_pil = pdf_to_images(input_path, dpi=300)
        pages = [pil_to_cv(p) for p in pages_pil]
        # save first page to temp PNG for aggregation/OCR
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(tmp.name, pages[0])
        temp_image_path = tmp.name
    else:
        pil = Image.open(input_path).convert('RGB')
        pages = [pil_to_cv(pil)]
        # save a temp copy (ensures consistent path for aggregator)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(tmp.name, pages[0])
        temp_image_path = tmp.name

    img = pages[0]
    print("Processing image...")
    result, all_cells = process_image(img, debug_dir=debug_dir, page_index=0)

    # save cell-level CSV
    cells_df = pd.DataFrame(all_cells)
    cells_csv = 'attendance_output.csv'
    cells_df.to_csv(cells_csv, index=False, encoding='utf-8-sig')
    print(f"Saved cell-level CSV to {cells_csv} ({len(cells_df)} cells).")

    # generate final matrix using the saved CSV & temp page image
    try:
        matrix_df = save_attendance_matrix(result=None, cells_csv=cells_csv, image_path=temp_image_path, out_csv='attendance_matrix.csv')
        print("Saved matrix to attendance_matrix.csv")
    except Exception as e:
        print("Aggregation failed:", e)
        # fallback: try saving from in-memory result
        df_from_mem = save_attendance_matrix(result=result, out_csv='attendance_matrix.csv')
        if df_from_mem is not None:
            print("Saved matrix from in-memory result to attendance_matrix.csv")

    # cleanup temp image
    try:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    except Exception:
        pass

    return result, all_cells

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python attendance_extractor_clean.py <input_image_or_pdf> [debug_output_dir]")
        sys.exit(1)
    inp = sys.argv[1]
    dbg = sys.argv[2] if len(sys.argv) > 2 else 'debug_cells'
    if not os.path.exists(inp):
        print("Input not found:", inp)
        sys.exit(1)
    result, all_cells = main(inp, debug_dir=dbg)
    print("Done.")
