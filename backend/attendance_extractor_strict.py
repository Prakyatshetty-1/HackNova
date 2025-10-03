"""
attendance_extractor_strict.py

Usage:
    python attendance_extractor_strict.py <input_image_or_pdf> [debug_output_dir]

Outputs:
    - attendance_output.csv   (cell-level: page,row,col,status,ocr,red_ratio,stroke_ratio,blue_ratio,bbox_x,...)
    - attendance_matrix.csv   (student_id, roll, name, Lec_1, Lec_2, ...)
    - debug_cells/            (crops to inspect)
"""
import os, sys, tempfile, math
import cv2
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path

# ---------- CONFIG ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # change if needed
POPPLER_PATH = None   # set if reading PDFs on Windows, e.g. r"C:\poppler-xx\Library\bin"

# thresholds (tweak if needed)
RED_RATIO_THRESHOLD = 0.04
STROKE_RATIO_THRESHOLD = 0.003    # fraction of pixels that are strokes -> Present
BLUE_RATIO_THRESHOLD = 0.02
MIN_CELL_AREA = 500
HEADER_SCAN_ROWS = 6

# ---------- UTIL ----------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def save_img(path, img):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cv2.imwrite(path, img)

def clean_text(s):
    return (s or "").strip().replace("\n"," ").replace("\r"," ").strip()

# ---------- PREPROCESS: deskew + crop page content ----------
def deskew_and_crop(img):
    # convert to grayscale and get big contour (page)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = 255 - th
    contours, _ = cv2.findContours(th_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    cx,cy = rect[0]; wrect, hrect = rect[1]; angle = rect[2]
    # fix angle
    if wrect < hrect:
        angle_c = angle
    else:
        angle_c = angle + 90
    # rotate image to deskew
    (h,w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx,cy), angle_c, 1.0)
    rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # recompute largest bounding rect on rotated
    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, th2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2_inv = 255 - th2
    contours2, _ = cv2.findContours(th2_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        return rotated
    cnt2 = max(contours2, key=cv2.contourArea)
    x,y,wc,hc = cv2.boundingRect(cnt2)
    # pad a little
    pad = 6
    x0 = max(0, x-pad); y0 = max(0, y-pad)
    x1 = min(rotated.shape[1], x+wc+pad); y1 = min(rotated.shape[0], y+hc+pad)
    cropped = rotated[y0:y1, x0:x1]
    return cropped

# ---------- detect vertical and horizontal separators ----------
def detect_separators(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edge map
    edges = cv2.Canny(gray, 50, 150)
    h,w = edges.shape
    # Hough lines to detect long vertical & horizontal lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(80, w//20),
                            minLineLength=max(50, h//6), maxLineGap=20)
    vert_x = []
    hor_y = []
    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2 = l
            if abs(x1-x2) < 10 and abs(y2-y1) > h//8:  # vertical
                vert_x.append((x1+x2)//2)
            elif abs(y1-y2) < 10 and abs(x2-x1) > w//8:  # horizontal
                hor_y.append((y1+y2)//2)
    # fallback: use morphological projection
    if len(vert_x) < 2:
        # vertical projection from morphological vertical lines
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
        vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//25)))
        vert = cv2.morphologyEx(255-th, cv2.MORPH_OPEN, vkernel, iterations=1)
        proj = np.sum(vert, axis=0)
        peaks = _peaks_from_projection(proj, min_rel=0.2, min_dist=max(8,w//60))
        vert_x = peaks
    if len(hor_y) < 2:
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
        hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//25), 1))
        horiz = cv2.morphologyEx(255-th, cv2.MORPH_OPEN, hkernel, iterations=1)
        proj = np.sum(horiz, axis=1)
        peaks = _peaks_from_projection(proj, min_rel=0.2, min_dist=max(8,h//120), axis=0)
        hor_y = peaks
    vert_x = sorted(list(set(int(x) for x in vert_x)))
    hor_y = sorted(list(set(int(y) for y in hor_y)))
    return vert_x, hor_y

def _peaks_from_projection(proj, min_rel=0.2, min_dist=10, axis=1):
    # find indices where projection value above threshold fraction of max
    maxv = proj.max() if proj.size>0 else 0
    if maxv <= 0:
        return []
    thr = max(min_rel * maxv, 1)
    idxs = np.where(proj >= thr)[0]
    if len(idxs) == 0:
        return []
    groups = []
    cur = [int(idxs[0])]
    for i in idxs[1:]:
        if i - cur[-1] <= min_dist:
            cur.append(int(i))
        else:
            groups.append(int(round(np.mean(cur))))
            cur = [int(i)]
    groups.append(int(round(np.mean(cur))))
    return groups

# ---------- construct grid intervals from separators ----------
def separators_to_intervals(positions, max_dim):
    if not positions:
        return [(0, max_dim)]
    positions = sorted(positions)
    intervals = []
    prev = 0
    for p in positions:
        intervals.append((prev, max(0, p-1)))
        prev = min(max_dim, p+1)
    if prev < max_dim:
        intervals.append((prev, max_dim))
    intervals = [(a,b) for (a,b) in intervals if b-a > 8]
    return intervals

# ---------- per-cell stroke / color detectors ----------
def red_ratio(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,60,40]), np.array([10,255,255]))
    m2 = cv2.inRange(hsv, np.array([160,60,40]), np.array([179,255,255]))
    mask = cv2.bitwise_or(m1,m2)
    return mask.sum() / 255.0 / (crop.shape[0]*crop.shape[1])

def blue_ratio(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([90,40,40]), np.array([140,255,255]))
    return m.sum() / 255.0 / (crop.shape[0]*crop.shape[1])

def stroke_ratio(crop):
    # detect strokes by Canny then small dilation -> fraction of pixels that are strokes
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # equalize to improve stroke contrast
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return float(np.count_nonzero(edges)) / (crop.shape[0]*crop.shape[1])

# ---------- OCR helpers ----------
def ocr_crop(crop, psm=7, whitelist=None):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    if h < 30 or w < 30:
        gray = cv2.resize(gray, (max(32,w*2), max(32,h*2)), interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = f'--psm {psm}'
    if whitelist:
        config += f' -c tessedit_char_whitelist={whitelist}'
    try:
        txt = pytesseract.image_to_string(th, config=config)
        return clean_text(txt)
    except Exception:
        return ""

# ---------- main extraction pipeline ----------
def extract_attendance(img, debug_dir=None, manual_cols=None):
    # deskew / crop
    page = deskew_and_crop(img)
    H,W = page.shape[:2]

    # detect separators (vertical x positions and horizontal y positions)
    vert_x, hor_y = detect_separators(page)

    # convert separators to intervals
    col_intervals = separators_to_intervals(vert_x, W)
    row_intervals = separators_to_intervals(hor_y, H)

    # if intervals too coarse or empty, try fallback: split left area for metadata & rest by projection
    if len(col_intervals) < 3 or len(row_intervals) < 4:
        # try detect left metadata columns by vertical projection in left 40% of width
        left_w = int(W*0.40)
        left_crop = page[:, :left_w]
        gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_inv = 255 - th
        # find connected components / contours in left region to approximate roll/id/name columns
        cnts,_ = cv2.findContours(th_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xs = sorted([cv2.boundingRect(c)[0] for c in cnts if cv2.contourArea(c) > 200])
        # map as simple columns: [0..min(xs)/2, min(xs)..left_w] then rest of page to columns by equal width
        # conservative fallback grid
        col_intervals = [(0, left_w//3), (left_w//3, left_w*2//3), (left_w*2//3, left_w)] 
        # rest to 10 equal columns (for lectures)
        rest_start = left_w
        rest_cols = max(6, min(20, (W-rest_start)//60))
        step = max(30, (W-rest_start)//rest_cols) or 60
        rest_intervals = []
        x0 = rest_start
        while x0 < W-8:
            x1 = min(W, x0+step)
            rest_intervals.append((x0,x1)); x0 = x1
        col_intervals += rest_intervals
        # rows: use horizontal projection peaks
        gray_full = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        _, thf = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thf_inv = 255 - thf
        proj = np.sum(thf_inv, axis=1)
        peaks = _peaks_from_projection(proj, min_rel=0.02, min_dist=8, axis=0)
        # create row intervals from peaks
        row_intervals = separators_to_intervals(peaks, H)
        if len(row_intervals)==0:
            # final fallback: split into 30-pixel tall rows
            row_intervals = [(i, min(H,i+40)) for i in range(0,H,40)]

    # build grid cells and compute metrics
    cells = []
    all_cells = []
    for r_idx, (y0,y1) in enumerate(row_intervals):
        for c_idx, (x0,x1) in enumerate(col_intervals):
            # ensure valid
            if x1-x0 < 8 or y1-y0 < 8: continue
            crop = page[y0:y1, x0:x1]
            area = (x1-x0)*(y1-y0)
            if area < MIN_CELL_AREA: continue
            rr = red_ratio(crop)
            br = blue_ratio(crop)
            sr = stroke_ratio(crop)
            # small ocr for header detection only (avoid slow mass OCR)
            ocr_small = ocr_crop(crop, psm=7) if (r_idx < HEADER_SCAN_ROWS or c_idx < 3) else ""
            # heuristic status
            status = 'Unknown'
            if rr >= RED_RATIO_THRESHOLD:
                status = 'Absent'
            elif sr >= STROKE_RATIO_THRESHOLD or br >= BLUE_RATIO_THRESHOLD:
                status = 'Present'
            else:
                # try single-letter OCR fallback
                ch = ocr_crop(crop, psm=10, whitelist='PpAa')
                if ch.strip().lower() == 'p':
                    status = 'Present'
                elif ch.strip().lower() in ('a','ab'):
                    status = 'Absent'
            cell_dict = {'row': r_idx, 'col': c_idx, 'bbox':(x0,y0,x1-x0,y1-y0), 'crop':crop,
                         'red_ratio':round(rr,4),'blue_ratio':round(br,4),'stroke_ratio':round(sr,4),
                         'ocr':ocr_small,'status':status}
            cells.append(cell_dict)
            all_cells.append({
                'page':0,'row':r_idx,'col':c_idx,'status':status,'ocr':ocr_small,
                'red_ratio':round(rr,4),'blue_ratio':round(br,4),'stroke_ratio':round(sr,4),
                'bbox_x':int(x0),'bbox_y':int(y0),'bbox_w':int(x1-x0),'bbox_h':int(y1-y0)
            })
            # debug save
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                save_img(os.path.join(debug_dir, f'cell_r{r_idx}_c{c_idx}_{status}.png'), crop)

    # assemble rows -> list of lists
    rows = {}
    for c in cells:
        rows.setdefault(c['row'],[]).append(c)
    rows_list = [ sorted(rows[r], key=lambda cc: cc['col']) for r in sorted(rows.keys()) ]

    # detect header row by keyword scanning on first few rows
    header_row_idx = 0
    best_score = -1
    keywords = ['roll','roll no','student id','name','date','subject','lecture','lpno']
    for r in range(min(HEADER_SCAN_ROWS, len(rows_list))):
        score = 0
        for cell in rows_list[r]:
            txt = (cell.get('ocr') or '').lower()
            if not txt:
                txt = ocr_crop(cell['crop'], psm=6).lower()
            for kw in keywords:
                if kw in txt:
                    score += 1
        if score > best_score:
            best_score = score; header_row_idx = r

    # determine columns: roll, student_id, name (guess near left side)
    left_row = rows_list[header_row_idx] if header_row_idx < len(rows_list) else rows_list[0]
    roll_col = None; sid_col = None; name_col = None
    for i, c in enumerate(left_row):
        t = (c.get('ocr') or '').lower()
        if not t:
            t = ocr_crop(c['crop'], psm=6).lower()
        if roll_col is None and any(x in t for x in ['roll','roll no','rollno','l/pno']):
            roll_col = c['col']
        if sid_col is None and any(x in t for x in ['student id','studentid','id','stid']):
            sid_col = c['col']
        if name_col is None and any(x in t for x in ['name','student']):
            name_col = c['col']
    # fallbacks
    all_cols = sorted({c['col'] for c in cells})
    if roll_col is None: roll_col = min(all_cols)
    if sid_col is None:
        # try roll+1 in left row
        sid_col = roll_col + 1 if (roll_col+1) in all_cols else None
    if name_col is None:
        name_col = sid_col + 1 if sid_col and (sid_col+1) in all_cols else (roll_col+1 if (roll_col+1) in all_cols else roll_col)

    # lecture columns: all remaining in columns order (to the right)
    lecture_cols = [c for c in sorted(all_cols) if c not in (roll_col, sid_col, name_col)]
    # if nothing, assume columns after name_col
    if not lecture_cols:
        start = (name_col or roll_col) + 1
        lecture_cols = [c for c in sorted(all_cols) if c >= start]

    result = {
        'page_img': page,
        'rows': rows_list,
        'all_cells': all_cells,
        'header_row': header_row_idx,
        'roll_col': roll_col,
        'studentid_col': sid_col,
        'name_col': name_col,
        'lecture_cols': lecture_cols
    }
    return result

# ---------- aggregation to matrix ----------
def aggregate_to_matrix(result, debug_dir=None):
    rows = result['rows']
    header = result['header_row']
    roll_col = result['roll_col']
    sid_col = result['studentid_col']
    name_col = result['name_col']
    lecture_cols = result['lecture_cols']
    img = result['page_img']

    students = []
    for r_idx in range(header+1, len(rows)):
        rowcells = {c['col']: c for c in rows[r_idx]}
        def read(col):
            c = rowcells.get(col)
            if c is None: return ""
            txt = c.get('ocr') or ""
            if txt.strip(): return txt.strip()
            return ocr_crop(c['crop'], psm=7)
        roll_txt = read(roll_col)
        sid_txt = read(sid_col) if sid_col is not None else ""
        name_txt = read(name_col)
        lec_status = {}
        for lc in lecture_cols:
            c = rowcells.get(lc)
            label = f"Lec_{lc}"
            if c is None:
                lec_status[label] = 'Unknown'; continue
            # prefer status in cell metadata
            status = c.get('status','')
            if status in ('Present','Absent'):
                lec_status[label] = status
                continue
            # else use metrics
            rr = c.get('red_ratio',0.0)
            sr = c.get('stroke_ratio',0.0)
            br = c.get('blue_ratio',0.0)
            if rr >= RED_RATIO_THRESHOLD:
                lec_status[label] = 'Absent'
            elif sr >= STROKE_RATIO_THRESHOLD or br >= BLUE_RATIO_THRESHOLD:
                lec_status[label] = 'Present'
            else:
                # fallback OCR for AB/P
                txt = ocr_crop(c['crop'], psm=7).lower()
                if 'ab' in txt or txt.strip() == 'a': lec_status[label] = 'Absent'
                elif 'p' == txt.strip() or 'pres' in txt: lec_status[label] = 'Present'
                else: lec_status[label] = 'Unknown'
        students.append({'student_id': sid_txt, 'roll': roll_txt, 'name': name_txt, **lec_status})
    df = pd.DataFrame(students)
    return df

# ---------- wrapper CLI ----------
def main(input_path, debug_dir='debug_cells'):
    if not os.path.exists(input_path):
        print("Input not found:", input_path); sys.exit(1)
    # load image (pdf support)
    name, ext = os.path.splitext(input_path)
    if ext.lower() == '.pdf':
        pages = convert_from_path(input_path, dpi=300, poppler_path=POPPLER_PATH) if POPPLER_PATH else convert_from_path(input_path, dpi=300)
        img = pil_to_cv(pages[0])
    else:
        pil = Image.open(input_path).convert('RGB')
        img = pil_to_cv(pil)

    print("Processing (deskew, detect grid, extract cells)...")
    result = extract_attendance(img, debug_dir=debug_dir)
    # save cell-level CSV
    cells_df = pd.DataFrame(result['all_cells'])
    cells_csv = 'attendance_output.csv'
    cells_df.to_csv(cells_csv, index=False, encoding='utf-8-sig')
    print(f"Saved cell-level CSV to {cells_csv} ({len(cells_df)} cells).")

    # aggregate matrix
    matrix_df = aggregate_to_matrix(result, debug_dir=debug_dir)
    out_csv = 'attendance_matrix.csv'
    matrix_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print("Saved attendance matrix to", out_csv)

    print("Detected header row:", result['header_row'], "roll_col:", result['roll_col'], "sid_col:", result['studentid_col'], "name_col:", result['name_col'])
    print("Detected lecture cols (indices):", result['lecture_cols'])
    return result, cells_df, matrix_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python attendance_extractor_strict.py <input_image_or_pdf> [debug_output_dir]")
        sys.exit(1)
    inp = sys.argv[1]
    dbg = sys.argv[2] if len(sys.argv) > 2 else 'debug_cells'
    res, cells_df, matrix_df = main(inp, debug_dir=dbg)
    print("Done.")
