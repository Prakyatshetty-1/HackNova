# attendance_extractor_fix.py
"""
Fix-focused attendance extractor:
- stronger preprocessing
- prints header OCR table and saves header crops
- supports forcing roll/student_id/name columns via CLI flags
Usage:
  python attendance_extractor_fix.py <input_image_or_pdf> <debug_dir> [--force-roll N] [--force-sid N] [--force-name N] [--dpi 300]
"""
import os, sys, argparse, re, tempfile
import cv2, numpy as np, pytesseract, pandas as pd
from PIL import Image
from pdf2image import convert_from_path

# ---- CONFIG ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # change if needed
POPPLER_PATH = None  # set to poppler bin if using PDF on Windows

RED_RATIO_THRESHOLD = 0.04
STROKE_RATIO_THRESHOLD = 0.003
MIN_CELL_AREA = 350
HEADER_SCAN_ROWS = 6
NUM_STUDENT_DEBUG = 8  # save first N student-row crops for debugging

# ---- helpers ----
def pil_to_cv(p):
    return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

def clean_text(s):
    if s is None: return ""
    return re.sub(r'\s+', ' ', str(s)).strip()

def ensure_dir(d): 
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

# ---- basic preprocessing: deskew + crop page area ----
def deskew_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = 255 - th
    cnts, _ = cv2.findContours(th_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (cx,cy),(wrect,hrect),angle = rect
    # choose rotation angle
    angle_corr = angle + 90 if wrect > hrect else angle
    M = cv2.getRotationMatrix2D((cx,cy), angle_corr, 1.0)
    h,w = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, th2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2_inv = 255 - th2
    cnts2, _ = cv2.findContours(th2_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts2:
        return rotated
    cnt2 = max(cnts2, key=cv2.contourArea)
    x,y,wc,hc = cv2.boundingRect(cnt2)
    pad=6
    x0,y0 = max(0,x-pad), max(0,y-pad)
    x1,y1 = min(rotated.shape[1], x+wc+pad), min(rotated.shape[0], y+hc+pad)
    crop = rotated[y0:y1, x0:x1]
    return crop

# ---- detect vertical/horizontal separators (projection + Hough fallback) ----
def detect_separators(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h,w = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(80, w//20),
                            minLineLength=max(50, h//6), maxLineGap=20)
    vert = []
    hor = []
    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2 = l
            if abs(x1-x2) < 10 and abs(y2-y1) > h//8:
                vert.append((x1+x2)//2)
            elif abs(y1-y2) < 10 and abs(x2-x1) > w//8:
                hor.append((y1+y2)//2)
    # fallback projection
    if len(vert) < 3:
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//25)))
        vert_lines = cv2.morphologyEx(255-th, cv2.MORPH_OPEN, vert_kernel, iterations=1)
        proj = np.sum(vert_lines, axis=0)
        vert = peaks_from_projection(proj, min_rel=0.18, min_dist=max(8,w//60))
    if len(hor) < 3:
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//25), 1))
        horiz_lines = cv2.morphologyEx(255-th, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        proj = np.sum(horiz_lines, axis=1)
        hor = peaks_from_projection(proj, min_rel=0.18, min_dist=max(8,h//120), axis=0)
    vert = sorted(list(set(int(x) for x in vert)))
    hor = sorted(list(set(int(y) for y in hor)))
    return vert, hor

def peaks_from_projection(proj, min_rel=0.15, min_dist=8, axis=1):
    if proj.max() <= 0:
        return []
    thr = max(min_rel * proj.max(), 1)
    idxs = np.where(proj >= thr)[0]
    if len(idxs) == 0: return []
    groups=[]
    cur=[int(idxs[0])]
    for i in idxs[1:]:
        if i-cur[-1] <= min_dist:
            cur.append(int(i))
        else:
            groups.append(int(round(np.mean(cur)))); cur=[int(i)]
    groups.append(int(round(np.mean(cur))))
    return groups

def separators_to_intervals(seps, max_dim):
    if not seps: return [(0, max_dim)]
    seps = sorted(seps)
    intervals=[]
    prev=0
    for s in seps:
        intervals.append((prev, max(0,s-1))); prev = min(max_dim, s+1)
    if prev < max_dim: intervals.append((prev, max_dim))
    intervals=[(a,b) for (a,b) in intervals if (b-a)>8]
    return intervals

# ---- small utility OCR preps ----
def ocr_for_roll(crop):
    # only digits & letters typical in roll/id; aggressively clean before OCR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (max(100, gray.shape[1]*2), max(30, gray.shape[0]*2)), interpolation=cv2.INTER_LINEAR)
    _,th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"
    txt = pytesseract.image_to_string(th, config=cfg)
    txt = re.sub(r'[^0-9A-Za-z\-]', '', txt)
    return txt.strip()

def ocr_general(crop, psm=7):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 30 or gray.shape[1] < 30:
        gray = cv2.resize(gray, (max(64, gray.shape[1]*2), max(64, gray.shape[0]*2)), interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = f"--psm {psm}"
    try:
        return clean_text(pytesseract.image_to_string(th, config=cfg))
    except Exception:
        return ""

def red_ratio(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,60,40]), np.array([10,255,255]))
    m2 = cv2.inRange(hsv, np.array([160,60,40]), np.array([179,255,255]))
    mask = cv2.bitwise_or(m1,m2)
    return mask.sum()/255.0/(crop.shape[0]*crop.shape[1])

def stroke_ratio(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 30, 110)
    return float(np.count_nonzero(edges)) / (crop.shape[0]*crop.shape[1])

# ---- main extraction ----
def extract_grid_and_cells(img, debug_dir):
    page = deskew_and_crop(img)
    ensure_dir(debug_dir)
    save_preview = os.path.join(debug_dir, "page_preview.png")
    cv2.imwrite(save_preview, page)
    vert, hor = detect_separators(page)
    W,H = page.shape[1], page.shape[0]
    cols = separators_to_intervals(vert, W)
    rows = separators_to_intervals(hor, H)
    # fallback if too coarse: try CC-based as in earlier scripts (simple)
    if len(cols) < 3 or len(rows) < 4:
        # fallback split: left 40% into 3 meta cols, rest into many equal lecture cols
        left_w = int(W*0.40)
        cols = [(0, left_w//3), (left_w//3, left_w*2//3), (left_w*2//3, left_w)]
        rest_start = left_w
        rest_unit = max(60, (W-rest_start)//8)
        ccur = rest_start
        while ccur < W-8:
            cols.append((ccur, min(W, ccur+rest_unit)))
            ccur += rest_unit
        # rows by projection peaks
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY); _,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th_inv = 255-th
        proj = np.sum(th_inv, axis=1)
        peaks = peaks_from_projection(proj, min_rel=0.02, min_dist=8, axis=0)
        rows = separators_to_intervals(peaks, H)
        if not rows:
            rows = [(i, min(H,i+40)) for i in range(0,H,40)]
    # build cells
    cells = []
    all_cells = []
    for r_idx, (y0,y1) in enumerate(rows):
        for c_idx, (x0,x1) in enumerate(cols):
            if x1-x0 < 8 or y1-y0 < 8: continue
            crop = page[y0:y1, x0:x1]
            area = (x1-x0)*(y1-y0)
            if area < MIN_CELL_AREA: continue
            rr = red_ratio(crop)
            sr = stroke_ratio(crop)
            # small OCR for header zone OR leftmost 3 columns
            ocr_small = ""
            if r_idx < HEADER_SCAN_ROWS or c_idx < 3:
                ocr_small = ocr_general(crop, psm=7)
            # quick status
            status = 'Unknown'
            if rr >= RED_RATIO_THRESHOLD:
                status = 'Absent'
            elif sr >= STROKE_RATIO_THRESHOLD:
                status = 'Present'
            else:
                ch = ocr_general(crop, psm=10).strip().lower()
                if ch == 'p':
                    status = 'Present'
                if 'ab' in ch or ch == 'a':
                    status = 'Absent'
            cell = {'row': r_idx, 'col': c_idx, 'bbox':(int(x0),int(y0),int(x1-x0),int(y1-y0)),
                    'crop':crop, 'ocr':ocr_small, 'status':status,
                    'red_ratio':round(rr,4), 'stroke_ratio':round(sr,4)}
            cells.append(cell)
            all_cells.append({'page':0,'row':r_idx,'col':c_idx,'status':status,'ocr':ocr_small,
                              'red_ratio':round(rr,4),'stroke_ratio':round(sr,4),
                              'bbox_x':int(x0),'bbox_y':int(y0),'bbox_w':int(x1-x0),'bbox_h':int(y1-y0)})
            # debug save some header crops
            if r_idx < HEADER_SCAN_ROWS:
                cv2.imwrite(os.path.join(debug_dir, f'header_r{r_idx}_c{c_idx}.png'), crop)
    # assemble rows -> list-of-lists
    rows_map = {}
    for c in cells:
        rows_map.setdefault(c['row'], []).append(c)
    rows_list = [sorted(rows_map[r], key=lambda cc: cc['col']) for r in sorted(rows_map.keys())]
    return page, rows_list, all_cells

# ---- aggregation and final CSVs ----
def build_and_save(rows, all_cells, page_img, debug_dir, force_roll, force_sid, force_name, input_path):
    ensure_dir(debug_dir)
    # dump per-cell CSV
    cells_df = pd.DataFrame(all_cells)
    cells_csv = 'attendance_output.csv'
    cells_df.to_csv(cells_csv, index=False, encoding='utf-8-sig')
    print(f"Saved cell-level CSV to {cells_csv} ({len(cells_df)} cells). Debug crops in {debug_dir}/")

    # detect header
    header_idx = 0
    best = -1
    kw = ['roll','roll no','rollno','student id','name','student','date','subject']
    for r in range(min(HEADER_SCAN_ROWS, len(rows))):
        score = 0
        for c in rows[r]:
            txt = (c.get('ocr') or "").lower()
            if not txt:
                txt = ocr_general(c['crop'], psm=6).lower()
            for k in kw:
                if k in txt: score += 1
        if score > best:
            best = score; header_idx = r
    print("Guessed header row:", header_idx)

    # print header OCR results to console so user can inspect columns
    print("\nHeader OCR (row {}, show col_index: OCR_text):".format(header_idx))
    if header_idx < len(rows):
        for c in rows[header_idx]:
            print(f"  col {c['col']}: '{(c.get('ocr') or '')[:80]}'")
    print("Saved header crops to debug folder: header_r{n}_c{col}.png")

    # decide columns (either forced or auto)
    all_cols = sorted({c['col'] for r in rows for c in r})
    if force_roll is not None:
        roll_col = force_roll
    else:
        # try to auto-detect near left side
        roll_col = min(all_cols)
    if force_sid is not None:
        sid_col = force_sid
    else:
        sid_col = roll_col + 1 if (roll_col + 1) in all_cols else None
    if force_name is not None:
        name_col = force_name
    else:
        name_col = sid_col + 1 if sid_col and (sid_col+1) in all_cols else (roll_col+1 if (roll_col+1) in all_cols else roll_col)

    print("Using columns -> roll:", roll_col, "sid:", sid_col, "name:", name_col)

    lecture_cols = [c for c in all_cols if c not in (roll_col, sid_col, name_col)]
    if not lecture_cols:
        lecture_cols = [c for c in all_cols if c > (name_col or roll_col)]
    lecture_cols = sorted(lecture_cols)
    print("Lecture cols detected:", lecture_cols)

    # build matrix rows from rows after header
    students = []
    for r_idx in range(header_idx+1, len(rows)):
        rowcells = {c['col']: c for c in rows[r_idx]}
        def read(col):
            c = rowcells.get(col)
            if c is None: return ""
            # prefer strict OCR for roll and sid
            if col == roll_col or col == sid_col:
                val = ocr_for_roll(c['crop'])
                if val: return val
            if c.get('ocr'): return c.get('ocr')
            return ocr_general(c['crop'], psm=7)
        roll_txt = read(roll_col)
        sid_txt = read(sid_col) if sid_col is not None else ""
        name_txt = read(name_col)
        # save debug first few student rows with crops and OCR
        if (r_idx - (header_idx+1)) < NUM_STUDENT_DEBUG:
            ensure_dir(debug_dir)
            for col, c in rowcells.items():
                fn = os.path.join(debug_dir, f'studentrow{r_idx}_col{col}.png')
                cv2.imwrite(fn, c['crop'])
        lec_status = {}
        for lc in lecture_cols:
            c = rowcells.get(lc)
            label = f"Lec_{lc}"
            if c is None:
                lec_status[label] = 'Unknown'; continue
            # use metrics
            rr = c.get('red_ratio', 0.0)
            sr = c.get('stroke_ratio', 0.0)
            if rr >= RED_RATIO_THRESHOLD:
                lec_status[label] = 'Absent'
            elif sr >= STROKE_RATIO_THRESHOLD:
                lec_status[label] = 'Present'
            else:
                txt = ocr_general(c['crop'], psm=7).lower()
                if 'ab' in txt or txt.strip() == 'a': lec_status[label] = 'Absent'
                elif txt.strip() == 'p' or 'pres' in txt: lec_status[label] = 'Present'
                else: lec_status[label] = 'Unknown'
        students.append({'student_id': sid_txt, 'roll': roll_txt, 'name': name_txt, **lec_status})

    matrix_df = pd.DataFrame(students)
    matrix_df.to_csv('attendance_matrix.csv', index=False, encoding='utf-8-sig')
    print("Saved attendance_matrix.csv (rows: {})".format(len(matrix_df)))
    print("Look in debug folder to verify header crops and first student rows for tuning.")
    return cells_df, matrix_df

# ---- CLI entrypoint ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image or pdf')
    parser.add_argument('debug_dir', help='folder to save debug crops')
    parser.add_argument('--force-roll', type=int, help='force roll column index (0-based)', default=None)
    parser.add_argument('--force-sid', type=int, help='force student-id column index (0-based)', default=None)
    parser.add_argument('--force-name', type=int, help='force name column index (0-based)', default=None)
    parser.add_argument('--dpi', type=int, default=300, help='dpi for PDF->image (if pdf input)')
    args = parser.parse_args()

    inp = args.input
    dbg = args.debug_dir
    ensure_dir(dbg)

    if not os.path.exists(inp):
        print("Input not found:", inp); sys.exit(1)

    name, ext = os.path.splitext(inp)
    if ext.lower() == '.pdf':
        pages = convert_from_path(inp, dpi=args.dpi, poppler_path=POPPLER_PATH) if POPPLER_PATH else convert_from_path(inp, dpi=args.dpi)
        img = pil_to_cv(pages[0])
    else:
        pil = Image.open(inp).convert('RGB')
        img = pil_to_cv(pil)

    print("Preprocessing (deskew, crop). Saving page preview in debug folder.")
    page, rows, all_cells = None, None, None
    try:
        page, rows, all_cells = extract_grid_and_cells(img, dbg)
    except Exception as e:
        print("Extraction error:", e)
        sys.exit(1)

    cells_df, matrix_df = build_and_save(rows, all_cells, page, dbg, args.force_roll, args.force_sid, args.force_name, inp)
    # print small preview
    print("\nMatrix preview (first 8 rows):")
    print(matrix_df.head(8).to_string(index=False))
    print("\nDone. If output is still wrong, inspect debug folder and run again using --force-roll/--force-sid/--force-name with the column index you want to use.")

if __name__ == "__main__":
    main()
