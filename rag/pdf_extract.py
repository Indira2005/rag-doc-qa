"""Layout-aware PDF extraction: tables become Markdown, multi-column text keeps reading order."""
import json
import re
from typing import Any, Dict, List


def _is_word_inside_bbox(word: Dict[str, Any], bbox: tuple) -> bool:
    """Return True when the word center falls inside a pdfplumber bbox."""
    x0, top, x1, bottom = bbox
    word_x = (float(word["x0"]) + float(word["x1"])) / 2
    word_y = (float(word["top"]) + float(word["bottom"])) / 2
    return x0 <= word_x <= x1 and top <= word_y <= bottom


def _words_to_text(words: List[Dict[str, Any]], line_tolerance: float = 3.0) -> str:
    """Rebuild readable text from positioned pdfplumber words, preserving multi-column reading order."""
    if not words:
        return ""

    # 1. Group words into horizontal lines based on vertical proximity
    sorted_words = sorted(words, key=lambda item: (float(item["top"]), item["x0"]))
    lines: List[List[Dict[str, Any]]] = []

    for word in sorted_words:
        if not lines:
            lines.append([word])
            continue

        current_top = sum(float(item["top"]) for item in lines[-1]) / len(lines[-1])
        if abs(float(word["top"]) - current_top) <= line_tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])

    # 2. Split each line into segments based on horizontal gaps (column gutters)
    GAP_THRESHOLD = 20.0
    line_segments_list = []
    for line in lines:
        sorted_line_words = sorted(line, key=lambda item: item["x0"])
        segments_in_line = []
        current_seg = [sorted_line_words[0]]

        for word in sorted_line_words[1:]:
            prev_word = current_seg[-1]
            gap = float(word["x0"]) - float(prev_word["x1"])
            if gap > GAP_THRESHOLD:
                segments_in_line.append(current_seg)
                current_seg = [word]
            else:
                current_seg.append(word)
        segments_in_line.append(current_seg)
        line_segments_list.append(segments_in_line)

    # 3. Detect global vertical gutters from lines with multiple layout segments
    gutter_intervals = []
    for seg_list in line_segments_list:
        if len(seg_list) > 1:
            for i in range(len(seg_list) - 1):
                x1_prev = max(float(w["x1"]) for w in seg_list[i])
                x0_next = min(float(w["x0"]) for w in seg_list[i+1])
                if x0_next > x1_prev:
                    gutter_intervals.append((x1_prev, x0_next))

    significant_gutter_centers = []
    if gutter_intervals:
        sorted_gutters = sorted(gutter_intervals, key=lambda g: g[0])
        current_gutter_groups = []
        for g in sorted_gutters:
            inserted = False
            for group in current_gutter_groups:
                group_x0 = max(item[0] for item in group)
                group_x1 = min(item[1] for item in group)
                if max(g[0], group_x0) < min(g[1], group_x1):
                    group.append(g)
                    inserted = True
                    break
            if not inserted:
                current_gutter_groups.append([g])

        num_multi_seg_lines = sum(1 for seg_list in line_segments_list if len(seg_list) > 1)
        min_group_size = max(2, int(num_multi_seg_lines * 0.15))

        for group in current_gutter_groups:
            if len(group) >= min_group_size:
                centers = [(g[0] + g[1]) / 2.0 for g in group]
                significant_gutter_centers.append(sum(centers) / len(centers))
        significant_gutter_centers.sort()

    # A real column boundary has plenty of text on both sides and little text running across it.
    # Author blocks and tables can suggest false boundaries that cut through both columns.
    # When a real one exists, drop the false ones. Other pages are left as before.
    def _is_column_boundary(center):
        left = right = across = 0
        for seg_list in line_segments_list:
            for seg in seg_list:
                if max(float(w["x1"]) for w in seg) <= center:
                    left += 1
                elif min(float(w["x0"]) for w in seg) >= center:
                    right += 1
                else:
                    across += 1
        return min(left, right) >= 10 and across < min(left, right)

    real_gutters = [c for c in significant_gutter_centers if _is_column_boundary(c)]
    if real_gutters:
        significant_gutter_centers = real_gutters

    # 4. Flatten segments and capture absolute structural bounds
    segment_objs = []
    for seg_list in line_segments_list:
        for seg in seg_list:
            seg_x0 = min(float(w["x0"]) for w in seg)
            seg_x1 = max(float(w["x1"]) for w in seg)
            seg_top = min(float(w["top"]) for w in seg)
            seg_bottom = max(float(w["bottom"]) for w in seg)
            seg_text = " ".join(w["text"] for w in seg)
            segment_objs.append({
                "x0": seg_x0,
                "x1": seg_x1,
                "top": seg_top,
                "bottom": seg_bottom,
                "text": seg_text
            })

    # 5. Sort segments vertically into horizontal layout bands divided by full-width text structures
    segment_objs.sort(key=lambda s: s["top"])

    final_segments = []
    current_band = []
    GUTTER_TOLERANCE = 3.0

    for seg in segment_objs:
        crosses_gutter = False
        for center in significant_gutter_centers:
            if (seg["x0"] + GUTTER_TOLERANCE) < center < (seg["x1"] - GUTTER_TOLERANCE):
                crosses_gutter = True
                break

        if crosses_gutter:
            if current_band:
                def get_col_idx(s):
                    idx = 0
                    s_mid = (s["x0"] + s["x1"]) / 2.0
                    for c in significant_gutter_centers:
                        if s_mid > c:
                            idx += 1
                    return idx
                current_band.sort(key=lambda s: (get_col_idx(s), s["top"]))
                final_segments.extend(current_band)
                current_band = []
            final_segments.append(seg)
        else:
            current_band.append(seg)

    if current_band:
        def get_col_idx(s):
            idx = 0
            s_mid = (s["x0"] + s["x1"]) / 2.0
            for c in significant_gutter_centers:
                if s_mid > c:
                    idx += 1
            return idx
        current_band.sort(key=lambda s: (get_col_idx(s), s["top"]))
        final_segments.extend(current_band)

    text_lines = [seg["text"] for seg in final_segments if seg["text"].strip()]
    return "\n".join(text_lines)


def _clean_table_cell(cell: Any) -> str:
    """Normalize extracted table cell text for Markdown serialization."""
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip().replace("|", "\\|")


def _table_to_markdown(rows: List[List[Any]]) -> str:
    """Serialize extracted table rows into Markdown for retrieval."""
    cleaned_rows = [
        [_clean_table_cell(cell) for cell in row]
        for row in rows
        if row and any(_clean_table_cell(cell) for cell in row)
    ]
    if not cleaned_rows:
        return ""

    width = max(len(row) for row in cleaned_rows)
    normalized = [row + [""] * (width - len(row)) for row in cleaned_rows]

    def fmt(row: List[str]) -> str:
        return "| " + " | ".join(row) + " |"

    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]
    return "\n".join([fmt(header), fmt(separator), *[fmt(row) for row in body]])


def extract_pdf_with_tables(filepath: str) -> List[Dict[str, Any]]:
    """Detect tables with pdfplumber, remove table text from paragraphs, and keep table bboxes."""
    import pdfplumber

    pages: List[Dict[str, Any]] = []

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables]

            words = page.extract_words(x_tolerance=1.5) or []
            paragraph_words = [
                word for word in words
                if not any(_is_word_inside_bbox(word, bbox) for bbox in table_bboxes)
            ]
            paragraph_text = _words_to_text(paragraph_words)

            if paragraph_text.strip():
                pages.append({
                    "text": paragraph_text,
                    "page": page_num,
                    "chunk_type": "text",
                })

            for table_index, table in enumerate(tables):
                table_text = _table_to_markdown(table.extract() or [])
                if table_text.strip():
                    W, H = float(page.width), float(page.height)
                    normalized_bbox = [
                        round(float(table.bbox[0]) / W, 4),
                        round(float(table.bbox[1]) / H, 4),
                        round(float(table.bbox[2]) / W, 4),
                        round(float(table.bbox[3]) / H, 4),
                    ]
                    pages.append({
                        "text": table_text,
                        "page": page_num,
                        "chunk_type": "table",
                        "bbox": json.dumps(normalized_bbox),
                        "table_index": table_index,
                    })

    return pages
