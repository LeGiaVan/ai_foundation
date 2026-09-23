"""
vision_zoom_corrector.py — Bộ Tự Sửa Sai Cục Bộ qua Vision-LLM Zoom (Agentic Self-Correction Loop).
Khi AccountingVerifier phát hiện sai lệch số học (is_balanced == False), module này:
  1. Phân tích Deductive Elimination để khoanh vùng các Concept / Fact bị nghi ngờ.
  2. Định vị tọa độ Bounding Box của dòng chứa Fact trên trang PDF.
  3. Cắt dải ảnh độ phân giải cao (Zoomed Sub-image) kèm padding.
  4. Gửi ảnh phóng to đến Vision API (Gemini/Groq) với prompt siêu tập trung để đọc lại số.
  5. Hot-patch Fact và kích hoạt re-verify tự động nhằm đưa BCTC về trạng thái cân đối.
"""

import base64
import io
import json
import logging
import re
from typing import Any

import pdfplumber
from PIL import Image

from src.config import Settings, get_settings
from src.models import FinancialFact, VerificationReport, VerificationStatus
from src.parser.ocr_postprocess import clean_ocr_number
from src.verifier.accounting_verifier import AccountingVerifier

logger = logging.getLogger(__name__)

ZOOM_SYSTEM_PROMPT = """Bạn là chuyên gia kiểm toán Báo cáo tài chính (BCTC) Việt Nam.
Nhiệm vụ của bạn là thẩm định cực kỳ cẩn thận ảnh chụp phóng to của DUY NHẤT 1 dòng số liệu kế toán:
- Tên khoản mục cần đọc: {raw_label}
- Mã số chuẩn Thông tư 200: {standard_code}

HÃY ĐẶC BIỆT CHÚ Ý CÁC BẪY SỐ LIỆU SAU:
1. Dấu ngoặc đơn chỉ số âm ví dụ (15.200.000) -> Giá trị thực là số âm: -15200000.
2. Dấu phân cách nghìn là dấu chấm hay dấu phẩy.
3. Chữ số bị lem/mờ (số 8 nhầm 0, số 3 nhầm 8, số 1 nhầm 7).
4. Cột 'Số cuối năm' / 'Kỳ này' thường nằm ngay sau cột Mã số / Thuyết minh.

Chỉ trả về một đối tượng JSON duy nhất (không giải thích thêm):
{{
  "raw_text": "chuỗi văn bản đọc được",
  "value_current": float hoặc null,
  "is_negative": bool,
  "confidence": float
}}"""


class VisionZoomCorrector:
    """Bộ tự động phát hiện, phóng to (Zoom) và sửa lỗi OCR cho các dòng số liệu tài chính."""

    def __init__(
        self,
        settings: Settings | None = None,
        verifier: AccountingVerifier | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.verifier = verifier or AccountingVerifier()

    def identify_suspect_facts(
        self,
        report: VerificationReport,
        facts: list[FinancialFact],
    ) -> list[FinancialFact]:
        """
        Dùng phương pháp Suy luận Loại trừ (Deductive Elimination) để khoanh vùng
        và sắp xếp thứ tự ưu tiên các Fact bị nghi ngờ gây ra sai lệch số học.
        """
        if report.is_balanced or not report.discrepancies:
            return []

        # Bản đồ liên kết giữa tên bài kiểm tra và các concepts cấu thành
        check_to_concepts: dict[str, list[str]] = {
            "TOTAL_ASSETS == TOTAL_RESOURCES": ["TOTAL_ASSETS", "TOTAL_RESOURCES"],
            "TOTAL_ASSETS == CURRENT + NON_CURRENT": ["TOTAL_ASSETS", "CURRENT_ASSETS", "NON_CURRENT_ASSETS"],
            "CURRENT_ASSETS == SUM_CHILDREN": [
                "CURRENT_ASSETS",
                "CASH_AND_EQUIVALENTS",
                "SHORT_TERM_INVESTMENTS",
                "SHORT_TERM_RECEIVABLES",
                "INVENTORIES",
                "OTHER_CURRENT_ASSETS",
            ],
            "TOTAL_RESOURCES == LIABILITIES + EQUITY": ["TOTAL_RESOURCES", "LIABILITIES", "EQUITY"],
            "GROSS_PROFIT == NET_REVENUE - COGS": ["GROSS_PROFIT", "NET_REVENUE", "COGS"],
        }

        # 1. Đếm số lần concept xuất hiện trong các bài kiểm tra thất bại
        suspect_scores: dict[str, int] = {}
        for disc in report.discrepancies:
            check_name = disc.get("check", "")
            for check_key, concepts in check_to_concepts.items():
                if check_key in check_name:
                    for c in concepts:
                        suspect_scores[c] = suspect_scores.get(c, 0) + 10

        # 2. Áp dụng Deductive Elimination: Nếu 1 phương trình nội bộ ĐẠT, giảm điểm nghi ngờ
        passed_text = " ".join(report.passed_checks)
        if "CỘNG_TỔNG_TÀI_SẢN" in passed_text:
            # Ngắn hạn + Dài hạn = Tổng tài sản cân đối -> nhóm Tài sản rất đáng tin cậy
            suspect_scores["TOTAL_ASSETS"] = suspect_scores.get("TOTAL_ASSETS", 0) - 15
            suspect_scores["CURRENT_ASSETS"] = suspect_scores.get("CURRENT_ASSETS", 0) - 10
            suspect_scores["NON_CURRENT_ASSETS"] = suspect_scores.get("NON_CURRENT_ASSETS", 0) - 10

        if "CỘNG_NGUỒN_VỐN" in passed_text:
            # Nợ + Vốn CSH = Tổng nguồn vốn cân đối -> nhóm Nguồn vốn rất đáng tin cậy
            suspect_scores["TOTAL_RESOURCES"] = suspect_scores.get("TOTAL_RESOURCES", 0) - 15
            suspect_scores["LIABILITIES"] = suspect_scores.get("LIABILITIES", 0) - 10
            suspect_scores["EQUITY"] = suspect_scores.get("EQUITY", 0) - 10

        # Lọc các concept có điểm nghi ngờ > 0 và sắp xếp giảm dần
        sorted_suspect_concepts = sorted(
            [c for c, score in suspect_scores.items() if score > 0],
            key=lambda c: suspect_scores[c],
            reverse=True,
        )

        # Lấy Fact tương ứng kỳ hiện tại
        current_facts = [f for f in facts if f.period_type == "current"]
        suspect_facts: list[FinancialFact] = []
        for concept in sorted_suspect_concepts:
            matching = [f for f in current_facts if f.concept == concept]
            if matching:
                suspect_facts.append(matching[0])

        return suspect_facts

    def locate_and_crop_row(
        self,
        pdf_path: str,
        fact: FinancialFact,
        resolution: int = 180,
        save_debug_dir: str | None = None,
    ) -> tuple[Image.Image, str] | None:
        """
        Định vị dòng chứa Fact trên trang PDF và cắt ảnh độ phân giải cao.

        Returns:
            tuple[PIL.Image, str]: (Ảnh đã crop, Chuỗi base64 PNG) hoặc None nếu không crop được
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_idx = fact.page - 1
                if page_idx < 0 or page_idx >= len(pdf.pages):
                    logger.warning("VisionZoomCorrector: Trang %d ngoài phạm vi PDF (%d trang)", fact.page, len(pdf.pages))
                    return None

                page = pdf.pages[page_idx]
                words = page.extract_words()

                target_box = None
                clean_code = re.sub(r"[^\d]", "", str(fact.standard_code))

                # 1. Tìm theo Mã số chuẩn TT 200 (ví dụ '270', '440', '110')
                if clean_code and words:
                    for w in words:
                        text_w = re.sub(r"[^\d]", "", w["text"])
                        if text_w == clean_code:
                            target_box = (w["top"], w["bottom"])
                            break

                # 2. Nếu không thấy theo mã số, tìm theo từ khóa chính trong raw_label
                if not target_box and words:
                    label_keywords = [
                        k for k in re.findall(r"\w+", fact.raw_label.lower())
                        if len(k) >= 4 and k not in ("báo", "cáo", "tài", "chính", "phần", "mục")
                    ]
                    best_match_y = None
                    best_match_count = 0

                    # Nhóm từ theo dòng (y xấp xỉ nhau)
                    lines_by_y: dict[int, list[dict[str, Any]]] = {}
                    for w in words:
                        y_key = int(w["top"] // 6) * 6
                        lines_by_y.setdefault(y_key, []).append(w)

                    for line_words in lines_by_y.values():
                        line_str = " ".join(w["text"].lower() for w in line_words)
                        match_count = sum(1 for kw in label_keywords if kw in line_str)
                        if match_count > best_match_count:
                            best_match_count = match_count
                            min_top = min(w["top"] for w in line_words)
                            max_bot = max(w["bottom"] for w in line_words)
                            best_match_y = (min_top, max_bot)

                    if best_match_count >= 1:
                        target_box = best_match_y

                # 3. Tính toán Bounding Box cắt dải ngang trang có padding
                full_img = page.to_image(resolution=resolution).original
                img_w, img_h = full_img.size

                if target_box:
                    top_y, bot_y = target_box
                    pad = 12.0
                    crop_bbox = (
                        0.0,
                        max(0.0, top_y - pad),
                        float(page.width),
                        min(float(page.height), bot_y + pad),
                    )
                    cropped_page = page.crop(crop_bbox)
                    pil_img = cropped_page.to_image(resolution=resolution).original
                    # Tọa độ pixel trên ảnh full
                    scale_y = img_h / float(page.height)
                    pixel_box = (0, int(crop_bbox[1] * scale_y), img_w, int(crop_bbox[3] * scale_y))
                else:
                    # Fallback cho trang scan không có native word text: sử dụng OpenCV phát hiện dải kẻ bảng
                    crop_top = int(img_h * 0.3)
                    crop_bot = int(img_h * 0.6)

                    try:
                        import cv2
                        import numpy as np

                        gray = cv2.cvtColor(np.array(full_img), cv2.COLOR_RGB2GRAY)
                        binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
                        cols = binary.shape[1]
                        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // 12, 1))
                        horizontal = cv2.erode(binary, horizontal_structure)
                        horizontal = cv2.dilate(horizontal, horizontal_structure)
                        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        major_lines = sorted([cv2.boundingRect(c)[1] for c in contours if cv2.boundingRect(c)[2] > cols * 0.3])

                        # Nếu là khoản mục Tổng cộng (270, 440) -> Bắt dải gạch chân đôi ở cuối bảng
                        if (fact.concept in ("TOTAL_ASSETS", "TOTAL_RESOURCES") or clean_code in ("270", "440")) and len(major_lines) >= 3:
                            # 1741 thường là chân trang/footer, các đường trước đó là gạch ngang dòng tổng
                            table_lines = [y for y in major_lines if y < img_h * 0.88]
                            if len(table_lines) >= 2:
                                bot_line = table_lines[-1]
                                top_line = table_lines[-2] if (bot_line - table_lines[-2]) < 120 else bot_line - 65
                                crop_top = max(0, top_line - 20)
                                crop_bot = min(img_h, bot_line + 25)
                        elif clean_code in ("100", "200", "300", "400") and len(major_lines) >= 4:
                            # Các chỉ tiêu lớn đầu mục
                            crop_top = max(0, major_lines[0] - 15)
                            crop_bot = min(img_h, major_lines[1] + 25)
                    except Exception as cv_err:
                        logger.debug("VisionZoomCorrector: OpenCV table line detection skipped: %s", cv_err)

                    pil_img = full_img.crop((0, crop_top, img_w, crop_bot))
                    pixel_box = (0, crop_top, img_w, crop_bot)

                # Lưu ảnh debug và ảnh trực quan hóa nếu có yêu cầu
                if save_debug_dir:
                    try:
                        import cv2
                        import numpy as np
                        from pathlib import Path

                        out_dir = Path(save_debug_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)

                        # 1. Lưu ảnh dòng đã crop
                        crop_filename = out_dir / f"zoomed_row_{fact.concept}_p{fact.page}.png"
                        pil_img.save(crop_filename)

                        # 2. Vẽ bounding box đỏ trên ảnh toàn trang để trực quan hóa
                        annotated = np.array(full_img)
                        x0, y0, x1, y1 = pixel_box
                        cv2.rectangle(annotated, (x0 + 4, y0), (x1 - 4, y1), (255, 0, 0), 4)
                        label_tag = f"Suspect Fact: {fact.concept} (Code {fact.standard_code})"
                        cv2.putText(annotated, label_tag, (x0 + 20, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                        annotated_filename = out_dir / f"annotated_page_{fact.page}_{fact.concept}.png"
                        Image.fromarray(annotated).save(annotated_filename)
                        logger.info("VisionZoomCorrector: Đã lưu ảnh debug tại: %s và %s", crop_filename, annotated_filename)
                    except Exception as save_err:
                        logger.warning("VisionZoomCorrector: Không thể lưu ảnh debug: %s", save_err)

                # Encode Base64 PNG
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return pil_img, img_b64

        except Exception as e:
            logger.error("VisionZoomCorrector: Lỗi khi crop dòng cho Fact %s: %s", fact.concept, e)
            return None

    def inspect_row_image(self, img_b64: str, fact: FinancialFact) -> dict[str, Any]:
        """Gửi dải ảnh phóng to đến Vision API để thẩm định lại con số."""
        prompt = ZOOM_SYSTEM_PROMPT.format(
            raw_label=fact.raw_label,
            standard_code=fact.standard_code or "Không có",
        )

        provider = self.settings.ocr_provider

        # Ưu tiên Gemini Vision
        if (provider in ("auto", "gemini")) and self.settings.gemini_api_key:
            res_text = self._call_gemini_zoom(img_b64=img_b64, prompt=prompt)
            if res_text:
                parsed = self._parse_json_result(res_text)
                if parsed:
                    return parsed

        # Fallback sang Groq Vision
        if (provider in ("auto", "groq")) and self.settings.groq_api_key:
            res_text = self._call_groq_zoom(img_b64=img_b64, prompt=prompt)
            if res_text:
                parsed = self._parse_json_result(res_text)
                if parsed:
                    return parsed

        return {}

    def _call_gemini_zoom(self, img_b64: str, prompt: str) -> str:
        """Gọi Gemini Flash Vision với prompt phóng to cục bộ và cơ chế fallback model."""
        api_key = self.settings.gemini_api_key
        if not api_key:
            return ""

        primary_model = self.settings.gemini_vision_model or "gemini-flash-latest"
        candidate_models = list(dict.fromkeys([
            primary_model,
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
        ]))

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 512,
            },
        }

        try:
            import httpx

            with httpx.Client(timeout=30.0) as client:
                for model in candidate_models:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                    try:
                        res = client.post(url, json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                return parts[0]["text"].strip()
                        elif res.status_code in (404, 503, 429):
                            logger.debug("VisionZoomCorrector: Model '%s' trả về HTTP %d, chuyển model dự phòng...", model, res.status_code)
                            continue
                    except Exception as err:
                        logger.warning("VisionZoomCorrector: Lỗi kết nối model %s: %s", model, err)
                        continue
        except Exception as e:
            logger.warning("VisionZoomCorrector: Lỗi gọi Gemini Zoom API: %s", e)

        return ""

    def _call_groq_zoom(self, img_b64: str, prompt: str) -> str:
        """Gọi Groq Vision API để thẩm định ảnh zoom."""
        api_key = self.settings.groq_api_key
        if not api_key:
            return ""

        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            model = self.settings.groq_vision_model or "llama-3.2-11b-vision-preview"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("VisionZoomCorrector: Lỗi gọi Groq Zoom API: %s", e)
            return ""

    def _parse_json_result(self, text: str) -> dict[str, Any] | None:
        """Trích xuất và parse an toàn JSON từ kết quả LLM."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                raw_json = match.group(0)
                data = json.loads(raw_json)
                return data
        except Exception as e:
            logger.warning("VisionZoomCorrector: Không thể parse JSON từ Vision Zoom '%s': %s", text[:100], e)
        return None

    def run_self_correction(
        self,
        pdf_path: str,
        facts: list[FinancialFact],
        report: VerificationReport,
        company: str = "DOANH_NGHIEP",
        year: int = 2024,
        max_attempts: int = 2,
        save_debug_dir: str | None = None,
    ) -> tuple[list[FinancialFact], VerificationReport, bool]:
        """
        Kích hoạt vòng lặp tự sửa sai cục bộ (Closed-loop Self-Correction):
          1. Khoanh vùng các Fact bị nghi ngờ.
          2. Cắt ảnh dòng và gọi Vision LLM đọc lại số.
          3. Hot-patch số mới và re-verify tự động.
          4. Nếu BCTC cân đối (is_balanced == True), cập nhật trạng thái và thoát thành công.

        Returns:
            tuple[list[FinancialFact], VerificationReport, bool]: (facts_mới, report_mới, is_corrected)
        """
        if report.is_balanced:
            return facts, report, False

        suspect_facts = self.identify_suspect_facts(report, facts)
        if not suspect_facts:
            logger.info("VisionZoomCorrector: Không xác định được suspect facts khả dĩ để Zoom.")
            return facts, report, False

        logger.info(
            "VisionZoomCorrector: Phát hiện %d suspect facts cần Zoom thẩm định: %s",
            len(suspect_facts),
            [f.concept for f in suspect_facts[:max_attempts]],
        )

        current_facts = list(facts)
        current_report = report

        for attempt, suspect in enumerate(suspect_facts[:max_attempts], start=1):
            logger.info(
                "VisionZoomCorrector [Lần %d/%d]: Phóng to dòng '%s' (Mã %s, Giá trị hiện tại: %s, Trang %d)...",
                attempt,
                max_attempts,
                suspect.raw_label,
                suspect.standard_code,
                f"{suspect.value:,.0f}",
                suspect.page,
            )

            crop_result = self.locate_and_crop_row(pdf_path=pdf_path, fact=suspect, save_debug_dir=save_debug_dir)
            if not crop_result:
                continue

            _, img_b64 = crop_result
            inspection = self.inspect_row_image(img_b64=img_b64, fact=suspect)

            raw_val = inspection.get("value_current")
            if raw_val is None and "raw_text" in inspection:
                raw_val = clean_ocr_number(inspection["raw_text"])

            if raw_val is None:
                continue

            new_value = float(raw_val)
            old_value = suspect.value

            # Nếu con số đọc lại khác biệt đáng kể so với con số cũ
            if abs(new_value - old_value) > self.verifier.absolute_tolerance:
                logger.info(
                    "VisionZoomCorrector: Phát hiện giá trị mới từ Vision Zoom: %s (Cũ: %s). Thử hot-patch...",
                    f"{new_value:,.0f}",
                    f"{old_value:,.0f}",
                )

                # Áp dụng tạm thời giá trị mới
                suspect.value = new_value

                # Re-verify toàn bộ facts
                new_report = self.verifier.verify_facts(current_facts, company=company, year=year)

                if new_report.is_balanced or len(new_report.failed_checks) < len(current_report.failed_checks):
                    # Sửa lỗi thành công!
                    history_entry = {
                        "concept": suspect.concept,
                        "standard_code": suspect.standard_code,
                        "raw_label": suspect.raw_label,
                        "old_value": old_value,
                        "new_value": new_value,
                        "page": suspect.page,
                        "reason": "Vision-LLM Zoom Self-Correction",
                    }
                    new_report.correction_history.append(history_entry)

                    if new_report.is_balanced:
                        suspect.verification_status = VerificationStatus.VERIFIED_AFTER_ZOOM_CORRECTION
                        suspect.verification_detail = (
                            f"Đã tự động sửa lỗi OCR từ {old_value:,.0f} sang {new_value:,.0f} qua Vision-LLM Zoom."
                        )
                        new_report.summary += (
                            f" [✅ Tự sửa thành công dòng {suspect.concept} từ {old_value:,.0f} -> {new_value:,.0f}]"
                        )
                        logger.info("VisionZoomCorrector: ✅ TỰ SỬA LỖI THÀNH CÔNG! BCTC đã trở nên HOÀN TOÀN CÂN ĐỐI.")
                        return current_facts, new_report, True

                    # Cải thiện được 1 phần
                    current_report = new_report
                else:
                    # Nếu sửa xong lại làm sai lệch nhiều hơn -> Revert
                    logger.warning("VisionZoomCorrector: Giá trị mới không giúp cân đối BCTC. Revert về giá trị cũ.")
                    suspect.value = old_value

        return current_facts, current_report, False
