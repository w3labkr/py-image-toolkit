# -*- coding: utf-8 -*-
import os
import sys
import re
from paddleocr import PaddleOCR
import argparse
import traceback

LABELS = {
    "O": ["기타"],  # Other: 특별한 의미가 없는 기타 텍스트 또는 배경 요소
    "name": ["이름", "성명"],  # 성명: 개인의 이름 (예: 홍길동)
    "rrn": [
        "주민등록번호",
        "주민번호",
    ],  # 주민등록번호 (Resident Registration Number): 개인 식별 번호 (예: 123456-1234567)
    "issue_date": [
        "발급일",
        "작성일",
        "발행일",
    ],  # 발급일: 문서가 발행된 날짜 (예: 2023-05-15)
    "expiration_date": [
        "만료일",
        "유효기간",
    ],  # 만료일: 문서나 자격의 유효기간이 끝나는 날짜
    "address": ["주소", "거주지", "소재지"],  # 주소: 현재 거주지 또는 사업장 주소
    "previous_address": ["이전주소", "전주소"],  # 이전 주소: 변경 전 주소
    "department": [
        "부서",
        "소속부서",
        "학과",
        "부",
    ],  # 부서: 조직 내의 특정 부서명 (예: 인사팀, 재무부, 학과)
    "position": [
        "직책",
        "직위",
        "담당",
        "급",
    ],  # 직위/직책: 조직 내에서의 역할이나 직급 (예: 과장, 팀장, 교수)
    "organization": [
        "소속",
        "학교명",
        "회사명",
        "기관명",
        "단체명",
    ],  # 소속/기관명: 개인이 속한 회사, 학교, 단체 등의 이름 (예: ㈜한국데이터, 서울대학교)
    "employment_status": [
        "재직",
        "휴직",
        "퇴직",
        "고용상태",
    ],  # 재직 상태: 고용 상태 (예: 재직, 휴직, 퇴직)
    "employment_date": [
        "입사일",
        "임용일",
        "채용일",
    ],  # 입사일/임용일: 조직에 고용되거나 임용된 날짜
    "retirement_date": [
        "퇴직일",
        "면직일",
        "해고일",
    ],  # 퇴직일/면직일: 조직에서 퇴사하거나 면직된 날짜
    "issuer": [
        "발급기관",
        "발행처",
        "기관명",
        "발급자",
    ],  # 발급기관: 문서를 발행한 기관이나 단체 (예: 서울특별시청, 한국산업인력공단)
    "document_title": [
        "주민등록증",
        "문서명",
        "서류명",
        "제목",
        "운전면허증",
        "공무원증",
        "여권",
        "신분증",
    ],  # 문서명: 해당 문서의 제목 (예: 사업자등록증, 졸업증명서, 공문)
    "id_number": [
        "문서번호",
        "번호",
        "등록번호",
        "관리번호",
    ],  # 문서번호/식별번호: 문서 자체의 고유 번호나 관리 번호 (예: 제2023-123호)
    "signature": [
        "서명",
        "직인",
        "날인",
        "도장",
    ],  # 서명/직인: 개인의 서명 또는 기관의 도장(직인)이 있는 영역
    "valid_period": [
        "유효기간",
        "사용기한",
    ],  # 유효 기간: 특정 권리나 자격이 유효한 전체 기간 (시작일 ~ 종료일 형태 포함)
    "note": [
        "비고",
        "주의사항",
        "특이사항",
        "참고사항",
    ],  # 비고/특이사항: 기타 참고해야 할 내용이나 특별한 조건
}

OCR_SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")


def extract_text_items_with_paddleocr(image_path_arg, paddleocr_args):
    try:
        ocr_engine = PaddleOCR(**vars(paddleocr_args))
        result = ocr_engine.ocr(image_path_arg, cls=paddleocr_args.use_angle_cls)

        extracted_items = []
        if result and result[0] is not None:
            for line_info in result[0]:
                if (
                    len(line_info) == 2
                    and isinstance(line_info[1], tuple)
                    and len(line_info[1]) == 2
                ):
                    text_content = line_info[1][0]
                    confidence = line_info[1][1]
                    box = line_info[0]
                    extracted_items.append(
                        {
                            "text": text_content,
                            "confidence": confidence,
                            "box": box,
                            "label": "O",
                        }
                    )
        return extracted_items
    except ImportError as ie:
        print(
            f"Library-related error: {ie}. Please check if paddleocr and paddlepaddle are properly installed."
        )
        return []
    except Exception as e:
        print(f"Unexpected error occurred while processing '{image_path_arg}': {e}")
        if paddleocr_args.show_log:
            traceback.print_exc()
        return []


def apply_labeling_heuristics(items):
    labeled_items = [item.copy() for item in items]

    address_keywords_hardcoded = [
        "특별시",
        "광역시",
        "도",
        "시",
        "군",
        "구",
        "읍",
        "면",
        "동",
        "리",
        "로",
        "길",
        "번길",
        "대로",
        "번지",
        "아파트",
        "빌라",
        "연립",
        "주택",
        "맨션",
        "오피스텔",
        "타워",
        "빌딩",
        "주공",
        "현대",
        "삼성",
        "자이",
        "푸르지오",
        "더샵",
        "롯데캐슬",
        "아이파크",
    ]

    for i, item in enumerate(labeled_items):
        current_text = item["text"]

        if current_text in LABELS.get("document_title", []):
            item["label"] = "document_title"
        elif re.fullmatch(r"\d{6}\s*-\s*\d{7}", current_text) or re.fullmatch(
            r"\d{13}", current_text.replace("-", "").replace(" ", "")
        ):
            item["label"] = "rrn"
        elif any(
            current_text.endswith(s)
            for s in [
                "청장",
                "시장",
                "경찰서장",
                "구청장",
                "교육감",
                "공단이사장",
                "총장",
                "장관",
            ]
        ):
            item["label"] = "issuer"
        elif (
            (
                any(
                    kw in current_text
                    for kw in ["경찰서", "구청", "시청", "도청", "공단", "법원"]
                )
                and any(current_text.endswith(s) for s in ["청", "서"])
            )
            or "학교장" in current_text
            or current_text.endswith("학교")
            and "대학교" not in current_text
        ):
            item["label"] = "issuer"

        is_year = re.fullmatch(r"(?:19|20)\d{2}", current_text)
        is_month = re.fullmatch(r"(0?[1-9]|1[0-2])", current_text)
        is_day = re.fullmatch(r"(0?[1-9]|[12]\d|3[01])", current_text)

        if item["label"] == "O" and (is_year or is_month or is_day):
            prev_item_is_address_component = False
            if i > 0:
                prev_text = labeled_items[i - 1]["text"]
                if (
                    any(kw in prev_text for kw in ["로", "길", "번길", "대로"])
                    and current_text.isdigit()
                    and len(current_text) <= 2
                ):
                    prev_item_is_address_component = True
            if not prev_item_is_address_component:
                item["label"] = "issue_date"

        if item["label"] == "O" or (
            item["label"] == "issue_date" and not (is_year or is_month or is_day)
        ):
            if (
                any(kw in current_text for kw in address_keywords_hardcoded)
                or re.search(r"^\d+(?:-\d+)?\s*[가-힣]*[동호층]$", current_text)
                or re.search(
                    r"^[가-힣0-9]+(?:로|길)\s*\d*(?:번길)?(?:[가-힣])?", current_text
                )
                or re.search(r"^[가-힣]+\d*[가-힣]*[동리가로길]$", current_text)
                or re.search(r"^\d{3,}-\d{3,}$", current_text)
                or (
                    len(current_text.split()) > 1
                    and any(
                        kw in current_text
                        for kw in ["로", "길", "동", "호", "번지", "아파트"]
                    )
                )
            ):
                is_potential_short_date_num = (
                    current_text.isdigit() and len(current_text) <= 2
                )
                has_clear_address_indicator = any(
                    kw in current_text for kw in ["동", "호", "길", "로", "번지", "층"]
                )
                if not (
                    is_potential_short_date_num and not has_clear_address_indicator
                ):
                    item["label"] = "address"

    doc_title_indices = [
        j
        for j, item_j in enumerate(labeled_items)
        if item_j["label"] == "document_title"
    ]
    rrn_indices = [
        j for j, item_j in enumerate(labeled_items) if item_j["label"] == "rrn"
    ]
    name_labeled_successfully = False

    if doc_title_indices and rrn_indices:
        doc_title_idx = doc_title_indices[0]

        relevant_rrn_idx = -1
        for r_idx in rrn_indices:
            if r_idx > doc_title_idx:
                relevant_rrn_idx = r_idx
                break

        if relevant_rrn_idx != -1:
            if relevant_rrn_idx == doc_title_idx + 2:
                potential_name_item_idx = doc_title_idx + 1
                potential_name_item = labeled_items[potential_name_item_idx]
                potential_name_text = potential_name_item["text"]

                is_name_pattern = re.fullmatch(r"^[가-힣]{2,5}$", potential_name_text)
                is_not_other_critical_keyword = not any(
                    kw == potential_name_text or kw in potential_name_text
                    for kw_list in [
                        LABELS.get("document_title", []),
                        LABELS.get("rrn", []),
                        LABELS.get("address", []),
                        address_keywords_hardcoded,
                    ]
                    for kw in kw_list
                )

                if (
                    is_name_pattern
                    and is_not_other_critical_keyword
                    and potential_name_item["label"]
                    not in ["document_title", "rrn", "issuer"]
                ):
                    potential_name_item["label"] = "name"
                    name_labeled_successfully = True

            elif not name_labeled_successfully and relevant_rrn_idx > doc_title_idx + 1:
                for k in range(doc_title_idx + 1, relevant_rrn_idx):
                    item_to_check = labeled_items[k]
                    item_text = item_to_check["text"]

                    is_name_pattern = re.fullmatch(r"^[가-힣]{2,5}$", item_text)
                    is_not_other_critical_keyword = not any(
                        kw == item_text or kw in item_text
                        for kw_list in [
                            LABELS.get("document_title", []),
                            LABELS.get("rrn", []),
                            LABELS.get("address", []),
                            address_keywords_hardcoded,
                        ]
                        for kw in kw_list
                    )

                    if (
                        item_to_check["label"] != "name"
                        and item_to_check["label"] in ["O", "address", "issue_date"]
                        and is_name_pattern
                        and is_not_other_critical_keyword
                    ):
                        if item_to_check["label"] == "address" and len(item_text) > 5:
                            continue
                        item_to_check["label"] = "name"
                        name_labeled_successfully = True
                        break

    for j in range(len(labeled_items) - 1):
        current_item = labeled_items[j]
        next_item = labeled_items[j + 1]
        if (
            (current_item["label"] == "address" or current_item["label"] == "O")
            and next_item["label"] == "issuer"
            and any(
                kw in current_item["text"]
                for kw in ["특별시", "광역시", "도", "시", "군", "구"]
            )
            and len(current_item["text"]) < 10
        ):
            current_item["label"] = "issuer"

    return labeled_items


def merge_and_format_items(labeled_items):
    merged_items_dict = {
        "document_title": None,
        "rrn": None,
        "address": None,
        "issue_date": None,
        "issuer": None,
        "name": None,
    }
    temp_merged_list = []
    i = 0
    n = len(labeled_items)
    while i < n:
        item = labeled_items[i]
        current_label = item["label"]
        current_text = item["text"]

        if current_label == "address":
            address_parts = [current_text]
            j = i + 1
            while j < n and labeled_items[j]["label"] == "address":
                address_parts.append(labeled_items[j]["text"])
                j += 1
            merged_text = " ".join(address_parts)
            temp_merged_list.append({"text": merged_text, "label": "address"})
            i = j
        elif current_label == "issue_date":
            date_components = []
            original_date_items = []
            j = i
            while (
                j < n
                and labeled_items[j]["label"] == "issue_date"
                and len(date_components) < 3
            ):
                date_components.append(labeled_items[j]["text"])
                original_date_items.append(labeled_items[j])
                j += 1

            if (
                len(date_components) == 3
                and re.fullmatch(r"(?:19|20)\d{2}", date_components[0])
                and re.fullmatch(r"(0?[1-9]|1[0-2])", date_components[1])
                and re.fullmatch(r"(0?[1-9]|[12]\d|3[01])", date_components[2])
            ):
                merged_text = (
                    f"{date_components[0]}.{date_components[1]}.{date_components[2]}"
                )
                temp_merged_list.append({"text": merged_text, "label": "issue_date"})
                i = j
            else:
                temp_merged_list.extend(original_date_items)
                i = j
        elif current_label == "issuer":
            issuer_parts = [current_text]
            j = i + 1
            while j < n and labeled_items[j]["label"] == "issuer":
                issuer_parts.append(labeled_items[j]["text"])
                j += 1
            merged_text = " ".join(issuer_parts)
            temp_merged_list.append({"text": merged_text, "label": "issuer"})
            i = j
        else:
            temp_merged_list.append(item)
            i += 1

    main_address_found = False
    for item in temp_merged_list:
        label = item["label"]
        if label == "document_title" and not merged_items_dict["document_title"]:
            merged_items_dict["document_title"] = item
        elif label == "rrn" and not merged_items_dict["rrn"]:
            merged_items_dict["rrn"] = item
        elif label == "address" and not main_address_found:
            merged_items_dict["address"] = item
            main_address_found = True
        elif label == "issue_date" and not merged_items_dict["issue_date"]:
            if "." in item["text"] and len(item["text"].split(".")) == 3:
                merged_items_dict["issue_date"] = item
        elif label == "issuer" and not merged_items_dict["issuer"]:
            merged_items_dict["issuer"] = item
        elif label == "name" and not merged_items_dict["name"]:
            merged_items_dict["name"] = item

    return merged_items_dict


def process_image_file(image_file_path, paddleocr_args):
    extracted_data = extract_text_items_with_paddleocr(image_file_path, paddleocr_args)

    if not extracted_data:
        return f"--- No text extracted for '{os.path.basename(image_file_path)}' ---"

    labeled_data = apply_labeling_heuristics(extracted_data)
    final_output_data = merge_and_format_items(labeled_data)

    output_lines = [
        f"\n--- Extraction Results for '{os.path.basename(image_file_path)}' ---"
    ]

    desired_output_fields = [
        (LABELS.get("document_title", ["문서 제목"])[0], "document_title"),
        (LABELS.get("name", ["이름"])[0], "name"),
        (LABELS.get("address", ["주소"])[0], "address"),
        (LABELS.get("rrn", ["주민등록번호"])[0], "rrn"),
        (LABELS.get("issue_date", ["발급일"])[0], "issue_date"),
        (LABELS.get("issuer", ["발급기관"])[0], "issuer"),
    ]

    for header_text, internal_label in desired_output_fields:
        item_data = final_output_data.get(internal_label)
        value_to_print = item_data["text"] if item_data and "text" in item_data else ""
        output_lines.append(f'"{header_text}": {value_to_print}')

    return "\n".join(output_lines)


def run(input_path_to_process, paddleocr_args):
    if not os.path.exists(input_path_to_process):
        print(f"The specified path '{input_path_to_process}' cannot be found.")
        sys.exit(1)

    if os.path.isfile(input_path_to_process):
        if input_path_to_process.lower().endswith(OCR_SUPPORTED_EXTENSIONS):
            result_string = process_image_file(input_path_to_process, paddleocr_args)
            if result_string:
                print(result_string)
        else:
            print(
                f"File '{input_path_to_process}' is not a supported image file. Supported extensions: {OCR_SUPPORTED_EXTENSIONS}"
            )
            sys.exit(1)
    else:
        print(
            f"Path '{input_path_to_process}' is not a valid file. Directory processing is not supported."
        )
        sys.exit(1)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extracts text from images using PaddleOCR and applies labeling heuristics. Processes a single image file."
    )
    parser.add_argument("input_file", help="Path to the image file to process.")

    # PaddleOCR arguments
    paddleocr_group = parser.add_argument_group(
        "PaddleOCR Arguments", "Arguments for configuring PaddleOCR"
    )
    paddleocr_group.add_argument(
        "--lang", type=str, default="korean", help="Language for OCR. Default: 'korean'"
    )
    paddleocr_group.add_argument(
        "--rec_model_dir",
        type=str,
        default="./models/ko_PP-OCRv3_rec_infer",
        help="Path to recognition model directory. Default: './models/ko_PP-OCRv3_rec_infer'",
    )
    paddleocr_group.add_argument(
        "--det_model_dir",
        type=str,
        default="./models/ch_PP-OCRv3_det_infer",
        help="Path to detection model directory. Default: './models/ch_PP-OCRv3_det_infer'",
    )
    paddleocr_group.add_argument(
        "--cls_model_dir",
        type=str,
        default="./models/ch_ppocr_mobile_v2.0_cls_infer",
        help="Path to classification model directory. Default: './models/ch_ppocr_mobile_v2.0_cls_infer'",
    )
    paddleocr_group.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Use GPU for OCR. Default: False",
    )
    paddleocr_group.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=None,
        help="Path to recognition character dictionary. Default: None (uses default)",
    )
    paddleocr_group.add_argument(
        "--rec_batch_num",
        type=int,
        default=6,
        help="Recognition batch size. Default: 6",
    )
    paddleocr_group.add_argument(
        "--det_db_thresh",
        type=float,
        default=0.4,
        help="Detection DB threshold. Default: 0.4",
    )
    paddleocr_group.add_argument(
        "--det_db_box_thresh",
        type=float,
        default=0.6,
        help="Detection DB box threshold. Default: 0.6",
    )
    paddleocr_group.add_argument(
        "--det_db_unclip_ratio",
        type=float,
        default=1.8,
        help="Detection DB unclip ratio. Default: 1.8",
    )
    paddleocr_group.add_argument(
        "--drop_score",
        type=float,
        default=0.6,
        help="Drop score for text detection. Default: 0.6",
    )
    paddleocr_group.add_argument(
        "--cls_thresh",
        type=float,
        default=0.9,
        help="Classification threshold. Default: 0.9",
    )
    paddleocr_group.add_argument(
        "--use_angle_cls",
        action="store_true",
        default=False,
        help="Use angle classification. Default: False",
    )
    paddleocr_group.add_argument(
        "--use_space_char",
        action="store_true",
        default=True,
        help="Use space character. Default: True",
    )
    paddleocr_group.add_argument(
        "--use_dilation",
        action="store_true",
        default=True,
        help="Use dilation on text regions. Default: True",
    )
    paddleocr_group.add_argument(
        "--show_log",
        action="store_true",
        default=False,
        help="Show PaddleOCR logs. Default: False",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file_arg = args.input_file

    paddleocr_args = argparse.Namespace()
    paddleocr_group = next(
        ag for ag in parser._action_groups if ag.title == "PaddleOCR Arguments"
    )
    for action in paddleocr_group._group_actions:
        setattr(paddleocr_args, action.dest, getattr(args, action.dest))

    run(input_file_arg, paddleocr_args)


if __name__ == "__main__":
    main()
