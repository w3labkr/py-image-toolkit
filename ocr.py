# PaddleOCR을 사용하여 이미지에서 텍스트를 추출하는 스크립트 (OpenCV 전처리 및 폴더/병렬 처리 추가)
import cv2 # OpenCV 라이브러리 임포트
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np # NumPy 임포트 (OpenCV 이미지 처리에 사용)
import os # 파일 및 디렉토리 관리를 위해 os 모듈 임포트
import argparse # 명령줄 인자 처리를 위해 argparse 모듈 임포트
import platform # 운영체제 감지를 위해 platform 모듈 임포트
from tqdm import tqdm # 진행률 표시를 위해 tqdm 임포트
import logging # 로깅 모듈 임포트
import multiprocessing # 병렬 처리를 위해 multiprocessing 임포트

__version__ = "0.8.1" # 스크립트 버전 정보 (num_workers 옵션 제거)

# --- 로거 설정 ---
# logger는 각 프로세스에서 호출될 때 해당 프로세스의 로거를 사용하게 됩니다.
# 기본 설정은 main에서 한 번 수행합니다.
logger = logging.getLogger(__name__)

def setup_logger(level=logging.INFO):
    """기본 로거를 설정합니다."""
    # 핸들러가 이미 설정되어 있다면 중복 추가하지 않음 (주 프로세스에서만 설정)
    if not logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.setLevel(level)
        # 기본적으로 tqdm과 잘 동작하도록 StreamHandler 사용
        ch = logging.StreamHandler() 
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else: # 이미 핸들러가 있다면 레벨만 조정
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def preprocess_image_for_ocr(image_path):
    """
    OCR 정확도 향상을 위해 이미지를 전처리합니다.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지를 불러올 수 없습니다: {image_path}")
            return None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_img = clahe.apply(gray_img)
        denoised_img = cv2.medianBlur(enhanced_contrast_img, 3)
        logger.debug(f"전처리 완료: {os.path.basename(image_path)}")
        return denoised_img
    except cv2.error as e:
        logger.error(f"OpenCV 오류 (전처리, 파일: {os.path.basename(image_path)}): {e}")
        return None
    except Exception as e:
        logger.error(f"예외 발생 (전처리, 파일: {os.path.basename(image_path)}): {e}")
        return None

def extract_text_from_image_worker(ocr_engine_params, image_data, filename_for_log=""):
    """
    주어진 이미지 데이터에서 텍스트를 추출합니다. (작업자 프로세스용)
    이 함수는 각 작업자 프로세스 내에서 PaddleOCR 엔진을 초기화합니다.
    """
    try:
        # 각 작업자 프로세스 내에서 PaddleOCR 엔진 초기화
        logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 초기화 중... (파일: {filename_for_log})")
        ocr_engine = PaddleOCR(**ocr_engine_params)
        logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 초기화 완료. (파일: {filename_for_log})")

        logger.debug(f"작업자 {os.getpid()}: OCR 시작: {filename_for_log}")
        result = ocr_engine.ocr(image_data, cls=True)
        logger.debug(f"작업자 {os.getpid()}: OCR 완료: {filename_for_log}")

        extracted_texts = []
        if result and result[0] is not None:
            for line_info in result[0]:
                text, confidence = line_info[1]
                bounding_box = line_info[0]
                extracted_texts.append({
                    "text": text,
                    "confidence": confidence,
                    "bounding_box": bounding_box
                })
        return extracted_texts
    except Exception as e:
        logger.error(f"작업자 {os.getpid()}: OCR 처리 중 오류 (파일: {filename_for_log}): {e}")
        return None # 오류 발생 시 None 반환


def get_os_specific_font_path():
    """
    운영 체제에 맞는 기본 한글 폰트 경로를 반환합니다.
    """
    system = platform.system()
    font_path = None
    if system == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf" 
    elif system == "Darwin": 
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc" 
    elif system == "Linux":
        common_linux_fonts = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/korean/NanumGothic.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 
        ]
        for p in common_linux_fonts:
            if os.path.exists(p):
                font_path = p
                break
    
    if font_path and os.path.exists(font_path):
        logger.debug(f"OS ({system}) 자동 감지 폰트 확인: {font_path}")
        return font_path
    else:
        logger.debug(f"OS ({system}) 자동 감지 폰트 없음 또는 경로 문제.")
        return None

def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename, 
                       preprocessed_img=None, show_image_flag=False, font_path_to_use=None):
    """
    OCR 결과를 이미지 위에 표시하고 저장합니다.
    font_path_to_use: 외부에서 결정된 폰트 경로를 전달받음
    """
    if not extracted_data:
        logger.info(f"{original_filename}: 시각화할 OCR 결과가 없습니다.")
        return

    try:
        if preprocessed_img is not None:
            image_to_draw_on = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB) if len(preprocessed_img.shape) == 2 else preprocessed_img
            image = Image.fromarray(image_to_draw_on)
        else:
            image = Image.open(original_image_path).convert('RGB')

        boxes = [item['bounding_box'] for item in extracted_data]
        txts = [item['text'] for item in extracted_data]
        scores = [item['confidence'] for item in extracted_data]
        
        try:
            im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path_to_use)
        except Exception as e:
            logger.error(f"draw_ocr 중 오류 (폰트: {font_path_to_use}, 파일: {original_filename}): {e}. 폰트 없이 재시도.")
            im_show = draw_ocr(image, boxes, font_path=None) 

        im_show_pil = Image.fromarray(im_show)
        base, ext = os.path.splitext(original_filename)
        output_image_filename = f"{base}_ocr_result{ext}" 
        output_image_path = os.path.join(output_dir, output_image_filename)
        
        im_show_pil.save(output_image_path)
        logger.debug(f"OCR 결과 시각화 이미지가 {output_image_path}에 저장되었습니다.")
        
        if show_image_flag:
            im_show_pil.show()

    except FileNotFoundError:
        logger.error(f"원본 이미지 파일 '{original_image_path}'을 찾을 수 없습니다 (결과 표시 중).")
    except Exception as e:
        logger.error(f"OCR 결과 표시/저장 중 예외 발생 (파일: {original_filename}): {e}")


def save_extracted_text(extracted_data, output_dir, original_filename):
    """추출된 텍스트를 .txt 파일로 저장합니다."""
    if not extracted_data:
        logger.info(f"{original_filename}: 저장할 추출된 텍스트가 없습니다.")
        return

    base, _ = os.path.splitext(original_filename)
    output_text_filename = f"{base}_ocr_text.txt" 
    output_text_path = os.path.join(output_dir, output_text_filename)

    try:
        with open(output_text_path, 'w', encoding='utf-8') as f:
            for item in extracted_data:
                f.write(item['text'] + '\n')
        logger.info(f"추출된 텍스트가 {output_text_path}에 저장되었습니다.")
    except IOError as e:
        logger.error(f"텍스트 파일 저장 중 오류 (파일: {output_text_path}): {e}")


# 병렬 처리를 위한 작업자 함수
def process_single_image_task(args_tuple):
    """
    단일 이미지 처리 작업을 수행하는 함수 (multiprocessing.Pool의 작업자용).
    """
    # 인자 언패킹
    (current_image_path, filename, ocr_engine_params, output_dir_path, 
     skip_preprocessing, save_text_flag, display_images_flag, determined_font_path) = args_tuple

    logger.debug(f"작업자 {os.getpid()}: 처리 시작: {filename}")
    
    ocr_input_data = current_image_path
    processed_image_data_for_display = None # 시각화에 사용될 전처리된 이미지

    if not skip_preprocessing:
        logger.debug(f"작업자 {os.getpid()}: 외부 전처리 시작: {filename}")
        processed_img = preprocess_image_for_ocr(current_image_path)
        if processed_img is not None:
            ocr_input_data = processed_img # 전처리된 NumPy 배열 사용
            processed_image_data_for_display = processed_img
        else:
            logger.warning(f"작업자 {os.getpid()}: {filename}: 외부 전처리에 실패하여 원본 이미지로 OCR을 시도합니다.")
    
    extracted_data = extract_text_from_image_worker(ocr_engine_params, ocr_input_data, filename_for_log=filename)

    if extracted_data:
        logger.debug(f"작업자 {os.getpid()}: 추출된 텍스트 항목 수 ({filename}): {len(extracted_data)}")
        if logger.isEnabledFor(logging.DEBUG):
            for i, item in enumerate(extracted_data):
                 logger.debug(f"  - \"{item['text']}\" (신뢰도: {item['confidence']:.3f})")
        
        if save_text_flag:
            save_extracted_text(extracted_data, output_dir_path, filename)

        display_ocr_result(current_image_path, extracted_data, output_dir_path, filename,
                           preprocessed_img=processed_image_data_for_display, 
                           show_image_flag=display_images_flag,
                           font_path_to_use=determined_font_path) # 결정된 폰트 경로 전달
    else:
        logger.info(f"작업자 {os.getpid()}: {filename}에서 텍스트를 추출하지 못했습니다.")
    
    return f"{filename} 처리 완료" # 작업 완료 시 간단한 메시지 반환 (선택 사항)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    setup_logger() 

    parser = argparse.ArgumentParser(description="이미지 폴더에서 텍스트를 추출하는 OCR 스크립트입니다.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_dir", nargs='?', default='input', 
                        help="텍스트를 추출할 이미지가 포함된 폴더의 경로입니다.\n(기본값: 'input')")
    parser.add_argument("--output_dir", default='output', 
                        help="OCR 결과 이미지 및 텍스트 파일을 저장할 폴더 경로입니다.\n(기본값: 'output')")
    parser.add_argument("--lang", default='korean',
                        help="OCR에 사용할 언어입니다. 예: 'korean', 'en', 'japan', 'ch_sim'.\n(기본값: 'korean')")
    # --num_workers 옵션 제거됨
    parser.add_argument("--show_image", action='store_true', help="처리된 각 이미지와 OCR 결과를 화면에 표시합니다.")
    parser.add_argument("--no_preprocess", action='store_true', help="이미지 전처리 단계를 건너뜁니다.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}', help="스크립트 버전을 표시하고 종료합니다.")
    parser.add_argument('--debug', action='store_true', help="디버그 레벨 로깅을 활성화하여 더 상세한 로그를 출력합니다.")
    parser.add_argument('--use_gpu', action='store_true', help="사용 가능한 경우 GPU를 사용하여 OCR 처리를 시도합니다.\n(NVIDIA GPU 및 CUDA 환경 필요)")
    parser.add_argument('--save_text', action='store_true', help="추출된 텍스트를 이미지와 동일한 기반 이름의 .txt 파일로 저장합니다.")

    args = parser.parse_args()

    if args.debug:
        setup_logger(logging.DEBUG) 
        logger.debug("디버그 모드가 활성화되었습니다.")

    input_dir_path = args.input_dir
    output_dir_path = args.output_dir
    
    if not os.path.isdir(input_dir_path):
        logger.error(f"입력 디렉토리 '{input_dir_path}'가 없거나 디렉토리가 아닙니다. 종료합니다.")
        exit(1)
    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            logger.info(f"출력 디렉토리 생성: {output_dir_path}")
        except OSError as e:
            logger.error(f"출력 디렉토리 '{output_dir_path}' 생성 실패: {e}")
            exit(1)

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files_to_process_names = [
        f for f in os.listdir(input_dir_path) if f.lower().endswith(supported_extensions)
    ]
    
    if not image_files_to_process_names:
        logger.warning(f"입력 디렉토리 '{input_dir_path}'에서 지원되는 이미지 파일을 찾을 수 없습니다.")
        exit(0)

    # num_workers는 항상 시스템 CPU 코어 수로 설정
    num_workers = os.cpu_count()
    logger.info(f"사용할 작업자 프로세스 수 (시스템 CPU 코어 수): {num_workers}")

    logger.info(f"OCR 스크립트 버전: {__version__}")
    logger.info(f"입력 디렉토리: {input_dir_path}")
    logger.info(f"출력 디렉토리: {output_dir_path}")
    logger.info(f"선택 언어: {args.lang}")
    logger.info(f"GPU 사용 시도: {args.use_gpu}")
    logger.info(f"전처리 수행: {not args.no_preprocess}")
    logger.info(f"결과 이미지 표시: {args.show_image}")
    logger.info(f"추출 텍스트 파일 저장: {args.save_text}")
    logger.info(f"총 {len(image_files_to_process_names)}개의 이미지 파일을 처리합니다.")

    ocr_engine_params = {
        'use_angle_cls': True, 
        'lang': args.lang, 
        'use_gpu': args.use_gpu, 
        'show_log': False 
    }
    if '+' in args.lang:
        logger.info(f"복합 언어 설정 '{args.lang}' 감지. PaddleOCR이 해당 설정을 지원하는지 확인하세요.")

    font_path_for_display = get_os_specific_font_path()
    # 폰트 경로 결정에 대한 로그는 get_os_specific_font_path 내부 또는 아래에서 한 번만 출력되도록 관리
    font_message_logged_this_run = False # 이 실행에서 폰트 메시지가 출력되었는지 추적
    if font_path_for_display:
        logger.info(f"텍스트 시각화에 사용할 자동 감지 폰트: {font_path_for_display}")
        font_message_logged_this_run = True
    else:
        local_korean_font = './fonts/malgun.ttf'
        if os.path.exists(local_korean_font):
            font_path_for_display = local_korean_font
            logger.info(f"로컬 한국어 폰트 사용 시도: {font_path_for_display}")
            font_message_logged_this_run = True
        else:
            local_english_font = './fonts/arial.ttf'
            if os.path.exists(local_english_font):
                font_path_for_display = local_english_font
                logger.warning(f"로컬 영문 대체 폰트 사용 시도: {font_path_for_display} (한글 깨짐 가능성)")
                font_message_logged_this_run = True
            else:
                logger.warning("사용 가능한 특정 폰트를 찾지 못했습니다. 시각화 시 PaddleOCR 내부 기본 폰트 사용.")
                font_message_logged_this_run = True # 메시지를 남겼으므로 true

    tasks_args = []
    for filename in image_files_to_process_names:
        current_image_path = os.path.join(input_dir_path, filename)
        tasks_args.append((
            current_image_path, 
            filename, 
            ocr_engine_params, 
            output_dir_path,
            args.no_preprocess, 
            args.save_text, 
            args.show_image,
            font_path_for_display 
        ))

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.starmap(process_single_image_task, tasks_args), 
                                total=len(tasks_args), desc="전체 이미지 처리 중"))
            for res_msg in results:
                if res_msg: 
                    logger.debug(res_msg) 
    except Exception as e:
        logger.error(f"병렬 처리 중 주 프로세스에서 오류 발생: {e}")
        
    logger.info(f"총 {len(image_files_to_process_names)}개의 이미지 파일 처리가 완료되었습니다.")
    logger.info("스크립트 실행 종료.")

