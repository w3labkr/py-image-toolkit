# PaddleOCR을 사용하여 이미지에서 텍스트를 추출하는 스크립트 (OpenCV 전처리 및 폴더 처리 추가)
import cv2 # OpenCV 라이브러리 임포트
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np # NumPy 임포트 (OpenCV 이미지 처리에 사용)
import os # 파일 및 디렉토리 관리를 위해 os 모듈 임포트
import argparse # 명령줄 인자 처리를 위해 argparse 모듈 임포트
import platform # 운영체제 감지를 위해 platform 모듈 임포트
from tqdm import tqdm # 진행률 표시를 위해 tqdm 임포트
import logging # 로깅 모듈 임포트

__version__ = "0.7.0" # 스크립트 버전 정보 (PaddleOCR 객체 재사용, tqdm 개선 등)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

def setup_logger(level=logging.INFO):
    """기본 로거를 설정합니다."""
    if not logger.handlers: # 핸들러가 중복 추가되는 것을 방지
        logger.setLevel(level)
        ch = logging.StreamHandler() # 콘솔 핸들러
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

def preprocess_image_for_ocr(image_path):
    """
    OCR 정확도 향상을 위해 이미지를 전처리합니다.
    (그레이스케일, 대비 향상, 노이즈 제거)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지를 불러올 수 없습니다. 경로를 확인하세요: {image_path}")
            return None

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_img = clahe.apply(gray_img)
        denoised_img = cv2.medianBlur(enhanced_contrast_img, 3)
        logger.debug(f"이미지 전처리 완료: {os.path.basename(image_path)}")
        return denoised_img
    except cv2.error as e:
        logger.error(f"OpenCV 오류 (전처리 중, 파일: {os.path.basename(image_path)}): {e}")
        return None
    except Exception as e:
        logger.error(f"예외 발생 (전처리 중, 파일: {os.path.basename(image_path)}): {e}")
        return None

def extract_text_from_image(ocr_engine, image_data, filename_for_log=""):
    """
    주어진 이미지 데이터에서 텍스트를 추출합니다. (PaddleOCR 엔진 객체를 인자로 받음)
    
    Args:
        ocr_engine (PaddleOCR): 초기화된 PaddleOCR 엔진 인스턴스.
        image_data (numpy.ndarray or str): OCR을 수행할 이미지 데이터(NumPy 배열) 또는 이미지 경로.
        filename_for_log (str, optional): 로그 출력 시 사용할 파일 이름입니다.

    Returns:
        list: 추출된 텍스트 정보 리스트. 각 항목은 딕셔너리 형태입니다.
              오류 발생 시 None을 반환합니다.
    """
    try:
        logger.debug(f"OCR 시작: {filename_for_log}")
        # PaddleOCR v2.6 이상에서는 ocr() 메서드가 직접 경로와 numpy 배열을 모두 처리합니다.
        result = ocr_engine.ocr(image_data, cls=True)
        logger.debug(f"OCR 완료: {filename_for_log}")

        extracted_texts = []
        if result and result[0] is not None: # PaddleOCR 결과 형식에 따라 result[0] 확인
            for line_info in result[0]:
                text = line_info[1][0]
                confidence = line_info[1][1]
                bounding_box = line_info[0]
                extracted_texts.append({
                    "text": text,
                    "confidence": confidence,
                    "bounding_box": bounding_box
                })
        return extracted_texts

    except Exception as e:
        logger.error(f"OCR 처리 중 오류 발생 (파일: {filename_for_log}): {e}")
        return None


# 폰트 관련 메시지 중복 출력을 방지하기 위한 플래그 (함수 속성으로 관리)
def _initialize_font_flags():
    if not hasattr(get_os_specific_font_path, 'font_message_printed'):
        get_os_specific_font_path.font_message_printed = False
    if not hasattr(display_ocr_result, 'final_font_choice_logged'):
        display_ocr_result.final_font_choice_logged = False
_initialize_font_flags()


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
            # "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # 영문 폰트는 최후의 보루
        ]
        for p in common_linux_fonts:
            if os.path.exists(p):
                font_path = p
                break
    
    if not get_os_specific_font_path.font_message_printed:
        if font_path and os.path.exists(font_path):
            logger.info(f"OS ({system}) 자동 감지 폰트 사용 시도: {font_path}")
        elif font_path: 
            logger.warning(f"OS ({system}) 자동 감지 폰트 '{font_path}'를 찾을 수 없습니다.")
        else: 
            logger.warning(f"OS ({system})에 대한 기본 한글 폰트 경로를 자동 감지하지 못했습니다.")
        get_os_specific_font_path.font_message_printed = True

    if font_path and os.path.exists(font_path):
        return font_path
    return None

def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename, 
                       preprocessed_img=None, show_image_flag=False):
    """
    OCR 결과를 이미지 위에 표시하고 저장합니다.
    """
    if not extracted_data:
        logger.info(f"{original_filename}: 시각화할 OCR 결과가 없습니다.")
        return

    try:
        if preprocessed_img is not None:
            # 전처리된 이미지가 그레이스케일일 수 있으므로 RGB로 변환
            image_to_draw_on = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB) if len(preprocessed_img.shape) == 2 else preprocessed_img
            image = Image.fromarray(image_to_draw_on)
        else:
            image = Image.open(original_image_path).convert('RGB')

        boxes = [item['bounding_box'] for item in extracted_data]
        txts = [item['text'] for item in extracted_data]
        scores = [item['confidence'] for item in extracted_data]

        font_path = get_os_specific_font_path() 
        
        if not font_path: # OS 자동 감지 폰트 실패 시 로컬 폰트 시도
            local_font_path_korean = './fonts/malgun.ttf' 
            if os.path.exists(local_font_path_korean):
                font_path = local_font_path_korean
                if not get_os_specific_font_path.font_message_printed: # 아직 폰트 관련 메시지 안나갔으면
                    logger.info(f"로컬 한국어 폰트 사용 시도: {font_path}")
                    get_os_specific_font_path.font_message_printed = True  # 메시지 출력 후 플래그 설정
        
        if not font_path: # 모든 한국어 폰트 실패 시 영문 대체 로컬 폰트 시도
            local_font_path_english = './fonts/arial.ttf'
            if os.path.exists(local_font_path_english):
                font_path = local_font_path_english
                if not get_os_specific_font_path.font_message_printed:
                    logger.info(f"로컬 영문 대체 폰트 사용 시도: {font_path} (한글 표시 안될 수 있음)")
                    get_os_specific_font_path.font_message_printed = True
        
        if not display_ocr_result.final_font_choice_logged: # 최종 폰트 선택 로그 (한 번만)
            if font_path and os.path.exists(font_path):
                logger.info(f"텍스트 시각화에 사용할 최종 폰트: {font_path}")
            elif font_path: # 경로는 있으나 파일이 없는 경우
                logger.warning(f"최종 선택된 폰트 '{font_path}'를 찾을 수 없어 폰트 없이 시도합니다.")
                font_path = None # 명시적으로 None 설정
            else:
                logger.warning("사용 가능한 특정 폰트를 찾지 못해 PaddleOCR 내부 기본 폰트를 사용합니다.")
            display_ocr_result.final_font_choice_logged = True

        try:
            im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        except Exception as e:
            logger.error(f"draw_ocr 중 오류 (폰트: {font_path}, 파일: {original_filename}): {e}. 폰트 없이 재시도.")
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

if __name__ == "__main__":
    setup_logger() 
    _initialize_font_flags() # 스크립트 실행 시 폰트 로그 플래그 초기화

    parser = argparse.ArgumentParser(description="이미지 폴더에서 텍스트를 추출하는 OCR 스크립트입니다.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_dir", nargs='?', default='input', 
                        help="텍스트를 추출할 이미지가 포함된 폴더의 경로입니다.\n(기본값: 'input')")
    parser.add_argument("--output_dir", default='output', 
                        help="OCR 결과 이미지 및 텍스트 파일을 저장할 폴더 경로입니다.\n(기본값: 'output')")
    parser.add_argument("--lang", default='korean',
                        help="OCR에 사용할 언어입니다. 예: 'korean', 'en', 'japan', 'ch_sim'.\n"
                             "'korean+en'과 같이 복합 언어 지정 시, 기본 한국어 모델을 사용하며 영어도 일부 인식합니다.\n"
                             "정확한 다국어 처리는 PaddleOCR 문서를 참고하여 적절한 모델을 사용하세요.\n(기본값: 'korean')")
    parser.add_argument("--show_image", action='store_true', 
                        help="처리된 각 이미지와 OCR 결과를 화면에 표시합니다.")
    parser.add_argument("--no_preprocess", action='store_true', 
                        help="이미지 전처리 단계를 건너뜁니다.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}', 
                        help="스크립트 버전을 표시하고 종료합니다.")
    parser.add_argument('--debug', action='store_true', 
                        help="디버그 레벨 로깅을 활성화하여 더 상세한 로그를 출력합니다.")
    parser.add_argument('--use_gpu', action='store_true', 
                        help="사용 가능한 경우 GPU를 사용하여 OCR 처리를 시도합니다.\n(NVIDIA GPU 및 CUDA 환경 필요)")
    parser.add_argument('--save_text', action='store_true', 
                        help="추출된 텍스트를 이미지와 동일한 기반 이름의 .txt 파일로 저장합니다.")

    args = parser.parse_args()

    if args.debug:
        setup_logger(logging.DEBUG) 
        logger.debug("디버그 모드가 활성화되었습니다.")

    input_dir_path = args.input_dir
    output_dir_path = args.output_dir
    selected_language = args.lang
    display_images_flag = args.show_image
    skip_preprocessing = args.no_preprocess
    use_gpu_flag = args.use_gpu
    save_text_flag = args.save_text

    if not os.path.isdir(input_dir_path):
        logger.error(f"입력 디렉토리 '{input_dir_path}'를 찾을 수 없거나 디렉토리가 아닙니다. 스크립트를 종료합니다.")
        logger.info(f"팁: '{input_dir_path}' 디렉토리를 생성하거나 올바른 경로를 지정해주세요.")
        exit(1)

    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            logger.info(f"출력 디렉토리 생성: {output_dir_path}")
        except OSError as e:
            logger.error(f"출력 디렉토리 '{output_dir_path}'를 생성할 수 없습니다: {e}")
            exit(1)
    
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files_to_process = [
        filename for filename in os.listdir(input_dir_path)
        if filename.lower().endswith(supported_extensions)
    ]
    
    if not image_files_to_process:
        logger.warning(f"입력 디렉토리 '{input_dir_path}'에서 지원되는 이미지 파일을 찾을 수 없습니다.")
        logger.warning(f"지원 확장자: {supported_extensions}")
        exit(0)

    logger.info(f"OCR 스크립트 버전: {__version__}")
    logger.info(f"입력 디렉토리: {input_dir_path}")
    logger.info(f"출력 디렉토리: {output_dir_path}")
    logger.info(f"선택 언어: {selected_language}")
    logger.info(f"GPU 사용 시도: {use_gpu_flag}")
    logger.info(f"전처리 수행: {not skip_preprocessing}")
    logger.info(f"결과 이미지 표시: {display_images_flag}")
    logger.info(f"추출 텍스트 파일 저장: {save_text_flag}")
    logger.info(f"총 {len(image_files_to_process)}개의 이미지 파일을 처리합니다.")
    
    # PaddleOCR 엔진 초기화 (한 번만)
    try:
        logger.info(f"PaddleOCR 엔진 초기화 중... (언어: {selected_language}, GPU: {use_gpu_flag})")
        # PaddleOCR 초기화 시 전달할 파라미터 구성
        ocr_engine_params = {'use_angle_cls': True, 'lang': selected_language, 'use_gpu': use_gpu_flag, 'show_log': False}
        # 'korean+en'과 같은 복합 언어 처리 (PaddleOCR 자체에서 처리하도록 lang 값을 그대로 전달)
        if '+' in selected_language:
             logger.info(f"복합 언어 설정 '{selected_language}' 감지. PaddleOCR이 해당 설정을 지원하는지 확인하세요.")
        
        ocr_engine = PaddleOCR(**ocr_engine_params)
        logger.info("PaddleOCR 엔진 초기화 완료.")
    except Exception as e:
        logger.error(f"PaddleOCR 엔진 초기화 실패: {e}")
        logger.error("PaddleOCR 관련 라이브러리가 올바르게 설치되었는지, 모델 파일 다운로드에 문제가 없는지 확인하세요.")
        exit(1)

    # 루프 시작 전 폰트 관련 플래그 초기화
    _initialize_font_flags()


    for filename in tqdm(image_files_to_process, desc="OCR 처리 중", unit="개"):
        tqdm.set_postfix_str(f"파일: {filename}", refresh=True) # 현재 처리 중인 파일명 표시
        current_image_path = os.path.join(input_dir_path, filename)
        logger.debug(f"처리 시작: {filename}")

        ocr_input_data = current_image_path # 기본적으로 이미지 경로 사용

        if not skip_preprocessing:
            logger.debug(f"외부 전처리 시작: {filename}")
            processed_image_data = preprocess_image_for_ocr(current_image_path)
            if processed_image_data is not None:
                ocr_input_data = processed_image_data # 전처리된 NumPy 배열 사용
            else:
                logger.warning(f"{filename}: 외부 전처리에 실패하여 원본 이미지로 OCR을 시도합니다.")
        # else: # --no_preprocess 사용 시 별도 로그는 생략 (위에서 이미 로깅됨)
            # logger.info(f"{filename}: 전처리 단계를 건너뜁니다.") 

        # extract_text_from_image 함수는 이제 ocr_engine을 첫 번째 인자로 받음
        extracted_data = extract_text_from_image(ocr_engine, ocr_input_data, filename_for_log=filename)

        if extracted_data:
            logger.debug(f"추출된 텍스트 항목 수 ({filename}): {len(extracted_data)}")
            if logger.isEnabledFor(logging.DEBUG): # 디버그 레벨일 때만 모든 텍스트 출력
                for i, item in enumerate(extracted_data):
                     logger.debug(f"  - \"{item['text']}\" (신뢰도: {item['confidence']:.3f})")
            
            if save_text_flag:
                save_extracted_text(extracted_data, output_dir_path, filename)

            # 시각화 결과는 전처리된 이미지가 있다면 그것을, 없다면 원본 이미지를 사용
            # display_ocr_result 함수는 원본 이미지 경로도 필요로 함 (예: 파일명 생성)
            display_ocr_result(current_image_path, extracted_data, output_dir_path, filename,
                               preprocessed_img=processed_image_data if not skip_preprocessing and processed_image_data is not None else None, 
                               show_image_flag=display_images_flag)
        else:
            logger.info(f"{filename}에서 텍스트를 추출하지 못했습니다.")
        
    logger.info(f"총 {len(image_files_to_process)}개의 이미지 파일 처리가 완료되었습니다.")
    logger.info("스크립트 실행 종료.")
