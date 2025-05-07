# PaddleOCR을 사용하여 이미지에서 텍스트를 추출하는 스크립트 (OpenCV 전처리 및 폴더 처리 추가)
import cv2 # OpenCV 라이브러리 임포트
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np # NumPy 임포트 (OpenCV 이미지 처리에 사용)
import os # 파일 및 디렉토리 관리를 위해 os 모듈 임포트
import argparse # 명령줄 인자 처리를 위해 argparse 모듈 임포트

__version__ = "0.2.0" # 스크립트 버전 정보

def preprocess_image_for_ocr(image_path):
    """
    OCR 정확도 향상을 위해 이미지를 전처리합니다.
    (그레이스케일, 대비 향상, 노이즈 제거)

    Args:
        image_path (str): 전처리할 이미지 파일의 경로입니다.

    Returns:
        numpy.ndarray: 전처리된 이미지 (OpenCV Mat 형식)
                       오류 발생 시 None을 반환합니다.
    """
    try:
        # 1. 이미지 읽기
        img = cv2.imread(image_path)
        if img is None:
            print(f"오류: 이미지를 불러올 수 없습니다. 경로를 확인하세요: {image_path}")
            return None

        # 2. 그레이스케일 변환
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_img = clahe.apply(gray_img)

        # 4. 노이즈 제거 (미디언 블러)
        denoised_img = cv2.medianBlur(enhanced_contrast_img, 3)

        print(f"이미지 전처리 완료: {image_path}")
        return denoised_img

    except cv2.error as e:
        print(f"OpenCV 오류 발생 (이미지 전처리 중): {e}")
        return None
    except Exception as e:
        print(f"이미지 전처리 중 예외 발생: {e}")
        return None


def extract_text_from_image(image_path_or_data, lang='korean', preprocess=True):
    """
    주어진 이미지 경로 또는 데이터에서 텍스트를 추출합니다.

    Args:
        image_path_or_data (str or numpy.ndarray): 텍스트를 추출할 이미지 파일의 경로 또는 전처리된 이미지 데이터입니다.
        lang (str, optional): 인식할 언어입니다. 기본값은 'korean'입니다.
        preprocess (bool, optional): 이미지 전처리 단계 수행 여부. 기본값은 True입니다.

    Returns:
        list: 추출된 텍스트 정보 리스트. 각 항목은 딕셔너리 형태입니다.
              {'text': str, 'confidence': float, 'bounding_box': list}
              오류 발생 시 None을 반환합니다.
    """
    try:
        # PaddleOCR 초기화
        ocr_params = {'use_angle_cls': True, 'use_gpu': False, 'show_log': False}
        if lang == 'korean':
            ocr_params['lang'] = 'korean'
        elif lang == 'english':
            ocr_params['lang'] = 'en'
        elif lang == 'korean+en':
            ocr_params['lang'] = 'korean' # 기본 한국어 모델 사용. 영어도 어느 정도 인식 가능.
            print("알림: 'korean+en'의 경우, 기본 한국어 모델을 사용합니다. 영어 인식 성능은 모델에 따라 다를 수 있습니다.")
        else:
            print(f"지원하지 않는 언어 설정입니다: {lang}. 기본값으로 한국어를 사용합니다.")
            ocr_params['lang'] = 'korean'
        
        ocr = PaddleOCR(**ocr_params)

        input_data = image_path_or_data
        if isinstance(image_path_or_data, str) and preprocess:
            print(f"이미지 전처리를 시작합니다: {image_path_or_data}")
            processed_img = preprocess_image_for_ocr(image_path_or_data)
            if processed_img is None:
                print("이미지 전처리에 실패하여 OCR을 진행할 수 없습니다.")
                return None
            input_data = processed_img
        elif isinstance(image_path_or_data, str) and not preprocess:
            print(f"원본 이미지로 OCR을 수행합니다: {image_path_or_data}")
            pass 
        elif not isinstance(image_path_or_data, np.ndarray):
            print("잘못된 입력 데이터 타입입니다. 이미지 경로(str) 또는 NumPy 배열이어야 합니다.")
            return None

        result = ocr.ocr(input_data, cls=True)

        extracted_texts = []
        if result and result[0] is not None:
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

    except ImportError:
        print("PaddleOCR 또는 PaddlePaddle 라이브러리가 설치되지 않았습니다.")
        print("pip install paddlepaddle paddleocr opencv-python 명령어로 설치해주세요.")
        return None
    except Exception as e:
        print(f"OCR 처리 중 오류 발생: {e}")
        return None

def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename, 
                       preprocessed_img=None, show_image=False):
    """
    OCR 결과를 이미지 위에 표시하고 저장합니다.

    Args:
        original_image_path (str): 원본 이미지 파일 경로입니다.
        extracted_data (list): extract_text_from_image의 결과 (딕셔너리 리스트).
        output_dir (str): 결과 이미지를 저장할 폴더 경로입니다.
        original_filename (str): 원본 이미지의 파일명입니다. (출력 파일명 생성에 사용)
        preprocessed_img (numpy.ndarray, optional): 전처리된 이미지. 제공되면 이 이미지에 결과를 그립니다.
        show_image (bool, optional): True이면 결과 이미지를 화면에 표시합니다. 기본값은 False입니다.
    """
    if not extracted_data:
        print(f"표시할 OCR 결과가 없습니다: {original_filename}")
        return

    try:
        if preprocessed_img is not None:
            if len(preprocessed_img.shape) == 2: # 그레이스케일 이미지인 경우
                image_to_draw_on = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
            else: # 이미 컬러 이미지인 경우 (예: 전처리 단계에서 컬러 유지)
                image_to_draw_on = preprocessed_img
            image = Image.fromarray(image_to_draw_on)
        else:
            image = Image.open(original_image_path).convert('RGB') # 원본 이미지 로드

        boxes = [item['bounding_box'] for item in extracted_data]
        txts = [item['text'] for item in extracted_data]
        scores = [item['confidence'] for item in extracted_data]

        font_path = './fonts/malgun.ttf' 
        default_font_path = './fonts/arial.ttf'

        try:
            im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        except FileNotFoundError:
            print(f"경고: 주 폰트 파일 '{font_path}'를 찾을 수 없습니다. 대체 폰트를 사용합니다. (한글이 깨질 수 있음)")
            try:
                im_show = draw_ocr(image, boxes, txts, scores, font_path=default_font_path)
            except FileNotFoundError:
                print(f"경고: 대체 폰트 파일 '{default_font_path}'도 찾을 수 없습니다. 텍스트 없이 바운딩 박스만 표시될 수 있습니다.")
                im_show = draw_ocr(image, boxes, font_path=None)
            except Exception as font_ex:
                 print(f"대체 폰트 로딩 중 오류: {font_ex}")
                 im_show = draw_ocr(image, boxes, font_path=None)
        except Exception as e:
            print(f"draw_ocr 중 오류 발생: {e}. 폰트 없이 시도합니다.")
            im_show = draw_ocr(image, boxes, font_path=None)


        im_show_pil = Image.fromarray(im_show)

        base, ext = os.path.splitext(original_filename)
        output_image_filename = f"{base}_ocr{ext}"
        output_image_path = os.path.join(output_dir, output_image_filename)
        
        im_show_pil.save(output_image_path)
        print(f"OCR 결과가 표시된 이미지가 {output_image_path}에 저장되었습니다.")
        
        if show_image:
            im_show_pil.show()

    except FileNotFoundError:
        print(f"오류: 원본 이미지 파일 '{original_image_path}'를 찾을 수 없습니다 (결과 표시 중).")
    except Exception as e:
        print(f"OCR 결과 표시/저장 중 오류 발생 ({original_filename}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 폴더에서 텍스트를 추출하는 OCR 스크립트입니다.")
    parser.add_argument("input_dir", nargs='?', default=None, help="텍스트를 추출할 이미지가 포함된 폴더의 경로입니다. (생략 시 버전 정보만 표시)")
    parser.add_argument("--output_dir", help="OCR 결과 이미지를 저장할 폴더 경로입니다. (기본값: 입력 폴더 내 'output_ocr_results')", default=None)
    parser.add_argument("--lang", help="OCR에 사용할 언어입니다 (예: korean, en, korean+en). (기본값: korean)", default='korean')
    parser.add_argument("--show_image", action='store_true', help="처리된 각 이미지와 OCR 결과를 화면에 표시합니다.")
    parser.add_argument("--no_preprocess", action='store_true', help="이미지 전처리 단계를 건너뜁니다.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}', help="스크립트 버전을 표시하고 종료합니다.")

    args = parser.parse_args()

    # --version 인자가 사용되면, argparse가 자동으로 버전 정보를 출력하고 종료하므로
    # input_dir이 None인지 여부로 실제 OCR 작업을 수행할지 결정합니다.
    if args.input_dir is None:
        # input_dir이 제공되지 않았고, --version 인자가 사용되지 않았다면 사용법을 출력하거나,
        # 여기서는 --version이 우선적으로 처리되므로 이 블록은 일반적인 경우에 도달하지 않을 수 있습니다.
        # 만약 input_dir을 필수로 하고 싶다면 nargs='?'를 제거하고, 아래 로직을 조정해야 합니다.
        # 현재는 input_dir 없이 실행하면 (그리고 --version도 없다면) 에러 없이 종료됩니다.
        # 사용자가 input_dir 없이 실행했을 때 도움말을 보여주려면 parser.print_help()를 호출할 수 있습니다.
        if not any(arg.startswith('--version') or arg == '-v' for arg in os.sys.argv): # --version 외의 경우
             parser.print_help()
        exit(0)


    input_dir_path = args.input_dir
    output_dir_path = args.output_dir
    selected_language = args.lang
    display_images = args.show_image
    skip_preprocessing = args.no_preprocess

    if not os.path.isdir(input_dir_path):
        print(f"오류: 입력 디렉토리 '{input_dir_path}'를 찾을 수 없거나 디렉토리가 아닙니다.")
        exit(1)

    if output_dir_path is None:
        output_dir_path = os.path.join(input_dir_path, "output_ocr_results")

    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            print(f"출력 디렉토리 생성: {output_dir_path}")
        except OSError as e:
            print(f"오류: 출력 디렉토리 '{output_dir_path}'를 생성할 수 없습니다. {e}")
            exit(1)
    
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files_processed_count = 0

    print(f"\nOCR 스크립트 버전: {__version__}")
    print(f"입력 디렉토리: {input_dir_path}")
    print(f"출력 디렉토리: {output_dir_path}")
    print(f"선택 언어: {selected_language}")
    print(f"전처리 수행: {not skip_preprocessing}")
    print(f"결과 이미지 표시: {display_images}")
    print("-" * 30)

    for filename in os.listdir(input_dir_path):
        if filename.lower().endswith(supported_extensions):
            current_image_path = os.path.join(input_dir_path, filename)
            print(f"\n[처리 시작] 이미지: {filename}")

            processed_image_data = None
            ocr_input_data = current_image_path 

            if not skip_preprocessing:
                processed_image_data = preprocess_image_for_ocr(current_image_path)
                if processed_image_data is not None:
                    ocr_input_data = processed_image_data 
                else:
                    print(f"경고: {filename} 전처리에 실패했습니다. 원본 이미지로 OCR을 시도합니다.")
            else:
                print(f"{filename} 전처리 단계를 건너뜁니다.")

            extracted_data = extract_text_from_image(ocr_input_data, 
                                                     lang=selected_language, 
                                                     preprocess=False)

            if extracted_data:
                print(f"추출된 텍스트 항목 수 ({filename}): {len(extracted_data)}")
                for i, item in enumerate(extracted_data):
                    print(f"  {i+1}. 텍스트: \"{item['text']}\", 신뢰도: {item['confidence']:.2f}")
                
                display_ocr_result(current_image_path, extracted_data, output_dir_path, filename,
                                   preprocessed_img=processed_image_data if not skip_preprocessing else None, 
                                   show_image=display_images)
            else:
                print(f"{filename}에서 텍스트를 추출하지 못했습니다.")
            
            image_files_processed_count += 1
            print("-" * 30)

    if image_files_processed_count == 0:
        print(f"입력 디렉토리 '{input_dir_path}'에서 지원되는 이미지 파일을 찾을 수 없습니다.")
        print(f"지원 확장자: {supported_extensions}")
    else:
        print(f"\n총 {image_files_processed_count}개의 이미지 파일 처리가 완료되었습니다.")

    print("\n스크립트 실행 종료.")
    print("참고: OCR 결과 시각화 시 폰트 경로('./fonts/malgun.ttf' 또는 시스템 폰트 경로)가 올바른지 확인하세요.")
    print("      한글 폰트가 없거나 경로가 잘못되면 시각화된 이미지의 글자가 깨질 수 있습니다.")
