# -*- coding: utf-8 -*-
import os
import sys
import importlib.util # 라이브러리 확인용
import argparse # 커맨드 라인 인자 처리용

# --- 라이브러리 확인 및 로드 ---
REQUIRED_LIBS = {
    "Pillow": "PIL",
    "piexif": "piexif" # EXIF 처리용
}

missing_libs = []
for package_name, import_name in REQUIRED_LIBS.items():
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        missing_libs.append(package_name)

if missing_libs:
    print("(!) 오류: 스크립트 실행에 필요한 라이브러리가 설치되어 있지 않습니다.")
    print("    아래 명령어를 사용하여 설치해주세요:")
    for lib in missing_libs:
        print(f"    pip install {lib}")
    sys.exit(1)

# 라이브러리 로드
from PIL import Image, UnidentifiedImageError
import piexif # EXIF 처리

# --- 상수 정의 ---
SCRIPT_VERSION = "1.8" # 버전 업데이트 (위치 인수 입력)

# Pillow 버전 호환성을 위한 리샘플링 필터 정의
try:
    # Pillow 9.1.0 이상
    RESAMPLE_FILTERS = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST
    }
    FILTER_NAMES = {
        "lanczos": "LANCZOS (고품질)",
        "bicubic": "BICUBIC (중간 품질)",
        "bilinear": "BILINEAR (낮은 품질)",
        "nearest": "NEAREST (최저 품질)"
    }
except AttributeError:
    # 이전 Pillow 버전 호환성
    RESAMPLE_FILTERS = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST
    }
    FILTER_NAMES = {
        "lanczos": "LANCZOS (고품질)",
        "bicubic": "BICUBIC (중간 품질)",
        "bilinear": "BILINEAR (낮은 품질)",
        "nearest": "NEAREST (최저 품질)"
    }

# 지원하는 출력 포맷 정의 (argparse에서 사용할 이름)
SUPPORTED_OUTPUT_FORMATS = {
    "original": "원본 유지",
    "png": "PNG",
    "jpg": "JPG",
    "webp": "WEBP",
}

# 지원하는 입력 이미지 확장자
SUPPORTED_INPUT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

# --- 이미지 처리 핵심 함수 (이전과 동일) ---

def resize_image_maintain_aspect_ratio(img, max_width, max_height, resample_filter):
    """ 가로세로 비율을 유지하며 이미지 크기를 조절합니다. """
    original_width, original_height = img.size
    if original_width <= 0 or original_height <= 0: return img # 너비 또는 높이가 0 이하인 경우 처리 방지
    ratio = min(max_width / original_width, max_height / original_height)
    # 크기가 커지는 것을 방지하려면 ratio > 1 인 경우 1로 설정
    # ratio = min(ratio, 1.0) # 필요하다면 이 주석 해제 (원본보다 크게 만들지 않음)
    new_width = max(1, int(original_width * ratio))
    new_height = max(1, int(original_height * ratio))
    # 크기 변경이 없는 경우 원본 반환
    if (new_width, new_height) == (original_width, original_height): return img
    try:
        return img.resize((new_width, new_height), resample_filter)
    except ValueError as e:
        print(f"   -> 경고: 리사이즈 중 오류 발생 (({original_width},{original_height}) -> ({new_width},{new_height})): {e}. 원본 이미지를 사용합니다.")
        return img

def resize_image_fixed_size(img, target_width, target_height, resample_filter):
    """ 지정된 크기로 이미지 크기를 강제 조절합니다. """
    original_width, original_height = img.size
    if (target_width, target_height) == (original_width, original_height): return img
    if target_width <= 0 or target_height <= 0:
        print(f"   -> 경고: 유효하지 않은 고정 크기 ({target_width}x{target_height})가 지정되었습니다. 원본 이미지를 사용합니다.")
        return img
    try:
        return img.resize((target_width, target_height), resample_filter)
    except ValueError as e:
        print(f"   -> 경고: 리사이즈 중 오류 발생 (({original_width},{original_height}) -> ({target_width},{target_height})): {e}. 원본 이미지를 사용합니다.")
        return img


def get_unique_filepath(filepath):
    """ 파일 경로가 이미 존재하면 고유한 이름(숫자 추가)을 반환합니다. """
    if not os.path.exists(filepath): return filepath
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(new_filepath):
            print(f"   -> 정보: '{filename}' 파일이 이미 존재하여 새 이름 '{new_filename}'으로 저장합니다.")
            return new_filepath
        counter += 1

def prepare_image_for_save(img, output_format_str):
    """ 지정된 출력 형식(문자열)에 맞게 이미지 모드를 변환합니다 (특히 JPG). """
    save_img = img
    # output_format_str 은 'JPG', 'PNG' 등의 문자열
    if output_format_str == 'JPG':
        if img.mode in ('RGBA', 'LA', 'P'): # 'L' 모드는 RGB로 변환 필요 없음 (저장은 가능)
            # 투명도가 있는 P 모드는 RGBA로 먼저 변환
            if img.mode == 'P' and 'transparency' in img.info:
                print(f"   -> 정보: 이미지 모드(P with transparency)를 JPG 저장을 위해 RGBA로 변환합니다.")
                save_img = img.convert('RGBA') # 이후 RGBA 처리 로직으로 넘어감
            elif img.mode == 'P': # 투명도 없는 P 모드
                print(f"   -> 정보: 이미지 모드(P)를 JPG 저장을 위해 RGB로 변환합니다.")
                save_img = img.convert('RGB')
            elif save_img.mode in ('RGBA', 'LA'): # RGBA 또는 LA 모드 처리
                print(f"   -> 정보: 이미지 모드({img.mode})를 JPG 저장을 위해 RGB로 변환합니다 (투명도 손실).")
                # 흰색 배경 생성
                background = Image.new("RGB", save_img.size, (255, 255, 255))
                try:
                    # 알파 채널을 마스크로 사용하여 배경 위에 붙여넣기
                    mask = save_img.split()[-1]
                    background.paste(save_img, mask=mask)
                    save_img = background
                except (IndexError, ValueError): # 알파 채널이 없거나 분리할 수 없는 경우 (LA 등)
                    save_img = save_img.convert('RGB') # 그냥 RGB로 변환

        # 'L' 모드(그레이스케일)는 JPG 저장 시 자동으로 처리되지만, 명시적으로 변환 원하면 아래 주석 해제
        # elif img.mode == 'L':
        #     print(f"   -> 정보: 이미지 모드(L)를 JPG 저장을 위해 RGB로 변환합니다 (컬러 정보 없음).")
        #     save_img = img.convert('RGB')

    elif output_format_str == 'WEBP':
         # WebP는 RGBA, RGB, L 모드를 지원함. P 모드는 변환 필요할 수 있음.
         if img.mode == 'P':
              # 투명도 유무에 따라 RGB 또는 RGBA로 변환
              if 'transparency' in img.info:
                   print(f"   -> 정보: 이미지 모드(P with transparency)를 WEBP 저장을 위해 RGBA로 변환합니다.")
                   save_img = img.convert("RGBA")
              else:
                   print(f"   -> 정보: 이미지 모드(P)를 WEBP 저장을 위해 RGB로 변환합니다.")
                   save_img = img.convert("RGB")

    return save_img


def process_images(input_folder, output_folder, resize_options, output_format_options, process_recursive, preserve_exif):
    """ 지정된 폴더(및 하위 폴더)의 이미지들을 처리하고 결과를 요약합니다. """
    processed_count = 0
    error_count = 0
    skipped_files = []
    error_files = [] # 오류 발생 파일 목록 (파일명, 오류 메시지)
    absolute_output_folder = os.path.abspath(output_folder) # 절대 경로 사용

    # 처리할 파일 목록 생성
    files_to_process = [] # (input_path, relative_path) 튜플 저장
    try:
        if process_recursive:
            print(f"\n하위 폴더 포함하여 '{input_folder}' 탐색 중...")
            for root, dirs, files in os.walk(input_folder):
                # 출력 폴더 자체는 탐색에서 제외 (무한 루프 방지)
                if os.path.abspath(root).startswith(absolute_output_folder):
                    # 출력 폴더 하위 디렉토리도 건너뛰도록 dirs 리스트 수정
                    dirs[:] = [] # 현재 root의 하위 디렉토리는 더 이상 탐색하지 않음
                    continue
                for filename in files:
                    if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                        input_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(input_path, input_folder)
                        files_to_process.append((input_path, relative_path))
                    else:
                         relative_path = os.path.relpath(os.path.join(root, filename), input_folder)
                         skipped_files.append(relative_path + " (미지원 형식)")
        else:
            print(f"\n'{input_folder}' 폴더 탐색 중...")
            for filename in os.listdir(input_folder):
                input_path = os.path.join(input_folder, filename)
                if os.path.isfile(input_path):
                    if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                        files_to_process.append((input_path, filename))
                    else:
                        skipped_files.append(filename + " (미지원 형식)")
                elif os.path.isdir(input_path):
                    # 출력 폴더가 입력 폴더 바로 아래에 있을 경우 건너뛰기 목록에 추가
                    if os.path.abspath(input_path) == absolute_output_folder:
                         skipped_files.append(filename + " (출력 폴더)")
                    else:
                         skipped_files.append(filename + " (폴더)")

        total_files = len(files_to_process)
        if total_files == 0:
             print("(!) 처리할 이미지 파일이 지정된 경로에 없습니다.")
             return 0, 0, [], [] # 처리 결과 반환
    except Exception as e:
        print(f"(!) 치명적 오류: 입력 경로 '{input_folder}' 접근 실패. ({e})")
        return 0, 0, [], [] # 처리 결과 반환

    print(f"\n--- 총 {total_files}개의 이미지 처리 시작 ---")

    # 파일 처리 루프
    for i, (input_path, relative_path) in enumerate(files_to_process):
        progress = f"({i+1}/{total_files})"
        filename = os.path.basename(input_path)

        # 출력 경로 설정 (하위 폴더 구조 유지)
        output_relative_dir = os.path.dirname(relative_path)
        output_dir_for_file = os.path.join(absolute_output_folder, output_relative_dir)
        if not os.path.exists(output_dir_for_file):
            try:
                os.makedirs(output_dir_for_file)
            except OSError as e:
                error_msg = f"출력 하위 폴더 생성 실패: {output_dir_for_file} ({e})"
                print(f" {progress} ✗ 오류: {error_msg}")
                error_files.append((relative_path, error_msg))
                error_count += 1
                continue

        base_name, original_ext = os.path.splitext(filename)
        # output_format_options['format_str'] 은 'original', 'png', 'jpg', 'webp'
        output_format_str = output_format_options['format_str']
        output_ext = ""

        if output_format_str == 'original': output_ext = original_ext
        elif output_format_str == 'jpg': output_ext = '.jpg'
        elif output_format_str == 'webp': output_ext = '.webp'
        elif output_format_str == 'png': output_ext = '.png'
        else: output_ext = original_ext # 안전 장치

        output_filename = base_name + output_ext
        output_path_base = os.path.join(output_dir_for_file, output_filename)
        output_path = get_unique_filepath(output_path_base)

        exif_data = None

        try:
            with Image.open(input_path) as img:
                # 원본 이미지 정보 로깅 (디버깅에 유용)
                # print(f"   -> 정보: '{filename}' 로드됨. 모드: {img.mode}, 크기: {img.size}")

                # EXIF 데이터 로드
                original_exif_bytes = None
                if preserve_exif and 'exif' in img.info and img.info['exif']:
                    original_exif_bytes = img.info['exif']
                    try:
                        # piexif는 bytes 형태의 EXIF 데이터를 처리
                        exif_data = piexif.load(original_exif_bytes)
                        # print(f"   -> 정보: '{filename}'에서 EXIF 데이터 로드 성공.")
                    except Exception as exif_err:
                        # piexif.InvalidImageDataError 등 다양한 오류 가능
                        print(f"   -> 경고: '{filename}' EXIF 데이터 로드/파싱 실패. 건너<0xEB><0x84>. ({type(exif_err).__name__})")
                        exif_data = None # 오류 시 EXIF 데이터 없음으로 처리
                        original_exif_bytes = None # 파싱 실패 시 원본 바이트도 사용 안 함

                # 출력 형식에 맞게 이미지 준비 (모드 변환 등)
                # prepare_image_for_save 함수는 대문자 형식 문자열('JPG', 'PNG' 등)을 기대
                save_format_upper = output_format_str.upper() if output_format_str != 'original' else None
                img_prepared = prepare_image_for_save(img, save_format_upper)

                # 리사이즈
                resample_filter = resize_options.get('filter_obj') # mode 'none'일 때 키가 없을 수 있으므로 .get 사용
                img_resized = img_prepared # 리사이즈 안 할 수도 있으므로 초기화
                if resize_options['mode'] == 'aspect_ratio':
                    img_resized = resize_image_maintain_aspect_ratio(
                        img_prepared, resize_options['width'], resize_options['height'], resample_filter
                    )
                elif resize_options['mode'] == 'fixed':
                    img_resized = resize_image_fixed_size(
                        img_prepared, resize_options['width'], resize_options['height'], resample_filter
                    )
                else: # resize_options['mode'] == 'none' 또는 기타 (리사이즈 안 함)
                    pass # img_prepared 그대로 사용

                # 저장 옵션 설정
                save_kwargs = {}
                # Pillow save() 함수의 format 인자는 대문자 형식을 선호 (예: 'JPEG', 'PNG', 'WEBP')
                save_format_arg = None
                if output_format_str != 'original':
                    save_format_arg = save_format_upper
                    # JPG는 JPEG로 지정
                    if save_format_arg == 'JPG':
                         save_format_arg = 'JPEG'

                # EXIF 데이터 처리 (piexif.dump 사용)
                final_exif_bytes = None
                if preserve_exif and exif_data:
                    try:
                        # Orientation 정보는 리사이즈/회전 시 Pillow가 처리하므로 제거 시도
                        if piexif.ImageIFD.Orientation in exif_data.get('0th', {}):
                            # print(f"   -> 정보: '{filename}' 저장 전 EXIF Orientation 태그 제거 시도.")
                            exif_data['0th'][piexif.ImageIFD.Orientation] = 1 # 1 = Horizontal (normal)
                        # 썸네일 정보도 보통 불필요하므로 제거 시도 (용량 감소)
                        if 'thumbnail' in exif_data and exif_data['thumbnail']:
                             # print(f"   -> 정보: '{filename}' 저장 전 EXIF 썸네일 데이터 제거.")
                             exif_data['thumbnail'] = None # 또는 del exif_data['thumbnail']

                        final_exif_bytes = piexif.dump(exif_data)
                    except Exception as dump_err:
                        print(f"   -> 경고: '{filename}' EXIF 데이터 dump 실패. EXIF 없이 저장. ({dump_err})")
                        final_exif_bytes = None
                elif preserve_exif and original_exif_bytes:
                     # piexif 로드 실패했지만 원본 EXIF 바이트가 있는 경우, 그대로 사용 시도 (위험 감수)
                     # print(f"   -> 정보: '{filename}' 파싱 실패한 원본 EXIF 바이트를 저장 시도.")
                     final_exif_bytes = original_exif_bytes


                # 포맷별 저장 옵션 및 EXIF 적용
                if output_format_str == 'jpg':
                    save_kwargs['quality'] = output_format_options.get('quality', 95)
                    save_kwargs['optimize'] = True
                    save_kwargs['progressive'] = True
                    if final_exif_bytes:
                         save_kwargs['exif'] = final_exif_bytes
                elif output_format_str == 'png':
                    save_kwargs['optimize'] = True
                    # PNG는 Pillow 9.1.0+에서 'exif' 키워드 인자로 EXIF 지원
                    if final_exif_bytes:
                         try:
                              # Pillow 최신 버전은 'exif' 인자 지원
                              save_kwargs['exif'] = final_exif_bytes
                              # print(f"   -> 정보: '{filename}' PNG 저장 시 EXIF 데이터 포함 (Pillow>=9.1.0).")
                         except TypeError: # 이전 버전 호환성
                              print(f"   -> 경고: '{filename}' 사용 중인 Pillow 버전에서 PNG EXIF 직접 저장을 지원하지 않을 수 있습니다.")
                elif output_format_str == 'webp':
                    save_kwargs['quality'] = output_format_options.get('quality', 80)
                    save_kwargs['lossless'] = False # 기본값은 손실 압축
                    # WebP EXIF 지원 (Pillow 7.1.0+)
                    if final_exif_bytes:
                         save_kwargs['exif'] = final_exif_bytes

                # 원본 형식 유지 시 EXIF 처리
                elif output_format_str == 'original' and final_exif_bytes:
                    # 원본 형식이 EXIF를 지원하는 경우에만 적용 시도
                    # Pillow는 일부 포맷에서 EXIF 저장을 지원 (JPEG, TIFF, PNG(최신), WEBP(최신))
                    # 원본이 GIF, BMP 등인 경우 EXIF 저장 안될 수 있음
                    if original_ext.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.png', '.webp']:
                         save_kwargs['exif'] = final_exif_bytes
                    else:
                         print(f"   -> 정보: '{filename}' 원본 형식({original_ext})은 EXIF 저장을 지원하지 않을 수 있습니다.")


                # 결과 이미지 저장
                # format 인자가 None이면 Pillow가 파일 확장자로부터 추측
                img_resized.save(output_path, format=save_format_arg, **save_kwargs)
                print(f" {progress} ✓ '{relative_path}' 처리 완료 -> '{os.path.relpath(output_path, absolute_output_folder)}'")
                processed_count += 1

        # --- 개별 파일 처리 오류 핸들링 ---
        except UnidentifiedImageError:
            error_msg = "유효하지 않거나 손상된 이미지 파일"
            print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
        except PermissionError:
            error_msg = "파일 읽기/쓰기 권한 부족"
            print(f" {progress} ✗ 오류: '{relative_path}' 또는 출력 경로 ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
        except OSError as e:
            error_msg = f"파일 시스템 오류 ({e})"
            print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
            if os.path.exists(output_path): # 실패 시 생성된 불완전 파일 삭제 시도
                  try: os.remove(output_path)
                  except OSError: pass
        except ValueError as e: # Pillow 내부 처리 오류 등
             error_msg = f"이미지 처리 값 오류 ({e})"
             print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
             error_files.append((relative_path, error_msg))
             error_count += 1
        except Exception as e:
            import traceback
            error_msg = f"예상치 못한 오류 ({type(e).__name__}: {e})"
            print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
            # traceback.print_exc() # 상세 디버깅 필요시 주석 해제
            error_files.append((relative_path, error_msg))
            error_count += 1


    # --- 최종 결과 요약 반환 ---
    return processed_count, error_count, error_files, skipped_files


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"이미지 크기 일괄 변경 스크립트 (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawTextHelpFormatter, # help 메시지 줄바꿈 유지
        usage="%(prog)s [input_directory] -m <mode> -O <format> [options]" # 사용법 예시 변경
    )

    # --- 인수 정의 ---
    # 위치 인수 (입력 폴더)
    parser.add_argument("input_directory", nargs='?', default='.',
                        help="이미지가 있는 원본 폴더 경로 (기본값: 현재 디렉토리)")

    # 필수 옵션 그룹
    required_group = parser.add_argument_group('필수 옵션')
    required_group.add_argument("-m", "--resize-mode", required=True, choices=['aspect_ratio', 'fixed', 'none'],
                                help="리사이즈 방식:\n"
                                     "  aspect_ratio: 가로세로 비율 유지 (최대 너비/높이 내 맞춤)\n"
                                     "  fixed: 지정된 크기로 강제 변경 (비율 왜곡 가능)\n"
                                     "  none: 리사이즈 안 함 (포맷 변경, EXIF 처리 등만 수행)")
    required_group.add_argument("-O", "--output-format", required=True, choices=SUPPORTED_OUTPUT_FORMATS.keys(),
                                help="저장할 파일 형식:\n" +
                                     "\n".join([f"  {k}: {v}" for k, v in SUPPORTED_OUTPUT_FORMATS.items()]))

    # 출력 경로 옵션
    parser.add_argument("-o", "--output-dir",
                        help="결과 저장 폴더 경로 (기본값: 입력 폴더 하위의 'resized_images')")


    # 리사이즈 관련 그룹 (mode가 none이 아닐 때 필요)
    resize_group = parser.add_argument_group('리사이즈 관련 옵션 (mode가 none이 아닐 때)')
    resize_group.add_argument("-w", "--width", type=int, default=0, # 기본값을 0으로 하여 mode=none일 때 무시 가능하게 함
                              help="리사이즈 너비 (px). mode가 'aspect_ratio' 또는 'fixed'일 때 필수.")
    resize_group.add_argument("-H", "--height", type=int, default=0,
                              help="리사이즈 높이 (px). mode가 'aspect_ratio' 또는 'fixed'일 때 필수.")
    resize_group.add_argument("-f", "--filter", choices=FILTER_NAMES.keys(), default=None,
                              help="리사이즈 필터(품질/속도):\n" +
                                   "\n".join([f"  {k}: {v}" for k, v in FILTER_NAMES.items()]) +
                                   "\n(mode가 'aspect_ratio' 또는 'fixed'일 때 필수)")

    # 기타 선택 옵션 그룹
    optional_group = parser.add_argument_group('기타 선택 옵션')
    optional_group.add_argument("-r", "--recursive", action="store_true",
                                help="하위 폴더의 이미지도 포함하여 처리")
    optional_group.add_argument("-q", "--quality", type=int, default=None,
                                help="JPG 또는 WEBP 저장 품질 (1-100). 기본값: JPG=95, WEBP=80")
    optional_group.add_argument("-p", "--preserve-exif", action="store_true",
                                help="EXIF 메타데이터(촬영 정보 등) 유지 시도")
    optional_group.add_argument('--version', action='version', version=f'%(prog)s {SCRIPT_VERSION}')


    args = parser.parse_args()

    # --- 인자 값 검증 및 설정 ---

    # 입력 폴더 검증 (위치 인수 사용)
    if not os.path.isdir(args.input_directory):
        parser.error(f"입력 폴더 경로가 유효하지 않습니다: {args.input_directory}")
    absolute_input_dir = os.path.abspath(args.input_directory)

    # 리사이즈 모드에 따른 필수 인자 검증
    if args.resize_mode in ['aspect_ratio', 'fixed']:
        if args.width <= 0 or args.height <= 0:
            parser.error(f"--resize-mode가 '{args.resize_mode}'일 때는 --width (-w) 와 --height (-H)에 0보다 큰 정수를 지정해야 합니다.")
        if not args.filter:
            parser.error(f"--resize-mode가 '{args.resize_mode}'일 때는 --filter (-f)를 지정해야 합니다.")
        # 리사이즈 옵션 설정
        resize_opts = {
            'mode': args.resize_mode,
            'width': args.width,
            'height': args.height,
            'filter_str': args.filter,
            'filter_obj': RESAMPLE_FILTERS[args.filter]
        }
    else: # args.resize_mode == 'none'
        if args.width > 0 or args.height > 0 or args.filter:
             print("   -> 정보: --resize-mode가 'none'이므로 --width, --height, --filter 설정은 무시됩니다.")
        resize_opts = {'mode': 'none'} # 리사이즈 안 함을 명시


    # 출력 폴더 설정 및 검증
    if args.output_dir:
        absolute_output_dir = os.path.abspath(args.output_dir)
    else:
        # 입력 폴더 경로(절대 경로)를 기준으로 출력 폴더 이름 생성
        absolute_output_dir = os.path.join(absolute_input_dir, "resized_images")
        print(f"   -> 정보: 출력 폴더를 기본값으로 설정합니다: '{absolute_output_dir}'")

    # 입력/출력 폴더 충돌 검증
    if absolute_input_dir == absolute_output_dir:
        parser.error("입력 폴더와 출력 폴더는 동일할 수 없습니다.")
    # recursive 옵션과 출력 폴더가 입력 폴더 내부에 있는지 검사 강화
    try:
        common_path = os.path.commonpath([absolute_input_dir, absolute_output_dir])
        if args.recursive and common_path == absolute_input_dir and absolute_output_dir != absolute_input_dir:
            parser.error("하위 폴더 포함(--recursive) 처리 시, 출력 폴더는 입력 폴더 내부에 지정할 수 없습니다.")
    except ValueError: # 드라이브가 다른 경우 등 commonpath 계산 불가 시
        pass # 이 경우는 충돌 아님

    # 출력 폴더 생성 (미리 생성)
    try:
        if not os.path.exists(absolute_output_dir):
            os.makedirs(absolute_output_dir)
            print(f"   -> 정보: 출력 폴더를 생성했습니다: '{absolute_output_dir}'")
    except OSError as e:
        parser.error(f"출력 폴더를 생성할 수 없습니다: {absolute_output_dir} ({e})")

    # 출력 포맷 및 품질 설정
    output_format_opts = {
        'format_str': args.output_format
    }
    if args.output_format in ('jpg', 'webp'):
        default_quality = 95 if args.output_format == 'jpg' else 80
        quality = args.quality if args.quality is not None else default_quality
        if not (1 <= quality <= 100):
            parser.error(f"품질 값(--quality)은 1에서 100 사이여야 합니다 (입력값: {args.quality}).")
        output_format_opts['quality'] = quality
    elif args.quality is not None:
        print(f"   -> 경고: --quality (-q) 옵션은 'jpg' 또는 'webp' 형식에만 적용됩니다. 입력된 값({args.quality})은 무시됩니다.")


    # --- 최종 설정 확인 및 처리 시작 ---
    print("\n" + "="*30 + " 스크립트 설정 " + "="*30)
    print(f"입력 폴더: {absolute_input_dir}")
    print(f"출력 폴더: {absolute_output_dir}")
    print(f"하위 폴더 포함: {'예' if args.recursive else '아니오'}")
    print(f"리사이즈 방식: {args.resize_mode}")
    if args.resize_mode != 'none':
        print(f"  크기 설정: {resize_opts['width']}x{resize_opts['height']} px")
        print(f"  리사이즈 필터: {FILTER_NAMES[resize_opts['filter_str']]}")
    print(f"출력 형식: {SUPPORTED_OUTPUT_FORMATS[output_format_opts['format_str']]}")
    if 'quality' in output_format_opts: print(f"  품질: {output_format_opts['quality']}")
    print(f"EXIF 메타데이터 유지: {'예' if args.preserve_exif else '아니오'}")
    print("="*72)

    print("\n이미지 처리를 시작합니다...")
    processed_count, error_count, error_files, skipped_files = process_images(
        absolute_input_dir,
        absolute_output_dir,
        resize_opts,
        output_format_opts,
        args.recursive,
        args.preserve_exif
    )

    # --- 최종 결과 요약 ---
    print(f"\n--- 처리 결과 요약 ---")
    print(f"성공적으로 처리된 이미지 수: {processed_count}")
    print(f"오류 발생 건수: {error_count}")
    if error_files:
        print("\n[오류 발생 파일 목록]")
        # 오류가 너무 많으면 일부만 표시
        max_errors_to_show = 20
        for i, (filepath, errmsg) in enumerate(error_files):
            if i >= max_errors_to_show:
                print(f"  ... 외 {len(error_files) - max_errors_to_show}개 오류 생략")
                break
            print(f"  - {filepath}: {errmsg}")

    if skipped_files:
        print(f"\n건너<0xEB><0x8><0xB7> 파일/폴더 수: {len(skipped_files)}")
        limit = 10
        if len(skipped_files) <= limit:
             print("  건너<0xEB><0x8><0xB7> 목록:", ", ".join(skipped_files))
        else:
             print(f"  건너<0xEB><0x8><0xB7> 목록 (처음 {limit}개):", ", ".join(skipped_files[:limit]), "...")
    else:
        print("\n건너<0xEB><0x8><0xB7> 파일/폴더 없음")

    print("\n--- 모든 작업 완료 ---")
    print(f"결과는 '{absolute_output_dir}' 폴더에서 확인할 수 있습니다.")

    # 오류가 있었다면 0이 아닌 코드로 종료
    if error_count > 0:
        sys.exit(1)
