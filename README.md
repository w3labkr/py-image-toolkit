# py-image-toolkit

`py-image-toolkit`은 Python으로 작성된 이미지 처리 도구 모음입니다. 이 도구는 이미지 크기 조정 및 얼굴 감지를 기반으로 한 자동 크롭 기능을 제공합니다. 명령줄 인터페이스(CLI)를 통해 사용이 간편하며, 다양한 옵션을 지원합니다.

---

## 프로젝트 목적

이미지 처리 작업은 다양한 분야에서 필수적입니다. `py-image-toolkit`은 다음과 같은 작업을 간소화하기 위해 설계되었습니다:

- 대량의 이미지를 일괄적으로 크기 조정.
- 얼굴 감지 및 구도 기반 크롭을 통해 사진의 품질을 향상.
- 간단한 명령줄 인터페이스로 빠르고 효율적인 이미지 처리.

---

## 주요 기능

### 1. 이미지 크기 조정 (`resizer.py`)

- **비율 유지 리사이즈**: 가로세로 비율을 유지하며 크기를 조정.
- **고정 크기 리사이즈**: 지정된 크기로 강제 조정.
- **포맷 변환**: PNG, JPG, WEBP 등 다양한 포맷으로 변환.
- **EXIF 메타데이터 유지**: 이미지의 EXIF 데이터를 보존.
- **하위 폴더 포함 처리**: 디렉토리 내 모든 이미지를 재귀적으로 처리.

### 2. 얼굴 감지 및 자동 크롭 (`cropper.py`)

- **DNN 기반 얼굴 감지**: OpenCV의 YuNet 모델을 사용하여 얼굴 감지.
- **구도 기반 크롭**: 3분할 법칙(Rule of Thirds) 또는 황금 비율(Golden Ratio)에 따라 이미지를 크롭.
- **주 피사체 선택**: 가장 큰 얼굴 또는 이미지 중앙에 가까운 얼굴을 기준으로 크롭.
- **비율 설정**: 원본 비율 유지 또는 특정 비율(예: 16:9, 1:1 등)로 크롭.
- **병렬 처리**: 다중 프로세스를 활용한 빠른 처리.

---

## 설치

### 요구 사항

- Python 3.8 이상
- 필수 라이브러리:
  - `opencv-python`
  - `opencv-contrib-python` (YuNet 모델 사용을 위해 필요)
  - `numpy`
  - `Pillow`
  - `tqdm`
  - `piexif` (리사이즈 기능에 필요)

### 설치 방법

1. 저장소를 클론하거나 스크립트를 다운로드합니다.
   ```bash
   git clone https://github.com/w3labkr/py-image-toolkit.git
   cd py-image-toolkit
   pyenv install 3.12.9
   pyenv virtualenv 3.12.9 py-image-toolkit-3.12.9
   pyenv local py-image-toolkit-3.12.9
   ```
2. 필요한 라이브러리를 설치합니다.
   ```bash
   pip install opencv-python opencv-contrib-python numpy Pillow tqdm piexif
   ```

---

## 사용법

### 1. 이미지 크기 조정 (`resizer.py`)

#### 기본 사용법

```bash
python resizer.py [input_directory] -m <mode> -O <format> [options]
```

#### 주요 옵션

- `-m`, `--resize-mode`: 리사이즈 방식 (`aspect_ratio`, `fixed`, `none`).
- `-O`, `--output-format`: 출력 파일 형식 (`original`, `png`, `jpg`, `webp`).
- `-w`, `--width`: 리사이즈 너비 (픽셀).
- `-H`, `--height`: 리사이즈 높이 (픽셀).
- `-f`, `--filter`: 리사이즈 필터 (`lanczos`, `bicubic`, `bilinear`, `nearest`).
- `-o`, `--output-dir`: 결과 저장 폴더 경로.
- `-r`, `--recursive`: 하위 폴더 포함 처리.

#### 예제

1. **비율 유지 리사이즈**
   ```bash
   python resizer.py ./input -o ./output -m aspect_ratio -f nearest -w 1280 -O jpg
   ```
   - 입력 폴더 내 이미지를 가로 1280px, 세로 720px 비율로 조정 후 JPG로 저장.

2. **고정 크기 리사이즈**
   ```bash
   python resizer.py ./input -o ./output -m fixed -w 720 -H 600 -f bicubic -O png
   ```
   - 입력 폴더 내 이미지를 720x600 크기로 강제 조정 후 PNG로 저장.

---

### 2. 얼굴 감지 및 자동 크롭 (`cropper.py`)

#### 기본 사용법

```bash
python cropper.py <input_path> [options]
```

#### 주요 옵션

- `-o`, `--output_dir`: 결과 저장 디렉토리.
- `-m`, `--method`: 주 피사체 선택 방법 (`largest`, `center`).
- `--ref`, `--reference`: 구도 기준점 (`eye`, `box`).
- `-r`, `--ratio`: 목표 크롭 비율 (예: `16:9`, `1.0`).
- `--rule`: 구도 규칙 (`thirds`, `golden`, `both`).
- `-p`, `--padding-percent`: 크롭 영역 주변 패딩 비율 (%).
- `--dry-run`: 실제 파일 저장 없이 처리 과정 시뮬레이션.

#### 예제

1. **단일 이미지 크롭**
   ```bash
   python cropper.py ./input/image.jpg -o ./output -r 16:9 --rule thirds
   ```
   - 입력 이미지를 16:9 비율로 3분할 법칙에 따라 크롭 후 저장.

2. **폴더 내 모든 이미지 크롭**
   ```bash
   python cropper.py ./input -o ./output -r 1:1 -m center --ref box --rule both
   ```
   - 입력 폴더 내 모든 이미지를 1:1 비율로 크롭하며, 3분할 법칙과 황금 비율을 모두 적용.

---

## 결과 예제

### 리사이즈된 이미지

- 입력: `image.jpg` (1920x1080)
- 출력: `image_resized.jpg` (1280x720)

### 크롭된 이미지

- 입력: `portrait.jpg` (3000x2000)
- 출력:
  - `portrait_thirds_r16-9_refEye.jpg`
  - `portrait_golden_r16-9_refEye.jpg`

---

## 프로젝트 구조

```
py-image-toolkit/
├── resizer.py       # 이미지 크기 조정 스크립트
├── cropper.py       # 얼굴 감지 및 자동 크롭 스크립트
├── README.md        # 프로젝트 설명 파일
└── requirements.txt # 필수 라이브러리 목록 (선택적)
```

---

## 에러 처리 및 디버깅

- **필요한 라이브러리가 설치되지 않은 경우**:
  - `pip install` 명령어로 누락된 라이브러리를 설치하세요.
- **DNN 모델 파일이 없는 경우**:
  - YuNet 모델 파일을 다운로드하거나 cropper.py 실행 시 자동으로 다운로드됩니다.
- **이미지가 처리되지 않는 경우**:
  - 입력 이미지의 경로와 확장자를 확인하세요.
  - 로그를 확인하려면 `--verbose` 옵션을 사용하세요.

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
