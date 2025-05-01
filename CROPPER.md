# 자동 이미지 크롭 도구 (Composition Rules Based Auto Cropper)

Python으로 작성된 이 스크립트는 이미지 내 얼굴을 자동으로 감지하고, 3분할 법칙(Rule of Thirds) 또는 황금 비율(Golden Ratio)에 맞춰 이미지를 자동으로 크롭합니다. OpenCV의 DNN 기반 얼굴 검출 모델(YuNet)을 사용하여 피사체(얼굴)의 위치와 랜드마크(눈)를 파악하고, 이를 기반으로 구도를 재구성합니다.

## 주요 기능

* **DNN 기반 얼굴 감지**: OpenCV의 YuNet 모델을 사용하여 정확하고 다양한 각도의 얼굴 감지 수행
* **랜드마크 활용**: 얼굴 경계 상자 중심 또는 양 눈의 중심점을 구도 기준으로 선택 가능
* **자동 구도 크롭**: 3분할 법칙 및 황금 비율 규칙에 맞춰 최적의 크롭 영역 계산
* **주 피사체 선택**: 여러 얼굴 감지 시, 가장 큰 얼굴 또는 이미지 중앙에 가장 가까운 얼굴을 주 피사체로 선택 가능
* **비율 설정**: 원본 이미지 비율 유지 또는 특정 가로세로 비율(16:9, 1:1 등)로 크롭 가능
* **명령줄 인터페이스**: 사용하기 쉬운 명령줄 인수를 통해 다양한 옵션 설정 및 배치 처리 지원
* **파일/디렉토리 처리**: 단일 이미지 파일 또는 폴더 내 여러 이미지 일괄 처리 지원
* **모델 자동 다운로드**: 필요한 DNN 모델 파일(`face_detection_yunet_2023mar.onnx`) 자동 다운로드 기능 내장
* **진행률 표시**: 디렉토리 처리 시 `tqdm` 라이브러리가 설치된 경우 진행률 표시줄 제공

## 요구 사항

* **Python 3.8 이상**: 스크립트 및 최신 라이브러리와의 호환성을 위해 권장됩니다. (Python 설치 시 `pip`도 함께 설치되는 것이 일반적입니다.)
* **필요한 Python 라이브러리**:
    * `opencv-python`
    * `numpy`
    * `Pillow`
    * `tqdm` (선택 사항이지만, 디렉토리 처리 시 진행률 표시에 권장)

## 환경 구축

```bash
# Using pyenv to manage Python versions
git clone https://github.com/w3labkr/py-image-toolkit.git
cd py-image-toolkit
pyenv install 3.12.9
pyenv virtualenv 3.12.9 py-image-toolkit-3.12.9
pyenv local py-image-toolkit-3.12.9

# Install dependencies and freeze
pip install opencv-python numpy Pillow tqdm
pip freeze > requirements.txt
```

## 설치

1.  **스크립트 다운로드:**
    * 이 저장소를 클론하거나 `cropper.py` 스크립트 파일을 원하는 위치에 다운로드합니다.
    * **중요:** 스크립트 파일 저장 시 **UTF-8 인코딩**으로 저장해야 합니다 (코드 내 한글 주석/설명 포함). 대부분의 최신 텍스트 에디터는 기본적으로 UTF-8을 사용합니다.
    ```bash
    # 예시 (Git 사용 시)
    # git clone <repository_url>
    # cd <repository_directory>
    ```
2.  **라이브러리 설치:** 터미널(명령 프롬프트, PowerShell 등)을 열고 다음 명령어를 실행하여 필요한 라이브러리를 설치합니다.
    ```bash
    pip install opencv-python numpy Pillow tqdm
    ```
3.  **DNN 모델 파일 (`face_detection_yunet_2023mar.onnx`):**
    * 스크립트를 처음 실행할 때, 스크립트 파일이 있는 **동일한 디렉토리**에 모델 파일이 없으면 자동으로 다운로드를 시도합니다 (인터넷 연결 필요).
    * 자동 다운로드 실패 시, 스크립트 내 `YUNET_MODEL_URL` 주소에서 파일을 직접 다운로드 받아 `cropper.py` 스크립트와 **같은 폴더**에 저장해주세요.

## 사용법

스크립트는 터미널에서 실행합니다.

```bash
python cropper.py <입력_경로> [옵션들...]
```

**주요 인수 및 옵션:**

* `<입력_경로>`: (필수) 처리할 이미지 파일 경로 또는 이미지 파일이 있는 디렉토리 경로.
* `-o`, `--output_dir <폴더명>`: 결과 이미지를 저장할 디렉토리 (기본값: `output_final`).
* `-m`, `--method <방식>`: 주 피사체 선택 방법 (`largest` 또는 `center`, 기본값: `largest`).
    * `largest`: 가장 큰 얼굴
    * `center`: 이미지 중앙에 가장 가까운 얼굴
* `-ref`, `--reference <기준점>`: 구도 기준점 타입 (`eye` 또는 `box`, 기본값: `eye`).
    * `eye`: 양 눈의 중심
    * `box`: 얼굴 경계 상자의 중심
* `-r`, `--ratio <비율>`: 목표 크롭 비율 (예: `16:9`, `1.0`, `4:3`, `None`). `None`이거나 미지정 시 원본 비율 유지 (기본값: `None`).
* `-c`, `--confidence <값>`: 얼굴 감지 최소 신뢰도 임계값 (0~1 사이, 기본값: `0.6`).
* `-n`, `--nms <값>`: 얼굴 감지 NMS(Non-Maximum Suppression) 임계값 (겹치는 얼굴 영역 제거 기준, 기본값: `0.3`).
* `-h`, `--help`: 도움말 메시지를 표시합니다.
* `--version`: 스크립트 버전 정보를 표시합니다.

## 실행 예시

* **단일 이미지 처리 (기본 설정)**
    ```bash
    python cropper.py photo.jpg
    ```

* **단일 이미지 처리 (출력 폴더 지정, 16:9 비율)**
    ```bash
    python cropper.py input.png -o ./output -r 16:9
    ```

* **단일 이미지 처리 (바운딩 박스 중심 기준)**
    ```bash
    python cropper.py portrait.jpeg -ref box
    ```

* **폴더 내 모든 이미지 처리 (16:9 비율, 바운딩 박스 중심 기준)**
    ```bash
    python cropper.py ./input -o ./output -r 16:9 -ref box
    ```

* **폴더 내 모든 이미지 처리 (1:1 비율, 중앙 가까운 얼굴, 눈 기준)**
    ```bash
    python cropper.py ./input_folder -o ./output_square -r 1:1 -m center -ref eye
    ```
    *(참고: `tqdm` 라이브러리가 설치되어 있으면 처리 진행률이 표시됩니다.)*

## 출력

* 처리된 이미지는 지정된 출력 폴더(기본값: `output_final`)에 저장됩니다.
* **주의:** 입력 이미지에서 얼굴이 감지되지 않거나, 유효한 크롭 영역 계산에 실패하는 경우 해당 이미지에 대한 결과 파일은 생성되지 않습니다. (스크립트 실행 시 관련 경고 메시지가 출력될 수 있습니다.)
* 성공적으로 생성된 결과 파일 이름 형식: `원본파일명_{규칙}{비율정보}{기준점정보}.확장자`
    * 규칙: `_thirds` 또는 `_golden`
    * 비율정보: `_r16-9`, `_r1.0`, `_rOrig` 등 (`:` 는 `-` 로 대체됨)
    * 기준점정보: `_refEye` 또는 `_refBox`
    * 예시: `photo_thirds_r16-9_refEye.jpg`, `input_golden_rOrig_refBox.png`

## 라이선스 (License)

이 프로젝트는 특정 라이선스 하에 배포되지 않았습니다. 사용 시 라이선스 정책을 명확히 하고자 한다면, MIT 라이선스, Apache 2.0 라이선스 등을 고려하여 라이선스 파일(예: `LICENSE`)을 추가하고 아래에 명시하는 것을 권장합니다.

```
[라이선스 정보를 여기에 명시하세요. 예: This project is licensed under the MIT License - see the LICENSE file for details.]
```

## 향후 개선 사항 (TODO)

* 얼굴 외 다양한 객체(사람, 동물 등) 감지 지원 (예: YOLO 모델 통합)
* Saliency Map (시각적으로 두드러지는 영역)을 이용한 주요 영역 추정 기능 추가
* 더 정교한 랜드마크 활용 (예: 인물 사진 구도 최적화)
* Python `logging` 모듈을 사용한 로그 관리 개선 (로그 레벨 설정, 파일 로깅 등)
