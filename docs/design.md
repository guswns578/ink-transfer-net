# 원작 선화 Ink Transfer 모델 설계 검토 요청

아래는 목표 웹툰의 선화 표현을 다른 선화에 구조적으로 안전하게 이전하기 위해 설계 중인 모델입니다. 일반적인 칭찬이나 개괄적 의견보다는, **실제로 구현·학습했을 때 실패할 가능성이 높은 지점과 더 단순하거나 타당한 대안**을 비판적으로 검토해 주세요.

## 1. 문제 배경

현재 SDXL Style LoRA를 img2img로 적용해 다른 작품 선화를 원작 선화 스타일로 변환하고 있습니다. 테스트에서 다음 문제가 확인됐습니다.

- denoise가 낮으면 선 표현이 거의 바뀌지 않음
- denoise를 높이면 선 굵기뿐 아니라 얼굴, 체형, 의상 구조가 변경됨
- ControlNet 또는 skeleton 전처리는 복잡한 옷 주름, 머리카락, 교차선, 해칭을 충분히 보존하지 못함
- 작은 얼굴은 Style LoRA 적용 후 눈·코·입 형태가 손상되는 경우가 있음
- 목표는 새로운 이미지를 생성하는 것이 아니라, 원래 선의 위치와 연결 구조를 유지하면서 원작 작가 특유의 얇은 선, 농도, 필압, 끝처리, 해칭 표현을 옮기는 것임

이에 디퓨전이 아닌 결정론적 line-to-line restoration/renderer 모델을 별도로 설계하고 있습니다. 임시 명칭은 `OIT-Net (Original-style Ink Transfer Network)`입니다.

## 2. 목표와 비목표

### 목표

- 입력과 출력의 폭·높이를 정확히 동일하게 유지
- 같은 입력과 설정에 항상 같은 출력
- 기존 선의 위치, 연결 관계, 얼굴과 의상 구조를 최대한 보존
- 선 굵기, 잉크 농도, 안티앨리어싱, 선 끝의 taper, 기존 해칭의 간격·농도를 원작 스타일로 변환
- 넓은 검은 면, 말풍선, 글자, 컷 테두리를 보호
- 전체 이미지를 latent로 압축하지 않고 픽셀 공간에서 처리

### 비목표

- 캐릭터 얼굴이나 인체를 새로 생성
- 원본에 없는 대규모 주름·머리카락·윤곽선을 자유롭게 추가
- 텍스트 프롬프트를 이용한 범용 이미지 생성
- Style LoRA 또는 확산모델을 OIT Core 내부에 포함

## 3. 보유 데이터와 환경

- 원작 작가의 선화 데이터: 약 6,600장
- 웹툰 컷 단위 PNG이며 해상도와 종횡비가 다양함
- 일부 컷에는 회색 톤, 넓은 검은 면, 말풍선, 효과음, 컷 테두리가 포함될 수 있음
- 타 작품 입력 선화와 목표 화풍으로 다시 그린 정답 사이의 실제 paired data는 거의 없음
- 학교 서버: NVIDIA L40S, VRAM 약 46GB
- 로컬 개발 환경: RTX 5070, VRAM 약 12GB
- 서버 학습 권장 초안: BF16, 1024 patch, batch 4, gradient accumulation 2

## 4. 전체 파이프라인

```text
원작 정답 선화 Y
    └─ 학습 시 실시간 합성 열화
       굵기 증가·회색선·블러·저해상도 계단 현상 등
                    ↓
              학습 입력 X

실사용 외부 선화 X ─────────────┐
                                ↓
배경 정규화 → Ink Alpha → 다중 임계 마스크
                                ↓
SDF20 / SDF50 / SDF80 + 방향장 + 검은 면 마스크
                                ↓
              OIT-Net 구조 보존 Core
                                ↓
ΔSDF / Opacity / Edge Softness / Preservation Gate
                                ↓
            미분 가능한 Ink Renderer
                                ↓
                    Core 선화 결과
                                ↓
           선택적 Artist Detail Head
                                ↓
                  최종 원작 선화
```

## 5. 선 위치·거리장 분석

입력 이미지를 grayscale로 변환하고 국소 배경 밝기를 추정해 `Ink Alpha`를 계산합니다.

```text
흰 배경       ≈ 0.0
회색 가장자리 ≈ 0.2~0.7
검은 잉크     ≈ 1.0
```

단일 이진화의 정보 손실을 줄이기 위해 세 임계값의 마스크를 만듭니다.

```text
M20 = Ink Alpha > 0.20   흐린 가장자리 포함
M50 = Ink Alpha > 0.50   주 선폭
M80 = Ink Alpha > 0.80   진한 잉크 중심
```

각 마스크에 signed Euclidean distance transform을 적용합니다.

```text
선 바깥: 양수
경계:     0
선 안쪽: 음수
```

해상도 차이는 다음 기준으로 정규화할 계획입니다.

```python
scale = sqrt(width * height / 2_000_000)
normalized_distance = pixel_distance / scale
```

해칭과 주름 보존을 위해 구조 텐서로부터 주 방향을 구하고 `sin(2θ)`, `cos(2θ)`, coherence를 추가합니다. 넓은 검은 영역은 내부 거리와 연결영역 크기를 이용해 `Solid Mask`로 분리합니다.

예상 입력은 8채널입니다.

```text
1. Ink Alpha
2. SDF20
3. SDF50
4. SDF80
5. Direction sin(2θ)
6. Direction cos(2θ)
7. Direction coherence
8. Solid-black mask
```

중요하게, skeleton은 모델 입력 전처리로 사용하지 않습니다. 필요할 경우 학습 중 연결성 손실 계산에만 보조적으로 사용합니다.

## 6. OIT-Net Core 구조

약 10~20M 파라미터의 NAFNet/U-Net 계열 완전 합성곱 복원 모델을 고려하고 있습니다.

### 고해상도 구조 경로

- 원본 해상도에서 NAF-style restoration block 수행
- 가는 눈선, 머리카락, 옷 주름의 좌표 정보를 skip connection으로 직접 전달
- 위치를 자유롭게 이동시키는 spatial transformer와 deformable convolution은 사용하지 않음

### 문맥 경로

- 1/2와 1/4 해상도까지만 축소
- 검은 면, 긴 옷 주름, 해칭 묶음과 교차점을 판단
- 1/8 이하 bottleneck은 사용하지 않는 초안

### 특징 융합

- 고해상도 좌표 특징과 문맥 특징을 원본 해상도에서 융합
- 완성 RGB 이미지를 직접 예측하지 않고 아래 네 가지 지도만 출력

## 7. 네 개의 출력 Head

### 7.1 ΔSDF Head

기준 거리장 `SDF50`에 더할 국소 거리 보정값을 예측합니다.

```python
delta = tanh(prediction) * allowed_change
allowed_change = min(2.5px, local_half_width * 0.45)
new_sdf = original_sdf + preservation_gate * delta
```

가는 선일수록 허용 이동량을 작게 하고, 굵은 선만 더 크게 얇아지거나 굵어지게 하려는 목적입니다.

### 7.2 Opacity Head

- 선 중심의 검정 농도
- 회색 잉크의 복원 강도
- 흰 배경으로 새는 잔여 픽셀 억제

### 7.3 Edge Softness Head

- 거리장에서 잉크로 다시 변환할 때 sigmoid의 온도/가장자리 폭을 국소적으로 제어
- 저해상도 확대에서 발생한 계단 현상을 줄이되 가는 선을 흐리지 않는 것이 목적

### 7.4 Preservation Gate

- 이미 올바른 가는 선, 눈 디테일, 검은 면은 원본을 통과
- 수정이 필요한 굵은 선과 자글거리는 경계에서만 높은 값

## 8. Ink Renderer

개념식은 다음과 같습니다.

```python
new_sdf = sdf50 + gate * delta_sdf
new_ink = sigmoid(-new_sdf / edge_softness) * opacity
output = 1.0 - new_ink
```

Solid Mask 내부는 원본을 보존하고 경계 띠만 모델이 수정합니다. 최종 출력은 흰 배경 위의 1채널 잉크 이미지이며 입력과 동일한 해상도로 저장합니다.

## 9. 선택적 Artist Detail Head

Core만으로는 기존 디테일의 표현은 바꿀 수 있지만, 원본에 없는 원작 작가 특유의 눈선이나 해칭을 결정해 추가하기 어렵다고 판단했습니다. 이를 위해 Core와 분리된 작은 sparse residual branch를 선택적으로 고려하고 있습니다.

대상:

- 눈꺼풀·눈꼬리·속눈썹 끝처리
- 홍채 내부선의 생략과 강조
- 머리카락 끝의 짧은 분기선
- 기존 명암·주름 부근의 해칭
- 옷 주름 주변의 제한된 잔선

안전 제한 초안:

- 기존 선 이동: 최대 2~3px
- 추가선은 얼굴·머리카락·주름·명암 후보 영역에 한정
- 추가 잉크 면적: 입력 잉크 면적의 3~8% 이내
- 신뢰도가 낮으면 잔차를 출력하지 않음
- Detail Head를 끄면 Core 결과만 반환

실제 paired data가 부족하기 때문에 원작 정답에서 속눈썹, 홍채선, 짧은 해칭, 머리카락 잔선 일부를 합성 제거하고 원본을 복원하는 방식으로 사전학습한 뒤, 사람이 보정한 실제 전후 패치 100~500개로 fine-tuning하는 방안을 생각하고 있습니다.

## 10. 데이터 전처리 계획

텍스트 태그와 자연어 캡션은 OIT Core에서 사용하지 않습니다. 대신 자동 분석 manifest와 보호 마스크를 만듭니다.

- 손상 파일·중복·저해상도 검사
- RGB/gray/alpha 정규화
- 흰 배경 정규화
- 검은 면 검출과 보호
- 말풍선·글자·효과음·컷 테두리 보호 또는 loss 제외
- 톤이 많은 컷은 V1에서 제외하거나 tone mask 적용
- 회차 단위 train/validation/test 분리
- 원본 파일은 변형하지 않고 합성 열화는 학습 중 메모리에서 생성

합성 열화 후보:

- 균일·불균일 dilation
- 영역별 선폭 변화
- 약한 blur
- 회색선과 낮은 대비
- 축소 후 nearest/bilinear/bicubic 확대
- JPEG/스캔 계열 노이즈
- 경계의 미세한 wobble

V1에서는 선 삭제와 새로운 선 추가를 합성 열화에 넣지 않을 계획입니다. 모델이 구조를 상상해서 복원하는 습관을 먼저 학습하지 않게 하기 위함입니다.

## 11. 손실 함수 초안

```text
L =
  1.00 × Charbonnier/L1 pixel restoration
+ 0.35 × MS-SSIM
+ 0.50 × bidirectional distance-transform alignment
+ 0.15 × soft-clDice/topology preservation
+ 0.10 × line-width distribution
+ 0.05 × residual magnitude penalty
```

- GAN loss와 VGG perceptual loss는 V1에서 제외
- 연결성 손실은 junction과 복잡한 해칭에서 오히려 잘못 작동할 가능성이 있어 낮은 가중치로 시작
- 필요하면 topology loss를 전체 이미지가 아니라 가는 선·junction 후보에만 적용

## 12. 학습 계획 초안

### Smoke test

- 데이터 300장
- 512 patch
- 약 10 epoch
- 목표: 위치 보존, loss 정상 감소, renderer 안정성 확인

### Core 본 학습

- 약 6,600장
- 768 또는 1024 patch
- effective batch 8
- 20~40 epoch
- 5/10/20/30/40 epoch에서 고정 검증 세트 출력
- validation 개선이 멈추면 조기 종료

### Detail Head

- Core를 우선 고정
- 얼굴·눈·머리카락·해칭·옷 주름 패치를 가중 샘플링
- 합성 디테일 제거 데이터로 사전학습
- 실제 보정 패치가 확보되면 fine-tuning

## 13. 추론과 배포

- 원본 전체 크기 유지
- 1024 tile, overlap 128, halo를 포함해 거리장을 계산하고 중앙 영역만 합성
- 타일 경계는 가중 블렌딩
- 동일 모델 코드를 Gradio UI와 ComfyUI custom node에서 공유
- 사용자 조절값: 스타일 강도, 목표 선 굵기, 잉크 농도, 가장자리 매끄러움, 해칭 보존, Detail 강도
- seed와 denoise는 없음

## 14. 평가 기준 초안

### 구조 보존

- 입력 대비 출력 중심선 displacement의 평균과 95 percentile
- bidirectional Chamfer/distance-transform error
- 연결 성분·endpoint·junction 변화량
- 가는 눈선과 머리카락 선의 소실률
- 원본 얼굴·의상 실루엣 변화

### 선 스타일

- 선폭 분포가 원작 정답 분포에 가까워지는지
- 검정 농도와 회색 가장자리 분포
- 해칭 방향·간격 분포
- 배경 오염 픽셀 비율
- 작가 또는 작업자의 blind A/B 평가

잠정적인 안전 기준으로 중심선 이동 95 percentile을 0.75~1px 이내로 둘지 검토 중입니다. 절대값은 데이터 해상도에 따라 정규화해야 할 수 있습니다.

## 15. 특히 검토받고 싶은 질문

다음 질문에 각각 명확한 판단과 근거를 주세요.

1. 이 문제를 SDF residual prediction으로 푸는 접근이 타당한가? 더 단순하고 안전한 formulation이 있는가?
2. 20/50/80 다중 임계 SDF가 안티앨리어싱과 회색선을 다루는 데 실제 이점이 있는가, 아니면 soft distance transform 하나가 나은가?
3. 입력 SDF는 비미분적인 CPU EDT로 계산해도 되는가? 학습 속도와 배포를 고려하면 어떤 구현이 적절한가?
4. `ΔSDF + opacity + softness + gate` 네 Head가 역할 중복이나 식별 불가능성 문제를 일으킬 가능성이 있는가?
5. 거리장 기반 renderer가 교차선, 촘촘한 해칭, 넓은 검은 면에서 발생시킬 구체적인 실패 모드는 무엇인가?
6. adaptive displacement 제한 `min(2.5px, local_half_width × 0.45)`이 합리적인가? 더 좋은 구조 보존 제약이 있는가?
7. soft-clDice를 만화 선화에 적용하는 것이 타당한가? junction·해칭 때문에 부작용이 예상되는가?
8. 고해상도 경로와 1/2~1/4 문맥 경로의 이중 구조가 필요한가, 아니면 plain NAFNet/U-Net baseline부터 시작하는 것이 나은가?
9. 합성 열화된 원작 선화 → 원본 원작 선화 학습이 실제 타 작가 선화에 일반화될 수 있는가? 가장 큰 domain gap은 무엇인가?
10. 실제 paired data가 거의 없을 때 원작 작가의 눈과 해칭 선택 습관을 학습하는 가장 현실적인 방법은 무엇인가?
11. Core와 Detail Head를 분리하는 것이 옳은가, 아니면 하나의 multi-task network와 강한 regularization이 나은가?
12. 말풍선·텍스트·검은 면을 마스킹하는 현재 전처리에서 놓친 데이터 문제가 있는가?
13. 6,600장, 1024 patch, effective batch 8, 20~40 epoch가 이 모델 규모에 적절한가?
14. 반드시 포함해야 할 ablation과 실패 판정 기준은 무엇인가?
15. 이 설계를 폐기해야 할 정도의 근본적인 결함이 있다면 직접 지적하고 대체 구조를 제안해 달라.

## 16. 원하는 답변 형식

아래 형식으로 답변해 주세요.

1. **한 줄 결론**: 진행 가치가 있는지, 구조 수정 후 진행해야 하는지, 폐기해야 하는지
2. **치명적 문제**: 구현 전에 반드시 수정해야 할 항목
3. **중요하지만 치명적이지 않은 문제**
4. **더 단순한 baseline**: 가장 먼저 구현할 최소 모델과 이유
5. **추천 최종 구조**: 입력·backbone·head·renderer·loss를 구체적으로 제안
6. **데이터 전략**: 합성 열화와 실제 paired patch의 적정 구성
7. **최소 실험표**: 적은 비용으로 가설을 검증할 ablation 순서
8. **통과/실패 기준**: 다음 단계로 넘어갈 정량·정성 기준
9. **예상 실패 사례와 디버깅 방법**

가능하면 모호한 표현 대신 구체적인 수치 범위, 구조, 손실 조합과 비교 기준을 제시해 주세요. 현재 설계에 동의하는 부분보다 잘못됐거나 과도하게 복잡한 부분을 우선적으로 지적해 주세요.
