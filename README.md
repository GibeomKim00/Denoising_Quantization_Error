# Output Correction for Accuracy Recovery in PTQ-Based INT4 Models
----

## 해당 주제를 생각하게된 계기
모델을 INT8 또는 더 낮은 저비트로 양자화를 할 경우 양자화 파라미터 scale과 zero-point에 의해 반드시 quantization error가 발생할 수 밖에 없다. 이는 곧 FP모델 대비 성능이 떨어지는 것을 의미한다.

아무리 quantization error를 줄이기 위한 양자화 기법(etc. GPTQ, QuaRot)이 있더라도 scale과 zero-point가 있는 한 quantization error를 없애지 못한다.

따라서 기존 모델의 아키텍쳐에 보정 모듈을 추가하여 quantization error를 줄여 기존 FP모델의 성능을 복원하는 것을 목표로 한다.

## Method
사용한 모델은 ResNet으로 총 두 개의 보정 모듈을 각각 global average pooling layer 전에, fully connected layer 이후에 삽입하였다.

### 1. Denoising Module 1
Global average pooling layer 전에 삽입된 모듈로 teacher model(FP model)에서 같은 위치에 출력한 값을 이용하여 MSE를 통해 학습을 진행하였다.

### 2. Denoising Module 2
Fully connected layer 이후에 삽인된 모듈로 실제 sample의 정답 label과 양자화된 모델의 출력 logit 간의 Cross Entropy Loss를 통해 학습을 진행하였다.
