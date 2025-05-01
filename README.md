# Output Correction for Accuracy Recovery in PTQ-Based INT4 Models
----

모델을 INT4로 양자화 했을 때 발생하는 양자화 오류는 noise한 값으로 본다.

이러한 노이즈한 값을 줄이기 위해 보정 module(denoising module 1, 2)를 삽입하여 양자화시 발생하는 정확도 손실을 최소화할 수 있도록 module을 설계하였다.
