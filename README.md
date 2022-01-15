# FocalLoss

Focal Loss는 학습 중 클래스 불균형(Class Imbalance) 문제를 해결하기 위해 RetinaNet 논문(https://arxiv.org/pdf/1708.02002.pdf) 에서 소개되었습니다.

클래스 불균형 문제는 다음 2가지 문제의 원인이 됩니다.

+ 대부분의 Location은 학습에 기여하지 않는 Background와 같은 Easy Negative이므로 (Detector에 의해 Background로 쉽게 분류될 수 있음을 의미함) 학습에 비효율적입니다.

+ Easy negative 각각은 높은 확률로 Object가 아님을 잘 구분할 수 있습니다. 즉, 각각의 Loss 값은 작습니다. 하지만 전체 비중이 굉장히 크므로 전체 Loss 및 Gradient를 계산할 때, Easy Negative의 영향이 압도적으로 커지는 문제가 발생합니다.

이 문제들을 해결하기 위해 FocalLoss가 나오게 되는데, Focal Loss는 간단히 말하면 Cross Entropy의 클래스 불균형 문제를 다루기 위한 개선된 버전이라고 말할 수 있으며 어렵거나 쉽게 오분류되는 케이스에 대하여 더 큰 가중치를 주는 방법을 사용합니다. (Object 일부분만 있거나, 실제 분류해야 되는 Object들이 이에 해당합니다.) 

반대로 쉬운 케이스의 경우 낮은 가중치를 반영합니다. (Background Object가 이에 해당합니다.)


<img src="https://user-images.githubusercontent.com/29745280/147806849-56b0b1f0-3ad8-43c7-8021-daae15e6e427.png"  width="400" height="300"/>

빨간색 케이스의 경우 Hard Case의 문제이며 초록색의 경우 Easy Case의 문제입니다. 

그래프에 적용된 α=1,γ=1 입니다.

똑같이 loss값이 작아지긴 했지만 Easy Case의 Loss값이 Hard Case의 Loss값에 비해 큰 폭으로 작아졌음을 볼 수 있습니다.

이러한 원리로 이미지의 대부분에 해당되는 Easy Negative (Bakground)의 학습을 줄임으로써 효율적인 학습이 가능해집니다.


### Usage

    from FocalLoss import WeightedFocalLoss
  
    loss_function = WeightedFocalLoss()  # Focal Loss
  
    loss = loss_function(outputs, y_train.to(device))
