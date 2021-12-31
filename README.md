# FocalLoss

Focal Loss는 학습 중 클래스 불균형(class imbalance) 문제를 해결하기 위해 https://arxiv.org/pdf/1708.02002.pdf 논문에서 소개되었습니다.





### Usage

    from FocalLoss import WeightedFocalLoss
  
    loss_function = WeightedFocalLoss()  # Focal Loss
  
    loss = loss_function(outputs, y_train.to(device))
