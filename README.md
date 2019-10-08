# mnist_regconizer
This is a first lab of Soft computing course. It aims to implement neural network (1 hidden layer) and optimize this network to increase the accuracy of the model. However, I've generalized the network for L layers in implementation.

## Install packages
```
pip install -r requirements.txt
```

## Model architecture
Mạng neural network gồm 1 hoặc nhiều lớp ẩn nhưng ở đây chỉ giới hạn khảo sát 1 hoặc 2 lớp ẩn.


## Chi tiết cài đặt
```
Input: (784, m) - m là số lượng mẫu

Output: (1, m)
```
Sử dụng numpy làm thư viện chính.

Phương thức khởi tạo là He initialization để tránh hiện tượng vanishing/exploding gradient và hoạt động hiệu quả với hàm Relu.

Trong đó, thuật toán optimizer được sử dụng là minibatch gradient descent. Theo đó, regularization sử dụng là weight decay (L2 norm) để tránh overfitting. Hàm lỗi sử dùng là softmax cross-entropy. 

Activation function sử dụng là Relu ở hiden layers và Softmax được sử dụng ở output layer.

## Tuning parameters

Ở đây, ta có các siêu tham số cần điều chỉnh như:
- nx: số chiều của X
- nh1: số units của hidden layer 1
- nh2: số units của hidden layer 2
- epoches: số epoch để train model
- batch size: mini batch size
- learning rate: tốc độ học
- weight decay: L2 regularization để tránh overfitting

## Run commandline
Để nhận sự trợ giúp, gõ lệnh:

```
python 1612174.py -h
```
 
Các tham số quan trọng cần cung cấp

```
python 1612174.py -train "Your train file" -test digit-recognizer/test.csv -nh1 1000 -epoches 50 -batch 64 -lr 0.5 -decay 5e-4
```

Parameters được lưu xuống dưới tên sau: ddmmYY-HMS (dd/mm/Y H:M:S)

Ví dụ: 08102019-165350
