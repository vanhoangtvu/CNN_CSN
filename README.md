# 🧠 Đồ Án Cơ Sở Ngành - CIFAR-10 Image Classification

[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-blue)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-High%20Level%20API-red)](https://keras.io/)

## 👨‍🎓 Thông Tin Sinh Viên

- **Họ và tên:** Nguyễn Văn Hoàng
- **Ngành học:** Công nghệ thông tin
- **Trường:** Đại học Trà Vinh
- **Khoa:** Kỹ thuật và Công nghệ
- **Năm học:** 2023-2024

## 📖 Tóm Tắt Đồ Án

Đồ án này nghiên cứu và triển khai một mô hình **Convolutional Neural Network (CNN)** để phân loại hình ảnh trên tập dữ liệu **CIFAR-10**. Mục tiêu là xây dựng một mô hình hiệu quả có thể đạt được độ chính xác cao (>90%) trong việc phân loại 10 lớp đối tượng khác nhau.

### 🎯 Mục Tiêu

- Nghiên cứu và ứng dụng Deep Learning trong Computer Vision
- Xây dựng mô hình CNN tối ưu cho phân loại hình ảnh
- Đạt được độ chính xác cao với mô hình compact
- Áp dụng các kỹ thuật hiện đại: Data Augmentation, Regularization

### 📊 Kết Quả Đạt Được

- ✅ **Độ chính xác:** >90% trên tập test
- ✅ **Mô hình compact:** Chỉ 1.2M parameters
- ✅ **Tối ưu hóa:** Áp dụng thành công các kỹ thuật chống overfitting
- ✅ **Khả năng tổng quát:** Hoạt động tốt trên hình ảnh thực tế

## 🗂️ Cấu Trúc Dự Án

```
CNN_CSN/
├── cifar-10-image-classification-with-cnn.ipynb 
├── README.md                                     
└── requirements.txt                                
```

## 📸 Tập Dữ Liệu CIFAR-10

CIFAR-10 là một tập dữ liệu benchmark nổi tiếng cho bài toán phân loại hình ảnh:

- **60,000 hình ảnh màu** (32x32 pixels)
- **10 lớp đối tượng:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **50,000 ảnh training** + **10,000 ảnh testing**
- **6,000 ảnh mỗi lớp** (phân bố đều)

## 🏗️ Kiến Trúc Mô Hình

Mô hình CNN được thiết kế dựa trên kiến trúc **VGG-inspired** với các đặc điểm:

### Các Thành Phần Chính:
- **Convolutional Layers:** 8 lớp Conv2D với filter tăng dần (32→64→128→256)
- **Pooling Layers:** 4 lớp MaxPooling2D để giảm kích thước
- **Regularization:** Batch Normalization, Dropout, L2 Regularization
- **Classification:** 1 lớp Dense với Softmax activation

### Kỹ Thuật Tối Ưu:
- **Data Augmentation:** Rotation, shifts, flips, zoom, brightness
- **Learning Rate Scheduling:** ReduceLROnPlateau
- **Early Stopping:** Ngăn overfitting, restore best weights

## 🛠️ Công Nghệ Sử Dụng

### Ngôn Ngữ & Framework:
- **Python 3.8+**
- **TensorFlow 2.x**
- **Keras** (High-level API)

### Thư Viện Chính:
- **numpy** - Tính toán số học
- **matplotlib** - Trực quan hóa
- **opencv-python** - Xử lý hình ảnh
- **scikit-learn** - Machine learning utilities

### Môi Trường Phát Triển:
- **Jupyter Notebook**
- **VS Code** (khuyên dùng)

## 🚀 Hướng Dẫn Chạy

### 1. Cài Đặt Dependencies

```bash
# Clone repository (nếu có)
git clone [your-repo-url]
cd CNN_CSN

# Tạo virtual environment (khuyên dùng)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt packages
pip install tensorflow numpy matplotlib opencv-python scikit-learn jupyter
```

### 2. Chạy Notebook

```bash
# Khởi động Jupyter
jupyter notebook

# Mở file: cifar-10-image-classification-with-cnn.ipynb
# Chạy từng cell theo thứ tự
```

### 3. Chạy Toàn Bộ Pipeline

```python
# Trong notebook, chạy tất cả cells theo thứ tự:
# 1. Import libraries
# 2. Load và explore data  
# 3. Preprocessing
# 4. Define model architecture
# 5. Train model
# 6. Evaluate results
```

## 📈 Quy Trình Thực Hiện

### Bước 1: Import Thư Viện
- Import TensorFlow, Keras, NumPy, Matplotlib
- Thiết lập warnings và configurations

### Bước 2: Chuẩn Bị Dữ Liệu
- Load CIFAR-10 dataset từ Keras
- Chia train/validation/test sets
- Visualize sample images

### Bước 3: Tiền Xử Lý
- **Normalization:** Chuẩn hóa pixel values
- **One-hot Encoding:** Transform labels
- **Data Augmentation:** Tăng cường dữ liệu

### Bước 4: Xây Dựng Mô Hình
- Define CNN architecture
- Compile với Adam optimizer
- Summary model parameters

### Bước 5: Huấn Luyện
- Training với callbacks (ReduceLR, EarlyStopping)
- Monitor validation performance
- Save best model weights

### Bước 6: Đánh Giá
- Evaluate trên test set
- Visualize learning curves
- Test với hình ảnh external

## 📊 Kết Quả Chi Tiết

### Performance Metrics:
- **Test Accuracy:** >90%
- **Model Size:** ~1.2M parameters
- **Training Time:** ~30-50 epochs
- **Convergence:** Stable without overfitting

### Visualization:
- Learning curves (accuracy & loss)
- Sample predictions
- Confusion matrix
- Feature maps visualization

## 🔮 Hướng Phát Triển

### Ngắn Hạn:
- [ ] Thử nghiệm với CIFAR-100
- [ ] Implement Transfer Learning
- [ ] Add more data augmentation techniques
- [ ] Experiment với different optimizers

### Dài Hạn:
- [ ] Deploy model lên web application
- [ ] Mobile app integration
- [ ] Real-time object detection
- [ ] Advanced architectures (ResNet, EfficientNet)

## 📚 Tài Liệu Tham Khảo

1. **Krizhevsky, A., Hinton, G.** (2009). Learning multiple layers of features from tiny images.
2. **Simonyan, K., Zisserman, A.** (2014). Very deep convolutional networks for large-scale image recognition.
3. **Goodfellow, I., Bengio, Y., Courville, A.** (2016). Deep Learning. MIT Press.
4. **Chollet, F.** (2021). Deep Learning with Python, Second Edition.
5. [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
6. [Keras Documentation](https://keras.io/)
7. [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🤝 Đóng Góp

Đây là đồ án cá nhân cho môn Cơ sở ngành. Mọi góp ý và thảo luận học thuật đều được hoan nghênh!

## 📞 Liên Hệ

- **Email:** [nguyenhoang4556z@gmail.com]
- **GitHub:** [https://github.com/vanhoangtvu]
- **LinkedIn:** [https://github.com/vanhoangtvu]

## 📄 License

Đồ án này được thực hiện cho mục đích học tập tại Đại học Trà Vinh.

---

<div align="center">

**🎓 Đồ Án Cơ Sở Ngành - Nguyễn Văn Hoàng**  
**🏫 Đại học Trà Vinh - Khoa Kỹ thuật và Công nghệ**  
**📅 Năm học 2024-2025**

</div>
