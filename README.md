# 🏥 Hệ thống Truy xuất Ảnh Y tế (Medical Image Retrieval System)

Dự án này triển khai và đánh giá các kiến trúc học sâu bao gồm **CNN truyền thống**, **Vision Transformer (ViT)** và **Kiến trúc Lai (Hybrid - MedViT)** cho bài toán Truy xuất Ảnh Y tế trên tập dữ liệu **Kvasir-v2**. Hệ thống thực hiện phân tích so sánh chi tiết giữa hai phương pháp huấn luyện: **Contrastive Learning** (Học đối chiếu) và **Cross Entropy** (Học phân loại), đi kèm với bộ công cụ XAI (Explainable AI) đa luồng để diễn giải quyết định của mạng nơ-ron.

---

## 📁 Cấu trúc Thư mục

Dự án được tổ chức theo cấu trúc sau:

~~~text
TinyViT/
│
├── kvasir-dataset-v2/          # Tập dữ liệu gốc tải về (Gồm 8 thư mục bệnh lý)
├── kvasir-dataset-v2-split/    # Tập dữ liệu đã được chia tỷ lệ Train/Test
├── saved_models/               # Nơi lưu trữ trọng số (.pth) và cơ sở dữ liệu vector (.pt)
│
├── models/                     # Thư mục chứa kiến trúc mạng cục bộ (Local Models)
│   ├── MedViT.py               # Mã nguồn định nghĩa mạng MedViT (đã fix lỗi cho Retrieval)
│   ├── utils.py                # Các hàm hỗ trợ tối ưu hóa và gộp Batch-Norm (merge_pre_bn)
│   └── requirements.txt        # Thư viện phụ thuộc riêng cho MedViT (einops, fvcore...)
│
├── requirements.txt            # Danh sách các thư viện phụ thuộc của hệ thống chính
├── tool_split_dataset.py       # Mã nguồn chia tập dữ liệu (Train/Val/Test)
├── train_models.py             # Mã nguồn huấn luyện toàn bộ các mô hình
└── demo_app.py                 # Giao diện Web UI (Gradio) để kiểm thử và trực quan hóa XAI
~~~

---

## 💻 Yêu cầu Hệ thống & Môi trường

Để hệ thống hoạt động ổn định và tránh lỗi tràn bộ nhớ, máy tính của bạn cần đáp ứng các tiêu chí sau:

- **Hệ điều hành:** Windows 10 / Windows 11.
- **Môi trường lập trình:** Tương thích tốt nhất với **Python 3.10** hoặc **3.11** (Lưu ý: Không dùng Python 3.12+ do một số thư viện PyTorch/XAI chưa hỗ trợ ổn định trên Windows).
- **Phần cứng:** - RAM hệ thống: Tối thiểu 16GB.
  - Card đồ họa (GPU): NVIDIA hỗ trợ CUDA với **tối thiểu 4GB VRAM** (Đã được cấu hình tối ưu luồng bộ nhớ tự động để chống tràn VRAM cho các dòng card phổ thông như RTX 3050).

---

## ⚙️ Hướng dẫn Cài đặt (Dành cho Windows)

**Bước 1: Cài đặt Python và Pip đúng phiên bản**
1. Truy cập trang chủ [Python.org](https://www.python.org/downloads/windows/).
2. Tải về bộ cài đặt **Python 3.10.x** hoặc **Python 3.11.x** (Windows installer 64-bit).
3. Trong quá trình cài đặt, **BẮT BUỘC** phải tích vào ô **"Add Python 3.x to PATH"** ở màn hình cài đặt đầu tiên.
4. Mở Command Prompt (cmd) và kiểm tra lại bằng lệnh:
   ~~~bash
   python --version
   pip --version
   ~~~
   *(Hệ thống phải trả về phiên bản 3.10.x hoặc 3.11.x mới là thành công).*

**Bước 2: Cài đặt PyTorch với hỗ trợ GPU (CUDA)**
Để tận dụng tối đa phần cứng NVIDIA trên Windows, bạn phải cài đặt PyTorch phiên bản hỗ trợ CUDA 11.8 trước tiên:
~~~bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
~~~

**Bước 3: Cài đặt các thư viện AI phụ trợ (2 gói Requirements)**
Mở cmd tại thư mục gốc `TinyViT` và tiến hành cài đặt lần lượt **thư viện hệ thống** và **thư viện cho Local Model**:
~~~bash
# Cài đặt thư viện lõi của hệ thống (timm, gradio, grad-cam,...)
pip install -r requirements.txt

# Cài đặt thư viện chuyên biệt để khởi tạo mạng MedViT nội bộ
pip install -r models\requirements.txt
~~~

**Bước 4: Cấu hình Token tải mô hình (Hugging Face)**
Hệ thống sử dụng thư viện `timm` để tự động tải các kiến trúc ViT/CNN. Mã nguồn đã được cấu hình sẵn biến môi trường Token, bạn không cần thao tác thêm:
~~~python
import os
os.environ["HF_TOKEN"] = "hf_xxxx"
~~~

---

## 🚀 Các bước chạy Hệ thống (Quy trình chuẩn)

Hệ thống cần được thực thi theo thứ tự tuyến tính gồm 3 bước dưới đây:

### 1. Tải và Chuẩn bị Dữ liệu (Data Preparation)
Đầu tiên, tải tập dữ liệu **Kvasir-v2** gốc từ link chính thức của Simula:
🔗 **[Download Kvasir-dataset-v2.zip](https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip)**

Giải nén tệp vừa tải về và đảm bảo thư mục giải nén có tên là `kvasir-dataset-v2` (đặt ngang hàng với các file code). Sau đó tiến hành chia tập dữ liệu bằng lệnh:
~~~bash
python tool_split_dataset.py
~~~
**Mục đích:** Kịch bản này sẽ đọc dữ liệu gốc, xáo trộn ngẫu nhiên và tự động chia thành tập Train (80%) và tập Test (20%), sau đó kết xuất vào thư mục `kvasir-dataset-v2-split`.

### 2. Huấn luyện Mô hình (Training Pipeline)
Bắt đầu quá trình huấn luyện toàn bộ các mô hình bằng lệnh:
~~~bash
python train_models.py
~~~
**Mục đích:** - Tự động huấn luyện lần lượt các kiến trúc: `DenseNet121`, `ResNet50`, `MIRViT_small`, `MIRdeit_small`, và mạng cục bộ `MedViT_T`.
- **Cấu hình thông minh:** Hệ thống tự động nhận diện mô hình để cấp phát bộ nhớ. Với CNN/ViT, `batch_size=32`. Riêng mạng lai `MedViT` phức tạp, hệ thống tự động giảm `batch_size=16` để bảo vệ an toàn cho VRAM 4GB.
- Trọng số `.pth` và Database Vector `.pt` sẽ tự động được lưu vào thư mục `saved_models`.

### 3. Khởi chạy Giao diện Trực quan (Web UI Demo)
Sau khi quá trình huấn luyện hoàn tất ít nhất 1 mô hình, khởi chạy ứng dụng Gradio để kiểm thử:
~~~bash
python demo_app.py
~~~
**Mục đích:**
- Mở máy chủ Web tại địa chỉ `http://127.0.0.1:8081`.
- Hệ thống tự động quét thư mục `saved_models` và nạp các mô hình đã huấn luyện lên giao diện. Mô hình cục bộ MedViT sẽ tự động kích hoạt hàm `merge_bn()` từ `utils.py` để tăng tốc độ tìm kiếm.
- Tính năng **Tất cả mô hình (So sánh)** cho phép truy xuất và đối chiếu kết quả của nhiều thuật toán cùng lúc bằng cơ chế *Nạp - Xả VRAM liên tục* để chống tràn bộ nhớ.
- Tự động sinh biểu đồ nhiệt (Heatmap) XAI theo cấu trúc mạng lưới (hỗ trợ TIS/Rollout cho ViT và LayerCAM/GradCAM cho CNN).

---

## 🔍 Khắc phục sự cố thường gặp (Troubleshooting)

- **Lỗi `ModuleNotFoundError`:** Do hệ thống cài Python Global, nếu máy tính của bạn cài nhiều bản Python cùng lúc (VD: 3.9 và 3.12), hãy chắc chắn bạn đang dùng đúng lệnh `pip` của bản 3.10/3.11. Lời khuyên là dùng lệnh: `python -m pip install -r requirements.txt`.
- **Lỗi tràn bộ nhớ (CUDA Out of Memory):** Nếu GPU báo lỗi tràn VRAM (đặc biệt khi mở tab Demo so sánh tất cả mô hình), hãy kiểm tra Task Manager và tắt các ứng dụng ngầm đang chiếm dụng VRAM. Cơ chế `gc.collect()` và `torch.cuda.empty_cache()` trong file demo đã được thiết kế để tự động dọn rác bộ nhớ sau mỗi phiên truy vấn.
- **Cảnh báo `triton not found` / `flop counting will not work`:** Đây là cảnh báo đặc thù của PyTorch 2.x trên hệ điều hành Windows khi không tìm thấy trình biên dịch Triton. Cảnh báo này **hoàn toàn vô hại**, không ảnh hưởng đến độ chính xác (mAP) cũng như luồng thực thi của mô hình.
- **Hiển thị GPU 0% trong Task Manager:** Mặc định Windows hiển thị biểu đồ lõi 3D (dành cho xử lý đồ họa/Game). Để xem công suất huấn luyện AI thực sự, hãy bấm vào mục "3D" trong biểu đồ GPU của Task Manager và chuyển sang xem lõi "Cuda" hoặc "Compute", hoặc sử dụng lệnh `nvidia-smi` trong Command Prompt.