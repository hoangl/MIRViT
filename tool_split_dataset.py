import os
import shutil
import random


def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    """
    Chia tập dữ liệu thành 2 tập train và test.

    Tham số:
    - source_dir: Đường dẫn tới thư mục chứa dữ liệu gốc (vd: 'kvasir-dataset-v2')
    - dest_dir: Đường dẫn tới thư mục đích sẽ chứa 'train' và 'test'
    - split_ratio: Tỷ lệ tập train (mặc định 0.8 tương đương 80%)
    """

    # Định nghĩa đường dẫn thư mục train và test mới
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')

    # Lấy danh sách các thư mục con (các lớp bệnh lý)
    try:
        classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    except FileNotFoundError:
        print(f"[-] Không tìm thấy thư mục gốc: {source_dir}")
        return

    print(f"[*] Bắt đầu chia dữ liệu. Tìm thấy {len(classes)} lớp bệnh lý.")

    for class_name in classes:
        # Tạo đường dẫn đầy đủ tới thư mục của lớp hiện tại
        class_source_path = os.path.join(source_dir, class_name)
        class_train_path = os.path.join(train_dir, class_name)
        class_test_path = os.path.join(test_dir, class_name)

        # Khởi tạo thư mục đích nếu chưa có
        os.makedirs(class_train_path, exist_ok=True)
        os.makedirs(class_test_path, exist_ok=True)

        # Lấy danh sách toàn bộ file ảnh trong thư mục
        images = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]

        # Xáo trộn ngẫu nhiên danh sách ảnh
        random.seed(42)  # Set seed để kết quả luôn cố định nếu chạy lại nhiều lần
        random.shuffle(images)

        # Tính toán điểm cắt chia tập (split point)
        split_index = int(len(images) * split_ratio)

        # Phân tách danh sách
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Quá trình sao chép ảnh sang thư mục train
        for img in train_images:
            src_path = os.path.join(class_source_path, img)
            dst_path = os.path.join(class_train_path, img)
            shutil.copy2(src_path, dst_path)

        # Quá trình sao chép ảnh sang thư mục test
        for img in test_images:
            src_path = os.path.join(class_source_path, img)
            dst_path = os.path.join(class_test_path, img)
            shutil.copy2(src_path, dst_path)

        print(f"[+] Đã xử lý lớp '{class_name}': {len(train_images)} ảnh (Train) | {len(test_images)} ảnh (Test)")

    print("\n[*] Quá trình phân chia dữ liệu đã hoàn tất thành công!")


# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == '__main__':
    # THAY ĐỔI 2 ĐƯỜNG DẪN DƯỚI ĐÂY CHO PHÙ HỢP VỚI MÁY TÍNH CỦA BẠN

    # 1. Thư mục chứa dữ liệu Kvasir vừa giải nén
    SOURCE_DIRECTORY = r"D:\Study\PTIT\TKTT\TinyViT\kvasir-dataset-v2"

    # 2. Thư mục bạn muốn xuất kết quả (có thể lưu ra một thư mục mới hoàn toàn)
    DESTINATION_DIRECTORY = r"D:\Study\PTIT\TKTT\TinyViT\kvasir-dataset-v2-split"

    split_dataset(
        source_dir=SOURCE_DIRECTORY,
        dest_dir=DESTINATION_DIRECTORY,
        split_ratio=0.8
    )