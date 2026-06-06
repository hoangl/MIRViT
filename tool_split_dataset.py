import os
import shutil
import random


def split_dataset(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Chia tập dữ liệu thành 3 tập: train, validation và test.

    Tham số:
    - source_dir: Đường dẫn tới thư mục chứa dữ liệu gốc
    - dest_dir: Đường dẫn tới thư mục đích sẽ chứa 'train', 'val' và 'test'
    - train_ratio: Tỷ lệ tập train (mặc định 0.8 tương đương 80%)
    - val_ratio: Tỷ lệ tập validation (mặc định 0.1 tương đương 10%)
    - Tỷ lệ tập test sẽ được tự động tính: 1.0 - train_ratio - val_ratio
    """

    # Định nghĩa đường dẫn 3 thư mục mới
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    test_dir = os.path.join(dest_dir, 'test')

    # Lấy danh sách các thư mục con (các lớp bệnh lý)
    try:
        classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    except FileNotFoundError:
        print(f"[-] LỖI: Không tìm thấy thư mục gốc '{source_dir}'")
        return

    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"[*] Bắt đầu chia dữ liệu. Tìm thấy {len(classes)} lớp bệnh lý.")
    print(
        f"[*] Tỷ lệ chia: Train {train_ratio * 100:.0f}% | Val {val_ratio * 100:.0f}% | Test {test_ratio * 100:.0f}%\n")

    for class_name in classes:
        # Tạo đường dẫn đầy đủ
        class_source_path = os.path.join(source_dir, class_name)
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)
        class_test_path = os.path.join(test_dir, class_name)

        # Khởi tạo thư mục đích nếu chưa có
        os.makedirs(class_train_path, exist_ok=True)
        os.makedirs(class_val_path, exist_ok=True)
        os.makedirs(class_test_path, exist_ok=True)

        # Lấy danh sách toàn bộ file ảnh trong thư mục
        images = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]

        # Xáo trộn ngẫu nhiên danh sách ảnh (Set seed 42 để cố định kết quả xáo trộn)
        random.seed(42)
        random.shuffle(images)

        # Tính toán các mốc cắt (split points)
        train_idx = int(len(images) * train_ratio)
        val_idx = train_idx + int(len(images) * val_ratio)

        # Phân tách danh sách ảnh
        train_images = images[:train_idx]
        val_images = images[train_idx:val_idx]
        test_images = images[val_idx:]

        # Hàm hỗ trợ sao chép file
        def copy_files(file_list, dst_folder):
            for img in file_list:
                src_path = os.path.join(class_source_path, img)
                dst_path = os.path.join(dst_folder, img)
                shutil.copy2(src_path, dst_path)

        # Thực thi sao chép
        copy_files(train_images, class_train_path)
        copy_files(val_images, class_val_path)
        copy_files(test_images, class_test_path)

        print(f"[+] Lớp '{class_name}': {len(train_images)} Train | {len(val_images)} Val | {len(test_images)} Test")

    print("\n[*] Quá trình phân chia dữ liệu (Train/Val/Test) đã hoàn tất thành công!")


# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == '__main__':
    # 1. Thư mục chứa dữ liệu Kvasir vừa giải nén
    SOURCE_DIRECTORY = r"kvasir-dataset-v2"

    # 2. Thư mục bạn muốn xuất kết quả
    DESTINATION_DIRECTORY = r"kvasir-dataset-v2-split"

    # Chạy hàm chia dữ liệu với tỷ lệ 80% Train - 10% Val - 10% Test
    split_dataset(
        source_dir=SOURCE_DIRECTORY,
        dest_dir=DESTINATION_DIRECTORY,
        train_ratio=0.8,
        val_ratio=0.1
    )