import os
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_TOKEN"] = "hf_xxx"

# =====================================================================
# CẤU HÌNH EXPERIMENT CHUẨN BÀI BÁO (BỔ SUNG ViT-BASE)
# =====================================================================
NUM_RUNS = 1
TOTAL_ITERATIONS = 10000

EXPERIMENTS = [
    # --- Contrastive ---
    {"name": "Densenet121", "model_str": "densenet121", "loss_type": "Contrastive", "lam": 0.7},
    {"name": "Resnet50", "model_str": "resnet50", "loss_type": "Contrastive", "lam": 0.7},
    {"name": "MIRViT_small", "model_str": "vit_small_patch16_224", "loss_type": "Contrastive", "lam": 0.7},
    {"name": "MIRdeit_small", "model_str": "deit_small_patch16_224", "loss_type": "Contrastive", "lam": 0.3},
    {"name": "MedViT_T", "model_str": "medvit_small", "loss_type": "Contrastive", "lam": 0.7},

    # --- Cross Entropy ---
    {"name": "deit_small", "model_str": "deit_small_patch16_224", "loss_type": "Cross Entropy", "lam": 0.0},
    {"name": "ViT_small", "model_str": "vit_small_patch16_224", "loss_type": "Cross Entropy", "lam": 0.0},
    {"name": "Densenet121", "model_str": "densenet121", "loss_type": "Cross Entropy", "lam": 0.0},
    {"name": "Resnet50", "model_str": "resnet50", "loss_type": "Cross Entropy", "lam": 0.0},
]


# =====================================================================
# CLASS HỖ TRỢ GHI LOG RA FILE VÀ CONSOLE CÙNG LÚC
# =====================================================================
class DualLogger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        # Mở file với chế độ 'a' (append) để nối tiếp dữ liệu, có mã hóa utf-8 để không lỗi font tiếng Việt
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Đảm bảo dữ liệu được đẩy ngay lập tức ra màn hình và file
        self.terminal.flush()
        self.log.flush()


# =====================================================================
# CÁC LỚP VÀ HÀM HỖ TRỢ LOSS (THUẬT TOÁN CHUẨN)
# =====================================================================
class CrossBatchMemory:
    def __init__(self, queue_size, embedding_dim, device='cuda'):
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.features = torch.zeros(queue_size, embedding_dim, device=device)
        self.labels = torch.zeros(queue_size, dtype=torch.long, device=device) - 1
        self.ptr = 0
        self.is_full = False

    def update(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.ptr)

        if ptr + batch_size <= self.queue_size:
            self.features[ptr:ptr + batch_size] = keys.detach()
            self.labels[ptr:ptr + batch_size] = labels.detach()
            self.ptr = (ptr + batch_size) % self.queue_size
            if ptr + batch_size == self.queue_size:
                self.is_full = True
        else:
            overflow = (ptr + batch_size) - self.queue_size
            self.features[ptr:self.queue_size] = keys.detach()[:batch_size - overflow]
            self.labels[ptr:self.queue_size] = labels.detach()[:batch_size - overflow]
            self.features[0:overflow] = keys.detach()[batch_size - overflow:]
            self.labels[0:overflow] = labels.detach()[batch_size - overflow:]
            self.ptr = overflow
            self.is_full = True


def contrastive_loss(z, labels, memory_queue, margin=0.5):
    z = F.normalize(z, p=2, dim=1)

    if memory_queue.is_full:
        valid_features = memory_queue.features
        valid_labels = memory_queue.labels
    else:
        if memory_queue.ptr == 0: return torch.tensor(0.0, device=z.device)
        valid_features = memory_queue.features[:memory_queue.ptr]
        valid_labels = memory_queue.labels[:memory_queue.ptr]

    valid_features = F.normalize(valid_features, p=2, dim=1)
    all_features = torch.cat([z, valid_features], dim=0)
    all_labels = torch.cat([labels, valid_labels], dim=0)

    similarity_matrix = torch.matmul(z, all_features.T)
    loss = 0.0
    N = z.size(0)

    for i in range(N):
        pos_mask = (all_labels == labels[i])
        neg_mask = (all_labels != labels[i])
        pos_mask[i] = False

        pos_sims = similarity_matrix[i][pos_mask]
        if len(pos_sims) > 0:
            loss += torch.mean(1.0 - pos_sims)

        neg_sims = similarity_matrix[i][neg_mask]
        if len(neg_sims) > 0:
            neg_loss = neg_sims - margin
            active_neg_loss = neg_loss[neg_loss > 0]
            if len(active_neg_loss) > 0:
                loss += torch.mean(active_neg_loss)

    return loss / N


def koleo_loss(z):
    z = F.normalize(z, p=2, dim=1)
    distances = torch.cdist(z, z)
    mask = torch.eye(distances.size(0), dtype=torch.bool, device=z.device)
    distances = distances.masked_fill(mask, float('inf'))
    min_distances, _ = torch.min(distances, dim=1)
    return -torch.mean(torch.log(min_distances + 1e-8))


def evaluate_retrieval_all_metrics(embeddings, labels, k_list=[1, 5, 10]):
    num_queries = embeddings.size(0)
    results = {'R@1': 0.0, 'R@5': 0.0, 'R@10': 0.0, 'mP@1': 0.0, 'mP@5': 0.0, 'mP@10': 0.0, 'mAP': 0.0}

    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    similarity_matrix.fill_diagonal_(-1.0)

    for i in range(num_queries):
        query_label = labels[i].item()
        scores = similarity_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True)
        retrieved_labels = labels[sorted_indices]
        is_relevant = (retrieved_labels == query_label)

        for k in k_list:
            if is_relevant[:k].sum() > 0: results[f'R@{k}'] += 1
            results[f'mP@{k}'] += is_relevant[:k].sum().item() / k

        total_relevant = is_relevant.sum().item()
        if total_relevant > 0:
            ap, hits = 0.0, 0
            for rank, relevant in enumerate(is_relevant):
                if relevant:
                    hits += 1
                    ap += hits / (rank + 1)
            results['mAP'] += ap / total_relevant

    for key in results: results[key] = (results[key] / num_queries) * 100
    return results


# =====================================================================
# HÀM HUẤN LUYỆN 1 LẦN CHẠY (SINGLE RUN)
# =====================================================================
def run_single_experiment(exp_config, train_loader, val_loader, val_dataset, train_dataset_size, device, run_idx,
                          num_classes=8):
    loss_type = exp_config["loss_type"]
    model_str = exp_config["model_str"]
    lam_koleo = exp_config["lam"]

    is_ce = (loss_type == "Cross Entropy")
    n_classes = num_classes if is_ce else 0

    model = timm.create_model(model_str, pretrained=True, num_classes=n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda')
    ce_criterion = nn.CrossEntropyLoss() if is_ce else None

    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        dummy_out = model.forward_features(dummy_input) if is_ce else model(dummy_input)
        if dummy_out.dim() > 2:
            embed_dim = dummy_out.flatten(1).shape[1] if is_ce else dummy_out.shape[1]
        else:
            embed_dim = dummy_out.shape[1]

    memory = CrossBatchMemory(queue_size=train_dataset_size, embedding_dim=embed_dim, device=device)

    model.train()
    current_iter = 0

    while current_iter < TOTAL_ITERATIONS:
        for images, labels in train_loader:
            if current_iter >= TOTAL_ITERATIONS: break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                if is_ce:
                    logits = model(images)
                    total_loss = ce_criterion(logits, labels)
                else:
                    embeddings = model(images)
                    if embeddings.dim() > 2: embeddings = embeddings.flatten(1)
                    loss_c = contrastive_loss(embeddings, labels, memory, margin=0.5)
                    loss_k = koleo_loss(embeddings)
                    total_loss = loss_c + lam_koleo * loss_k
                    memory.update(embeddings, labels)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_iter += 1
            if current_iter % 200 == 0:
                print(
                    f"   [Run {run_idx}/{NUM_RUNS}] Iteration {current_iter}/{TOTAL_ITERATIONS} | Loss: {total_loss.item():.4f}")

    model.eval()
    if is_ce: model.reset_classifier(0)

    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda'):
                embeddings = model(images)
                if embeddings.dim() > 2: embeddings = embeddings.flatten(1)
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(normalized_embeddings.cpu())
            all_labels.append(labels)

    db_embeddings = torch.cat(all_embeddings, dim=0)
    db_labels = torch.cat(all_labels, dim=0)

    metrics = evaluate_retrieval_all_metrics(db_embeddings, db_labels)

    if run_idx == NUM_RUNS:
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        safe_name = f"{loss_type.lower().replace(' ', '')}_{exp_config['name'].lower()}"
        torch.save(model.state_dict(), os.path.join(save_dir, f'{safe_name}.pth'))

        db_image_paths = [sample[0] for sample in val_dataset.samples]
        database_data = {'embeddings': db_embeddings, 'labels': db_labels, 'paths': db_image_paths,
                         'class_names': val_dataset.classes}
        torch.save(database_data, os.path.join(save_dir, f'db_{safe_name}.pt'))

    return metrics


# =====================================================================
# LUỒNG ĐIỀU KHIỂN CHÍNH
# =====================================================================
def main():
    # KÍCH HOẠT GHI LOG RA FILE
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    sys.stdout = DualLogger(log_filename)

    print("=" * 80)
    print(f"[*] Toàn bộ log của phiên huấn luyện này sẽ được lưu tại: {log_filename}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang khởi chạy hệ thống trên thiết bị: {device}")

    train_dir = r"kvasir-dataset-v2-split\train"
    val_dir = r"kvasir-dataset-v2-split\val"

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("[-] LỖI: Không tìm thấy thư mục dữ liệu train hoặc val.")
        return

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n[*] Đang đọc và phân tích cấu trúc dữ liệu...")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    train_size = len(train_dataset)

    print(f"[+] Tổng số ảnh tập Huấn luyện (Train): {train_size} ảnh")
    print(f"[+] Tổng số ảnh tập Xác thực/Cơ sở dữ liệu (Val): {len(val_dataset)} ảnh")
    print(f"[+] Danh sách các lớp bệnh lý ({len(train_dataset.classes)} lớp): {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    final_summary = {}

    for config in EXPERIMENTS:
        model_name = config["name"]
        loss_type = config["loss_type"]

        print(f"\n" + "=" * 80)
        print(f"[*] BẮT ĐẦU HUẤN LUYỆN: {model_name} | LOSS: {loss_type}")
        print("=" * 80)

        run_metrics = {k: [] for k in ['R@1', 'R@5', 'R@10', 'mAP', 'mP@1', 'mP@5', 'mP@10']}

        for run in range(1, NUM_RUNS + 1):
            metrics = run_single_experiment(config, train_loader, val_loader, val_dataset, train_size, device, run,
                                            len(train_dataset.classes))
            for k, v in metrics.items():
                run_metrics[k].append(v)

        mean_metrics = {k: np.mean(v) for k, v in run_metrics.items()}
        std_map = np.std(run_metrics['mAP']) if NUM_RUNS > 1 else 0.00

        print(f"\n[+] KẾT QUẢ TỔNG HỢP {model_name} (Over {NUM_RUNS} runs):")
        print(f"    mAP: {mean_metrics['mAP']:.2f} ± {std_map:.2f}")
        for k in ['R@1', 'R@5', 'R@10', 'mP@1', 'mP@5', 'mP@10']:
            print(f"    {k}: {mean_metrics[k]:.2f}")

        final_summary[(model_name, loss_type)] = {'means': mean_metrics, 'std_map': std_map}

    print("\n\n" + "=" * 110)
    print(f"{'Table 1: Medical Image retrieval results':^110}")
    print("=" * 110)
    print(
        f"| {'Dataset':<8} | {'Model':<16} | {'Loss':<13} | {'R@1':<6} | {'R@5':<6} | {'R@10':<6} | {'mAP':<12} | {'mP@1':<6} | {'mP@5':<6} | {'mP@10':<6} |")
    print("-" * 110)

    first_row = True
    for (model_name, loss_type), stats in final_summary.items():
        m = stats['means']
        std = stats['std_map']
        dataset_str = "Kvasir" if first_row else ""
        loss_str = loss_type if model_name in ["Densenet121", "deit_small"] else ""
        map_str = f"{m['mAP']:.2f}±{std:.2f}"

        print(
            f"| {dataset_str:<8} | {model_name:<16} | {loss_str:<13} | {m['R@1']:<6.2f} | {m['R@5']:<6.2f} | {m['R@10']:<6.2f} | {map_str:<12} | {m['mP@1']:<6.2f} | {m['mP@5']:<6.2f} | {m['mP@10']:<6.2f} |")
        first_row = False
    print("=" * 110)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()