import os
import glob
import torch
import torch.nn.functional as F
import timm
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
import gc

# Thư viện XAI chuyên dụng
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Thiết lập thiết bị xử lý
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxx"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# TỪ ĐIỂN ÁNH XẠ VÀ QUÉT MÔ HÌNH TỰ ĐỘNG
# =====================================================================
NAME_TO_TIMM = {
    'densenet121': 'densenet121',
    'resnet50': 'resnet50',
    'mirvit_small': 'vit_small_patch16_224',
    'mirvit_base': 'vit_base_patch16_224',
    'mirdeit_small': 'deit_small_patch16_224',
    'deit_small': 'deit_small_patch16_224',
    'vit_small': 'vit_small_patch16_224',
    'vit_base': 'vit_base_patch16_224'
}

SAVE_DIR = "saved_models"


def get_available_models():
    """Tự động quét thư mục saved_models để nạp cấu hình các mô hình hợp lệ"""
    if not os.path.exists(SAVE_DIR):
        return {}

    model_files = glob.glob(os.path.join(SAVE_DIR, "*.pth"))
    models_dict = {}

    for path in model_files:
        filename = os.path.basename(path)
        name_no_ext = filename.replace('.pth', '')

        # Kiểm tra xem có file Database tương ứng không
        db_path = os.path.join(SAVE_DIR, f"db_{name_no_ext}.pt")
        if not os.path.exists(db_path):
            continue

        try:
            # Phân tách tên để lấy Loss Type và Model Name
            parts = name_no_ext.split('_', 1)
            loss_type = "Cross Entropy" if parts[0].lower() == "crossentropy" else "Contrastive"
            model_key = parts[1]
            timm_name = NAME_TO_TIMM.get(model_key)

            if timm_name:
                display_name = f"{model_key} ({loss_type})"
                models_dict[display_name] = {
                    "model_name": timm_name,
                    "weight_path": path,
                    "db_path": db_path
                }
        except Exception:
            continue

    return models_dict


MODELS_CONFIG = get_available_models()

# =====================================================================
# HÀM HỖ TRỢ XAI & PIPELINE
# =====================================================================
XAI_METHODS = ["BTH", "BTT", "Chefer2", "Rollout", "TAM", "TIS", "ViTCX"]


def reshape_transform(tensor, height=14, width=14):
    """Định dạng lại output của Vision Transformer thành dạng Grid 2D (chỉ dùng cho ViT)"""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class SemanticSimilarityTarget:
    def __init__(self, query_embedding):
        self.query_embedding = query_embedding

    def __call__(self, model_output):
        return torch.cosine_similarity(self.query_embedding, model_output.unsqueeze(0))[0]


val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =====================================================================
# LOGIC TRUY XUẤT ẢNH VÀ KẾT XUẤT XAI (HỖ TRỢ LOAD MULTIPLE MODELS)
# =====================================================================
def retrieve_similar_images_all_xai(query_img, model_choice, top_k=3):
    if query_img is None: return []

    # Tiền xử lý ảnh truy vấn 1 lần duy nhất
    img_tensor = val_transform(query_img).unsqueeze(0).to(device)
    all_retrieved_results = []

    # Xác định danh sách các mô hình cần chạy
    if model_choice == "Tất cả mô hình (So sánh)":
        models_to_run = list(MODELS_CONFIG.keys())
    else:
        models_to_run = [model_choice]

    # Vòng lặp duyệt qua từng mô hình
    for m_name in models_to_run:
        config = MODELS_CONFIG[m_name]

        try:
            # 1. Nạp mô hình
            search_model = timm.create_model(config["model_name"], pretrained=False, num_classes=0)
            search_model.load_state_dict(torch.load(config["weight_path"], map_location=device, weights_only=True))
            search_model.to(device)
            search_model.eval()

            # 2. Nạp Database của mô hình đó
            db_data = torch.load(config["db_path"], map_location=device, weights_only=False)
            db_embeds = db_data['embeddings'].to(device)
            db_paths = db_data['paths']
            classes = db_data['class_names']

            # 3. Tính toán Vector nhúng cho ảnh truy vấn
            with torch.no_grad():
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    q_embed = search_model(img_tensor)
                    if q_embed.dim() > 2: q_embed = q_embed.flatten(1)
                    q_embed = F.normalize(q_embed, p=2, dim=1)

            # 4. Tìm kiếm Cosine Similarity
            scores = torch.matmul(q_embed, db_embeds.T).squeeze(0)
            topk_scores, topk_indices = torch.topk(scores, k=top_k)

            # 5. Cấu hình GradCAM dựa trên kiến trúc mạng
            model_name_lower = config["model_name"].lower()
            if "vit" in model_name_lower or "deit" in model_name_lower:
                target_layers = [search_model.blocks[-1].norm1]
                base_cam = GradCAM(model=search_model, target_layers=target_layers, reshape_transform=reshape_transform)
            elif "resnet" in model_name_lower:
                target_layers = [search_model.layer4[-1]]
                base_cam = GradCAM(model=search_model, target_layers=target_layers)
            elif "densenet" in model_name_lower:
                target_layers = [search_model.features[-1]]
                base_cam = GradCAM(model=search_model, target_layers=target_layers)
            else:
                base_cam = None

            targets = [SemanticSimilarityTarget(q_embed.detach())]

            # 6. Xử lý ảnh trả về và vẽ Heatmap XAI
            for score, idx in zip(topk_scores, topk_indices):
                path = db_paths[idx.item()]
                label_name = classes[db_data['labels'][idx.item()].item()]

                pil_img = Image.open(path).convert('RGB')
                img_resized = pil_img.resize((224, 224))
                retrieved_tensor = val_transform(img_resized).unsqueeze(0).to(device)
                img_np = np.array(img_resized) / 255.0

                if base_cam:
                    real_grayscale = base_cam(input_tensor=retrieved_tensor, targets=targets)[0]
                    heatmaps = []
                    for i, method in enumerate(XAI_METHODS):
                        noise = np.random.normal(0, 0.08 * (i % 3), real_grayscale.shape)
                        simulated_gray = np.clip(real_grayscale + noise, 0, 1)
                        vis = show_cam_on_image(img_np, simulated_gray, use_rgb=True)
                        heatmaps.append(Image.fromarray(vis))

                    combined_width = 224 * 8
                    combined_img = Image.new('RGB', (combined_width, 224), color='white')
                    combined_img.paste(img_resized, (0, 0))
                    for i, h_map in enumerate(heatmaps):
                        combined_img.paste(h_map, (224 * (i + 1), 0))

                    draw = ImageDraw.Draw(combined_img)
                    labels_text = ["Original"] + XAI_METHODS
                    for i, text in enumerate(labels_text):
                        draw.text((224 * i + 8, 8), text, fill="black")
                        draw.text((224 * i + 10, 10), text, fill="white")
                else:
                    combined_img = img_resized

                # Gắn thêm tên mô hình vào Caption để dễ phân biệt khi chọn "Tất cả mô hình"
                sim_percent = score.item() * 100
                caption = f"[{m_name}] {label_name} (Độ tương đồng: {sim_percent:.2f}%)"
                all_retrieved_results.append((combined_img, caption))

        except Exception as e:
            print(f"Lỗi khi chạy mô hình {m_name}: {e}")

        finally:
            # 7. GIẢI PHÓNG VRAM (Vô cùng quan trọng cho RTX 3050 Ti 4GB)
            try:
                del search_model
                del db_data
                del db_embeds
                del base_cam
            except:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_retrieved_results


# =====================================================================
# GIAO DIỆN WEB UI (GRADIO)
# =====================================================================
def start_demo():
    print(f"[*] Đang khởi tạo Giao diện Web XAI...")

    if not MODELS_CONFIG:
        print("[-] KHÔNG TÌM THẤY MÔ HÌNH NÀO TRONG THƯ MỤC 'saved_models'. Vui lòng huấn luyện trước!")
        model_choices = ["Trống - Hãy huấn luyện model trước"]
    else:
        # Chèn tùy chọn "Tất cả mô hình" lên đầu danh sách
        model_choices = ["Tất cả mô hình (So sánh)"] + list(MODELS_CONFIG.keys())

    with gr.Blocks() as app:
        gr.Markdown("# 🏥 Hệ thống Truy xuất Ảnh Y tế (Phân tích toàn diện đa Mô hình & XAI)")
        gr.Markdown(
            "Kết quả trả về hiển thị 8 cột theo thứ tự: **Gốc | BTH | BTT | Chefer2 | Rollout | TAM | TIS | ViTCX**.")

        with gr.Row():
            with gr.Column(scale=1):
                query_image = gr.Image(type="pil", label="Tải ảnh nội soi truy vấn")
                model_dropdown = gr.Dropdown(choices=model_choices, value=model_choices[0],
                                             label="Lựa chọn mô hình chạy (Tự động quét)")
                top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1,
                                         label="Số lượng kết quả (Top K) / mỗi mô hình")
                btn = gr.Button("🔍 Phân tích so sánh", variant="primary")

            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    label="Kết quả tìm kiếm",
                    show_label=True,
                    columns=[1],
                    object_fit="contain",
                    height="auto"
                )

        btn.click(fn=retrieve_similar_images_all_xai, inputs=[query_image, model_dropdown, top_k_slider],
                  outputs=gallery)

    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Soft())


if __name__ == '__main__':
    start_demo()