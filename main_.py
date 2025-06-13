import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
from yc_convnext_arcface_classifier import ConvNeXtArcFaceClassifier
import pandas as pd
from scipy.stats import zscore
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from yc_preprocessing import Gray3ch

# # 이거는 시작부터 이미지로 input 넣을 때 코드
# def load_input_images(dict) -> List[np.ndarray]:
#     """
#     입력 이미지 리스트로 불러오기(나중에 웹페이지에서 불러오는걸로 바꿔야함)
#     :return: 이미지 리스트
#     """
#     image_dir = os.path.join(os.path.dirname(__file__), "test") # test 폴더에 사진들을 넣으면 된다!!!!!
#     image_list = []
#     for filename in os.listdir(image_dir):
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             path = os.path.join(image_dir, filename)
#             img = cv2.imread(path)
#             if img is not None:
#                 image_list.append(img)
#             else:
#                 print(f"이미지 로드 실패: {filename}")
#     return image_list


# 이거는 비디오 넣어서 프레임 단위로 짜르고 input 넣을 때 코드
import os
import cv2
import numpy as np
from typing import List
from rembg import remove
from PIL import Image
import io

def load_input_images(dummy=None) -> List[np.ndarray]:
    """
    입력 동영상 파일에서 프레임을 추출하고,
    시계 방향으로 90도 회전하여 반환합니다.
    """
    image_dir = os.path.join(os.path.dirname(__file__), "test")
    image_list = []
    interval = 30  # 프레임 간격

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            path = os.path.join(image_dir, filename)
            cap = cv2.VideoCapture(path)

            if not cap.isOpened():
                print(f"❌ 동영상 로드 실패: {filename}")
                continue

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    # 시계 방향 90도 회전
                    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    image_list.append(rotated)

                frame_idx += 1
            cap.release()

    # 결과 저장
    save_dir = "./debug_frames"
    os.makedirs(save_dir, exist_ok=True)

    for idx, img in enumerate(image_list):
        save_path = os.path.join(save_dir, f"frame_{idx + 1:03d}.jpg")
        cv2.imwrite(save_path, img)

    print(f"✅ 총 {len(image_list)}장의 회전된 프레임이 저장되었습니다. 경로: {save_dir}")
    return image_list




def process_image_list(images: List[np.ndarray], model_path: str) -> List[np.ndarray]:
    """
    1. Edge 기반 필터를 먼저 적용해 초기 이상치 제거
    2. 이후 남은 이미지들에 대해 feature 기반 이상치 탐지를 수행해 최종 필터링
    """
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True
    )

    intermediate_pass = []

    with ImageSegmenter.create_from_options(options) as segmenter:
        for idx, image in enumerate(images):
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=image_rgba)
                segmentation_result = segmenter.segment(mp_image)
                category_mask = segmentation_result.category_mask.numpy_view()
                hair_mask = (category_mask == 1).astype(np.uint8) * 255

                # 가장 큰 컴포넌트 유지
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask, connectivity=8)
                if num_labels <= 1:
                    continue
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                hair_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

                coords = cv2.findNonZero(hair_mask)
                if coords is None:
                    continue

                x, y, w, h = cv2.boundingRect(coords)
                cropped_img = image[y:y + h, x:x + w]
                cropped_mask = hair_mask[y:y + h, x:x + w]

                # Edge 필터 적용
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, threshold1=50, threshold2=150)
                kernel = np.ones((3, 3), np.uint8)
                dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                edge_inside_mask = cv2.bitwise_and(cropped_mask, dilated_edges)

                mask_area = np.count_nonzero(cropped_mask)
                if mask_area == 0:
                    continue

                edge_density = np.count_nonzero(edge_inside_mask) / mask_area
                if edge_density < 0.05:
                    print(f"[{idx}] Edge 필터 탈락 (edge density: {edge_density:.2%})")
                    continue

                # 통과한 이미지는 보관
                intermediate_pass.append({
                    "idx": idx,
                    "cropped_img": cropped_img,
                    "cropped_mask": cropped_mask,
                    "gray": gray
                })

            except Exception:
                continue

    # Edge 필터 통과한 이미지들에 대해 feature 기반 이상치 탐지
    features = []
    for item in intermediate_pass:
        try:
            idx = item["idx"]
            cropped_img = item["cropped_img"]
            cropped_mask = item["cropped_mask"]
            gray = item["gray"]

            h_, w_ = cropped_mask.shape
            mask_area = np.count_nonzero(cropped_mask)
            aspect_ratio = h_ / w_ if w_ != 0 else 0

            brightness_values = gray[cropped_mask > 0]
            brightness_mean = np.mean(brightness_values)
            brightness_std = np.std(brightness_values)

            mask_coords = np.column_stack(np.where(cropped_mask > 0))
            mask_center = np.mean(mask_coords, axis=0)
            img_center = np.array([h_ / 2, w_ / 2])
            center_distance = np.linalg.norm(mask_center - img_center)

            features.append({
                "idx": idx,
                "aspect_ratio": aspect_ratio,
                "brightness_mean": brightness_mean,
                "brightness_std": brightness_std,
                "center_distance": center_distance,
                "mask_area": mask_area,
                "cropped_img": cropped_img,
                "cropped_mask": cropped_mask
            })

        except Exception:
            continue

    df = pd.DataFrame(features)
    feature_cols = ["aspect_ratio", "brightness_mean", "brightness_std", "center_distance", "mask_area"]
    z_scores = np.abs(zscore(df[feature_cols]))
    outlier_rows = (z_scores > 2.5).any(axis=1)

    # 이상치 제외한 최종 RGBA 생성
    final_outputs = []
    for i, row in df[~outlier_rows].iterrows():
        cropped_img = row["cropped_img"]
        cropped_mask = row["cropped_mask"]
        cropped_rgba = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
        cropped_rgba[:, :, 3] = cropped_mask
        final_outputs.append(cropped_rgba)

    return final_outputs
img_dict = "./input"
model_path = 'hair_segmentation.tflite'
images = load_input_images(img_dict)
cropped_images = process_image_list(images, model_path)

#크롭 결과 확인
# plt.close('all')
#
# for idx, img in enumerate(cropped_images):
#     plt.figure(figsize=(4, 4))
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f'Cropped Image {idx + 1}')
#     plt.show()
#     plt.close('all')

# Transform 설정 (ConvNeXt 입력용)

transform = transforms.Compose([
    Gray3ch(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️모델 3개 각각 불러오기
model1 = ConvNeXtArcFaceClassifier(num_classes=3, feature_dim=512, model_name='convnext_tiny',device=device, freeze_backbone=False)
model1.load_state_dict(torch.load("curl_model_epoch_29.pt", map_location=device))
model1.eval()

model2 = ConvNeXtArcFaceClassifier(num_classes=3, feature_dim=512, model_name='convnext_base', device=device, freeze_backbone=False)
model2.load_state_dict(torch.load("damage_model_epoch_7.pt", map_location=device))
model2.eval()

model3 = ConvNeXtArcFaceClassifier(num_classes=3, feature_dim=512, model_name='convnext_tiny', device=device, freeze_backbone=False)
model3.load_state_dict(torch.load("width_model_epoch_28.pt", map_location=device))
model3.eval()

# 크롭된 이미지 분류
preds_model1, preds_model2, preds_model3 = [], [], []

for idx, img in enumerate(cropped_images):
    if img.shape[2] == 4:  # RGBA → RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # model1
        feat1 = model1.projector(model1.backbone(img_tensor))
        logits1 = F.linear(F.normalize(feat1), F.normalize(model1.arc_margin.weight))
        pred1 = torch.argmax(logits1, dim=1).item()
        preds_model1.append(pred1)

        # model2
        feat2 = model2.projector(model2.backbone(img_tensor))
        logits2 = F.linear(F.normalize(feat2), F.normalize(model2.arc_margin.weight))
        pred2 = torch.argmax(logits2, dim=1).item()
        preds_model2.append(pred2)

        # model3
        feat3 = model3.projector(model3.backbone(img_tensor))
        logits3 = F.linear(F.normalize(feat3), F.normalize(model3.arc_margin.weight))
        pred3 = torch.argmax(logits3, dim=1).item()
        preds_model3.append(pred3)

curl_labels = ['곱슬', '반곱슬', '직모']
damage_labels = ['건강모', '극손상모', '손상모']
width_labels = ['굵음', '보통', '얇음']

# 분류 결과 저장
data = {
    "Image": [f"image_{i+1:03d}" for i in range(len(preds_model1))],
    "Curl": [curl_labels[p] for p in preds_model1],
    "Damage": [damage_labels[p] for p in preds_model2],
    "Width": [width_labels[p] for p in preds_model3],
}

df_result = pd.DataFrame(data)
df_result.to_csv("predictions.csv", index=False, encoding='utf-8-sig')
print("예측 결과가 'predictions.csv'로 저장되었습니다.")




# # 크롭된 이미지 분류
# preds_model1, preds_model2, preds_model3 = [], [], []
#
# for idx, img in enumerate(cropped_images):
#     if img.shape[2] == 4:  # RGBA → RGB 변환
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#
#     img_tensor = transform(img).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         # model1 (Curl)
#         feat1 = model1.projector(model1.backbone(img_tensor))
#         logits1 = F.linear(F.normalize(feat1), F.normalize(model1.arc_margin.weight))
#         probs1 = F.softmax(logits1, dim=1)
#         preds_model1.append(probs1.squeeze(0).cpu().numpy())
#
#         # model2 (Damage)
#         feat2 = model2.projector(model2.backbone(img_tensor))
#         logits2 = F.linear(F.normalize(feat2), F.normalize(model2.arc_margin.weight))
#         probs2 = F.softmax(logits2, dim=1)
#         preds_model2.append(probs2.squeeze(0).cpu().numpy())
#
#         # model3 (Width)
#         feat3 = model3.projector(model3.backbone(img_tensor))
#         logits3 = F.linear(F.normalize(feat3), F.normalize(model3.arc_margin.weight))
#         probs3 = F.softmax(logits3, dim=1)
#         preds_model3.append(probs3.squeeze(0).cpu().numpy())
#
# # 클래스 레이블 정의
# curl_labels = ['곱슬', '반곱슬', '직모']
# damage_labels = ['건강모', '극손상모', '손상모']
# width_labels = ['굵음', '보통', '얇음']
#
# # 이미지별 확률 결과 저장
# row_data = []
#
# for i in range(len(cropped_images)):
#     row = {"Image": f"image_{i+1:03d}"}
#
#     # Curl 확률
#     for j, label in enumerate(curl_labels):
#         row[f"Curl_{label}"] = preds_model1[i][j]
#
#     # Damage 확률
#     for j, label in enumerate(damage_labels):
#         row[f"Damage_{label}"] = preds_model2[i][j]
#
#     # Width 확률
#     for j, label in enumerate(width_labels):
#         row[f"Width_{label}"] = preds_model3[i][j]
#
#     row_data.append(row)
#
# # DataFrame 생성 및 저장
# df_result = pd.DataFrame(row_data)
# df_result.to_csv("predictions.csv", index=False, encoding='utf-8-sig')
# print("모든 이미지에 대한 확률 예측 결과가 'predictions.csv'로 저장되었습니다.")
#
