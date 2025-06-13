import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = torch.cos(theta + self.m)
        logits = self.s * (one_hot * output + (1.0 - one_hot) * cosine)
        return logits


class ConvNeXtArcFaceClassifier(nn.Module):
    def __init__(self, num_classes=3, feature_dim=512, model_name='convnext_tiny', device='cuda', freeze_backbone=True):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.backbone = create_model(model_name, pretrained=True, num_classes=0)
        in_dim = self.backbone.num_features
        self.projector = nn.Linear(in_dim, feature_dim)
        self.arc_margin = ArcMarginProduct(feature_dim, num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.freeze_backbone = freeze_backbone
        self.to(self.device)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.training_time = 0

    def forward(self, x, labels):
        x = self.backbone(x)
        x = self.projector(x)
        logits = self.arc_margin(x, labels)
        return logits

    def fit(self, train_loader, val_loader=None, test_loader=None, epochs=20, lr=1e-4):
        if self.freeze_backbone:
            optimizer = torch.optim.AdamW([
                {"params": self.projector.parameters()},
                {"params": self.arc_margin.parameters()}
            ], lr=lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        best_acc_info = {'epoch': -1, 'acc': 0, 'f1': 0}
        best_f1_info = {'epoch': -1, 'acc': 0, 'f1': 0}

        start = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            self.train()
            total_loss = 0

            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.forward(x_batch, y_batch)
                loss = self.ce_loss(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print( f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} ")

            # 평가 결과 저장
            if test_loader is not None:
                print(f"[Epoch {epoch + 1}] 테스트셋 평가:")
                acc, weighted_f1 = self.evaluate(test_loader, save_csv_path=None, return_summary=True)

                if acc > best_acc_info['acc']:
                    best_acc_info = {'epoch': epoch + 1, 'acc': acc, 'f1': weighted_f1}

                if weighted_f1 > best_f1_info['f1']:
                    best_f1_info = {'epoch': epoch + 1, 'acc': acc, 'f1': weighted_f1}

        self.training_time = time.time() - start
        print(f"\n⏱️ 총 학습 시간: {self.training_time:.2f}초")

        print(
            f"\n🎯 최고 정확도 에포크: {best_acc_info['epoch']} (Accuracy: {best_acc_info['acc'] * 100:.2f}%, F1: {best_acc_info['f1']:.4f})")
        print(
            f"🏆 최고 Weighted-F1 에포크: {best_f1_info['epoch']} (Accuracy: {best_f1_info['acc'] * 100:.2f}%, F1: {best_f1_info['f1']:.4f})")

    def evaluate(self, dataloader, save_csv_path=None, return_summary=False):
        self.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                x_feat = self.projector(self.backbone(x_batch))
                logits = F.linear(F.normalize(x_feat), F.normalize(self.arc_margin.weight))
                pred = torch.argmax(logits, dim=1)
                y_true.append(y_batch.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        acc = accuracy_score(y_true, y_pred)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)

        classwise_precision, classwise_recall, classwise_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0)

        print(f"\n📊 테스트 성능 요약")
        print(f"Accuracy      : {acc * 100:.2f}%")
        print(f"Macro-F1      : {macro_f1:.4f} | Precision: {macro_precision:.4f} | Recall: {macro_recall:.4f}")
        print(
            f"Weighted-F1   : {weighted_f1:.4f} | Precision: {weighted_precision:.4f} | Recall: {weighted_recall:.4f}")

        print("\n📌 클래스별 F1 성능:")
        for i, (p, r, f) in enumerate(zip(classwise_precision, classwise_recall, classwise_f1)):
            print(f" - Class {i}: Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")

        print("\n📉 Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        if save_csv_path:
            with open(save_csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['정확도(Accuracy)', 'Macro-F1', 'Weighted-F1', '학습 시간(초)'])
                writer.writerow([f"{acc:.4f}", f"{macro_f1:.4f}", f"{weighted_f1:.4f}", f"{self.training_time:.2f}"])
            print(f"📁 평가 결과가 '{save_csv_path}'에 저장되었습니다.")

        if return_summary:
            return acc, weighted_f1

        return acc



