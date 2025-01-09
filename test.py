import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 1. 定义BERT嵌入提取器
# ------------------------
class BertEmbeddingExtractor:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 取[CLS]的嵌入
        return embeddings.cpu().detach().numpy()

# ------------------------
# 2. 文本聚类
# ------------------------
def cluster_texts(texts, num_clusters=3):
    bert_extractor = BertEmbeddingExtractor()
    embeddings = [bert_extractor.get_embedding(text) for text in texts]
    embeddings = np.vstack(embeddings)
    
    # 归一化嵌入向量
    embeddings = normalize(embeddings)
    
    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
    cluster_labels = kmeans.labels_
    return cluster_labels, embeddings

# ------------------------
# 3. 支持/反对/中立 分类任务
# ------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

def train_sentiment_classifier(train_texts, train_labels, val_texts, val_labels, model_name="bert-base-uncased", num_epochs=3, batch_size=16, lr=2e-5):
    # 加载BERT分类模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

    # 数据加载
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_loader)}")

        # 验证模式
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == inputs['labels']).sum().item()
                total += inputs['labels'].size(0)
        print(f"Validation Accuracy: {correct/total:.4f}")

    return model, tokenizer

# ------------------------
# 4. 综合处理
# ------------------------
def main():
    # 示例数据
    texts = [
        "疫苗接种对社会有重要意义。",
        "我认为疫苗是不安全的。",
        "支持疫苗接种的政策。",
        "反对强制接种疫苗。",
        "疫苗有助于控制疫情。",
        "疫苗可能会有副作用。"
    ]

    # ---------------- 聚类 ----------------
    print("1. 聚类结果：")
    num_clusters = 2
    cluster_labels, embeddings = cluster_texts(texts, num_clusters)
    clustered_texts = {i: [] for i in range(num_clusters)}
    for text, label in zip(texts, cluster_labels):
        clustered_texts[label].append(text)
    for cluster, cluster_texts in clustered_texts.items():
        print(f"Cluster {cluster}: {cluster_texts}")

    # ---------------- 支持/反对/中立分类 ----------------
    print("\n2. 观点分类：")
    labels = [0, 1, 0, 1, 0, 2]  # 示例标签：0=支持，1=反对，2=中立
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 训练观点分类器
    model, tokenizer = train_sentiment_classifier(train_texts, train_labels, val_texts, val_labels)

    # 预测新数据
    test_texts = ["我完全支持疫苗接种！", "疫苗是不安全的，我反对接种。", "疫苗政策需要更多讨论。"]
    test_dataset = SentimentDataset(test_texts, [0]*len(test_texts), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            print(f"文本: {test_texts.pop(0)}")
            print(f"支持: {probs[0][0]:.4f}, 反对: {probs[0][1]:.4f}, 中立: {probs[0][2]:.4f}")

if __name__ == "__main__":
    main()
