from xgboost import XGBClassifier
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import joblib  # 用于保存/加载scaler

# Load data
file_path = 'GraspClassifierModel\data\F3_duck_test_dropped.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path).dropna()

# 数据准备
features = data.drop(columns=["Label"]).values
labels = data["Label"].values

# 特征标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 将数据划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# 定义PyTorch数据集


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# 定义随机搜索的参数空间
param_grid = {
    "lr": [0.001, 0.0005, 0.0001],
    "batch_size": [16, 32, 64],
    "dropout": [0.3, 0.5, 0.7],
    "hidden_dim": [32, 64, 128]
}

# 定义模型


class SimpleNN(nn.Module):
    def __init__(self, input_dim, dropout, hidden_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# 随机搜索函数


def random_search(n_trials):
    best_accuracy = 0
    best_params = {}
    best_model_path = "model_F3_cube.pth"

    for trial in range(n_trials):
        # 随机采样超参数
        params = {
            "lr": random.choice(param_grid["lr"]),
            "batch_size": random.choice(param_grid["batch_size"]),
            "dropout": random.choice(param_grid["dropout"]),
            "hidden_dim": random.choice(param_grid["hidden_dim"]),
        }

        print(f"Trial {trial + 1}/{n_trials} with parameters: {params}")

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=params["batch_size"], shuffle=False)

        # 初始化模型、损失函数和优化器
        input_dim = features.shape[1]
        model = SimpleNN(input_dim, params["dropout"], params["hidden_dim"])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])

        # 训练模型
        epochs = 20  # 每次随机搜索训练的轮数
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                outputs = model(val_features).squeeze()
                predictions = (outputs > 0.5).float()
                correct += (predictions == val_labels).sum().item()
                total += val_labels.size(0)

        accuracy = correct / total
        print(
            f"Trial {trial + 1}/{n_trials}, Validation Accuracy: {accuracy:.4f}")

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            torch.save(model, best_model_path)

    print(f"Best parameters: {best_params}")
    print(
        f"Best model saved to {best_model_path} with accuracy {best_accuracy:.4f}")

    # 保存scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to scaler.pkl")


# 执行随机搜索
if __name__ == "__main__":
    random_search(n_trials=10)


# 加载数据
file_path = 'GraspClassifierModel\data\F3_cube_train_dropped.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path).dropna()

# 数据准备
features = data.drop(columns=["Label"]).values
labels = data["Label"].values

# 特征标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 数据集划分
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# 定义随机搜索的参数空间
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}


def random_search(n_trials):
    best_accuracy = 0.0
    best_params = {}
    best_model_path = "model_xgboost.pkl"

    for trial in range(n_trials):
        # 随机抽取参数
        params = {
            "n_estimators": random.choice(param_grid["n_estimators"]),
            "max_depth": random.choice(param_grid["max_depth"]),
            "learning_rate": random.choice(param_grid["learning_rate"]),
            "subsample": random.choice(param_grid["subsample"]),
            "colsample_bytree": random.choice(param_grid["colsample_bytree"]),
            "use_label_encoder": False,
            "eval_metric": "logloss"  # 避免警告信息
        }

        print(f"Trial {trial+1}/{n_trials}, parameters: {params}")

        # 创建模型
        model = XGBClassifier(**params)

        # 训练模型
        model.fit(X_train, y_train)

        # 验证模型
        accuracy = model.score(X_val, y_val)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            joblib.dump(model, best_model_path)

    print(f"Best parameters: {best_params}")
    print(
        f"Best model saved to {best_model_path} with accuracy {best_accuracy:.4f}")

    # 保存scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to scaler.pkl")


if __name__ == "__main__":
    random_search(n_trials=10)
