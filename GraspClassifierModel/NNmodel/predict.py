# import torch
# import torch.nn as nn
# import pandas as pd
# import joblib
# from sklearn.metrics import precision_score, recall_score, f1_score

# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, dropout, hidden_dim):
#         super(SimpleNN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.fc(x)

# def load_model(model_path):
#     model = torch.load(model_path)
#     model.eval()
#     return model

# if __name__ == "__main__":
#     文件路径(请根据实际情况修改)
#     test_file_path = 'GraspClassifierModel\data\F3_cube_test_dropped.csv'
#     model_path = 'model_PR2_cube_with_kfold.pth'
#     scaler_path = 'scaler_PR2_cube.pkl'  # 训练时保存的scaler

#     加载测试数据
#     test_data = pd.read_csv(test_file_path).dropna()

#     提取特征与标签（如果有标签）
#     if "Label" in test_data.columns:
#         test_features = test_data.drop(columns=["Label"]).values
#         test_labels = test_data["Label"].values
#     else:
#         test_features = test_data.values
#         test_labels = None

#     加载训练时的scaler并对测试数据进行标准化
#     scaler = joblib.load(scaler_path)
#     test_features = scaler.transform(test_features)

#     转换为Tensor
#     test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

#     加载模型
#     model = load_model(model_path)

#     预测
#     with torch.no_grad():
#         outputs = model(test_features_tensor).squeeze()
#         predictions = (outputs > 0.5).float()

#     打印预测结果
#     print("Predictions:")
#     print(predictions.numpy())

#     如果有真实标签，则计算并打印指标
#     if test_labels is not None:
#         test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
#         correct = (predictions == test_labels_tensor).sum().item()
#         total = test_labels_tensor.size(0)
#         accuracy = correct / total

#         转换为numpy方便计算指标
#         test_preds = predictions.numpy()
#         test_labels_np = test_labels

#         precision = precision_score(test_labels_np, test_preds)
#         recall = recall_score(test_labels_np, test_preds)
#         f1 = f1_score(test_labels_np, test_preds)

#         print(f"Accuracy on test set: {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1-score: {f1:.4f}")

import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

if __name__ == "__main__":
    # 文件路径(请根据实际情况修改)
    test_file_path = 'GraspClassifierModel\data\PR2_Duck_test_dropped.csv'
    model_path = 'model_xgboost.pkl'   # 已训练好的XGBoost模型
    scaler_path = 'scaler.pkl'     # 训练时保存的scaler

    # 加载测试数据
    test_data = pd.read_csv(test_file_path).dropna()

    # 提取特征与标签（如果有标签列）
    if "Label" in test_data.columns:
        test_features = test_data.drop(columns=["Label"]).values
        test_labels = test_data["Label"].values
    else:
        test_features = test_data.values
        test_labels = None

    # 加载训练时的scaler并对测试数据进行标准化
    scaler = joblib.load(scaler_path)
    test_features = scaler.transform(test_features)

    # 加载XGBoost模型
    model = joblib.load(model_path)

    # 预测
    predictions = model.predict(test_features)  # XGBoost的predict返回类别预测值

    # 打印预测结果
    print("Predictions:")
    print(predictions)

    # 如果有真实标签，则计算并打印评价指标
    if test_labels is not None:
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)

        print(f"Accuracy on test set: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
