from dataclasses import dataclass
import torch from torch
import numpy as np 

@dataclass
class Parameters:
    classes: int = 5
    epochs: int = 15
    batch_size: int = 12
    learning_rate: float = 0.001


params = Parameters()


class Model(nn.ModuleList):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return out


class Preprocessing:
    # 预处理应该根据实际问题开展，比如分词，去除停用词
    def __init__(self, num_words, seq_len):
        self.file = os.path.join(os.getcwd(), 'train.csv')

    def load_split(self):
        from sklearn.model_selection import train_test_split
        df = pd.read_excel(self.file, header=None)
        tmpx = df.T.to_numpy()
        y = torch.from_numpy(y).to(dtype=torch.long)
        x = torch.from_numpy(x)
        # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
        return {'x': x, 'y': y}


class DatasetMaper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Run:
    # 考虑到k折验证和引入不同的model进行自动测试，应该把model和data作为参数送入train流程
    @staticmethod
    def train(model, data):
        train = DatasetMaper(data['x_train'], data['y_train'])
        val = DatasetMaper(data['x_val'], data['y_val'])

        loader_train = DataLoader(train, batch_size=params.batch_size)
        loader_val = DataLoader(val, batch_size=params.batch_size)

        # todo： optimizer和损失函数 或许也应该作为参数传入
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(params.epochs):
            # 模型训练，然后把预测结果保存
            model.train()
            train_predictions = torch.IntTensor([])
            running_loss = 0.0

            for x_batch, y_batch in loader_train:
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # 把每个batch预测出的结果push到train_predictions中
                train_predictions = torch.cat(
                    (train_predictions, torch.argmax(y_pred, 1).int()), 0)

            # Save predictions
            val_predictions = Run.evaluation(model, loader_val)
            train_accuary = Run.calculate_accuray(data['y_train'],
                                                  train_predictions)
            val_accuracy = Run.calculate_accuray(data['y_val'],
                                                 val_predictions)

            print(
                "Epoch: %d, loss: %.5f, Train accuracy: %.5f, Val accuracy: %.5f"
                % (epoch + 1, running_loss, train_accuary, val_accuracy))

    @staticmethod
    def evaluation(model, loader):
        model.eval()
        predictions = torch.IntTensor([])
        with torch.no_grad():
            for x_batch, y_batch in loader:
                y_pred = model(x_batch)
                predictions = torch.cat(
                    (predictions, torch.argmax(y_pred, 1).int()), 0)
        return predictions

    @staticmethod
    def calculate_accuray(labels, pred):
        # label 和 预测的结果pred进行对比，预测对的sum后除以total为准确率
        return (pred == labels).sum().float().div(pred.size(0)).item()


if __name__ == '__main__':
    Run().train()
