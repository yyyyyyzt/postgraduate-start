import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

file = os.path.join(os.getcwd(), 'titanic/train.csv')
train = pd.read_csv(file)
print('实验数据大小:', train.shape)
print('信息： ', train.info())
# 年龄填充0 未知数据填充unk
train['Age'] = train['Age'].fillna(0)
train = train.fillna('unk')


def Sex(ax):
    labels = ['Male', 'Female']
    # 这里的train还是dataframe
    men = train[(train['Survived'] == 1) & (train['Sex'] == 'male')]
    women = train[(train['Survived'] == 1) & (train['Sex'] == 'female')]
    counts = [men.shape[0], women.shape[0]]
    ax.set_title('Sex - Survived')
    rects = ax.bar(labels, counts)
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    ha='center',
                    va='bottom')


def Embarked(ax):
    labels = np.unique(np.array(train['Embarked']))
    # labels = np.array(train['Embarked'].unique())
    counts = [((train['Survived'] == 1) & (train['Embarked'] == 'S')).sum(),
              ((train['Survived'] == 1) & (train['Embarked'] == 'C')).sum(),
              ((train['Survived'] == 1) & (train['Embarked'] == 'Q')).sum(),
              ((train['Survived'] == 1) & (train['Embarked'] == 'unk')).sum()]
    ax.set_title('Embarked - Survived')
    rects = ax.bar(labels, counts)
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    ha='center',
                    va='bottom')


def Pclass(ax):
    labels = np.unique(np.array(train['Pclass']))
    # labels = np.array(train['Embarked'].unique())
    counts = [((train['Survived'] == 1) & (train['Pclass'] == 1)).sum(),
              ((train['Survived'] == 1) & (train['Pclass'] == 2)).sum(),
              ((train['Survived'] == 1) & (train['Pclass'] == 3)).sum()]
    ax.set_title('Pclass - Survived')
    rects = ax.bar(labels, counts)
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    ha='center',
                    va='bottom')


def PclassAndEmbark(ax):
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        ha='center',
                        va='bottom')

    labels = ['C', 'Q', 'S']  # 不看unk的数据
    pclass1, pclass2, pclass3 = [], [], []
    for embark in labels:  # 循环 s c q unk
        pclass1.append(
            ((train['Survived'] == 1) & (train['Embarked'] == embark) &
             (train['Pclass'] == 1)).sum())
        pclass2.append(
            ((train['Survived'] == 1) & (train['Embarked'] == embark) &
             (train['Pclass'] == 2)).sum())
        pclass3.append(
            ((train['Survived'] == 1) & (train['Embarked'] == embark) &
             (train['Pclass'] == 3)).sum())
    rects1 = ax.bar([0, 4, 8], pclass1, 1, label='1st')
    rects2 = ax.bar([1, 5, 9], pclass2, 1, label='2nd')
    rects3 = ax.bar([2, 6, 10], pclass3, 1, label='3rd')
    ax.set_title('Embarked, Pclass - Survived')
    ax.set_xticks([1, 5, 9])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)


def Age(ax):
    age = train[(train['Age'] > 0) & (train['Survived'] == 1)]
    ax.set_title('Age - Survived')
    ax.hist(age['Age'], bins=30)


def Fare(ax):
    fare = train[(train['Fare'] > 0) & (train['Survived'] == 1)]
    ax.set_title('Fare - Survived')
    # age = train[(train['Fare'] > 0)]
    ax.hist(fare['Fare'], bins=50)


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 3, tight_layout=True)  # 返回多个画布，每个图用一个
    Sex(axes[0][0])
    Embarked(axes[0][1])
    Pclass(axes[0][2])
    PclassAndEmbark(axes[1][0])
    Age(axes[1][1])
    Fare(axes[1][2])
    plt.show()
    fig.tight_layout()
