import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib


def show_digit(idx):
    data = pd.read_csv('./data/手写数字识别.csv')
    if idx < 0 or idx > len(data) - 1:
        return
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print('当前数字的标签为:', y[idx])

    # data 修改为 ndarray 类型
    data_ = x.iloc[idx].values
    data_ = data_.reshape(28, 28)  # modify the shape of data to 28*28
    plt.axis('off')  # disable axis labels
    plt.imshow(data_)
    plt.show()


def train_model():
    data = pd.read_csv('./data/手写数字识别.csv')
    x = data.iloc[:, 1:] / 255  # normalization
    y = data.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=0)

    # model training
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(x_train, y_train)

    # evaluate the model
    acc = knn_model.score(x_test, y_test)
    print('the accurancy of test dataset: %.2f' % acc)

    # save model
    joblib.dump(knn_model, './model/knn.pkl')


def test_model():
    # 读取图片数据
    img = plt.imread('./data/5.png')
    plt.imshow(img)
    # 加载模型
    knn = joblib.load('./model/knn.pkl')
    y_pred = knn.predict(img.reshape(1, -1))
    print('The digit you drawn was:', y_pred)


if __name__ == '__main__':
    show_digit(1)
    train_model()
    test_model()
