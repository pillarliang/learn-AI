from sklearn.preprocessing import MinMaxScaler

data = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]
transformer = MinMaxScaler()

data = transformer.fit_transform(data)
print(data)
