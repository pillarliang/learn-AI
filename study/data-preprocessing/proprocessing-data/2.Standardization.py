from sklearn.preprocessing import StandardScaler

data = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

transformer = StandardScaler()
data = transformer.fit_transform(data)
print(data)

# data = [[0, 0], [0, 0], [1, 1], [1, 1]]
# scaler = StandardScaler()

# print(scaler.fit(data))
# print(scaler.mean_)
