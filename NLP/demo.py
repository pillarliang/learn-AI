import numpy as np

val = [30, 40, 10, 20]
a = np.array(val)
print('a', a)
b = a.argsort()
print(b)
print(type(b))

c = a[b]
print("c", c)
print(list(range(0, 5)))
