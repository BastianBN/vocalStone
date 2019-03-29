from matplotlib import pyplot as plt
ffts = [1.0, 3.1, 0.1, 9.6, 4.4, 2.1, 2.3, 2.3, 5.2, 4.2]
x,y = [], []
i=0
for v in ffts:
    x.append(i)
    x.append(i)
    x.append(i)
    y.append(0)
    y.append(v)
    y.append(0)
    i+=1
plt.plot(x, y)
plt.ylabel('amplitude')
plt.xlabel('fréquence')
plt.show()