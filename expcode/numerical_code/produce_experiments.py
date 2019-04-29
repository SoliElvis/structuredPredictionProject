import numpy as np
from sklearn.datasets import load_digits
from pystruct.models import MultiClassClf
from pystruct.learners import FrankWolfeSSVM 


digits = load_digits()
X_train, y_train = digits.data, digits.target
X_train = X_train / 16.
y_train = y_train.astype(np.int)
model = MultiClassClf()

bcfw = FrankWolfeSSVM(model=model, C=.1, max_iter=1000, tol=0.1, verbose=3, check_dual_every=10)
batch_fw = FrankWolfeSSVM(model=model, C=.1, max_iter=1000, batch_mode=True,tol=0.1, verbose=3, check_dual_every=10)

bcfw.fit(X_train, y_train)
batch_fw.fit(X_train, y_train)

itr = [i*10 for i in range(0,12)]
plt.plot(list(itr), list(d_gapBCFW), 'go-', label='line 1', linewidth=2)
plt.title('BCFW duality gap at each 10 timesteps')
plt.xlabel('t')
plt.ylabel('d_t')
plot = plt
plt.savefig('d_gapBCFW1.png')

plt.plot(list(itr), list(d_gapBatchFW), 'go-', label='line 1', linewidth=2)
plt.title('Batch FW duality gap at each 10 timesteps')
plt.xlabel('t')
plt.ylabel('d_t')
plot = plt
plt.savefig('d_gapBatchFW2.png')
