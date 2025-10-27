import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.constrained_layout.use'] = True

mat_contents = scipy.io.loadmat('allFaces.mat')
faces = mat_contents['faces']
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

print(mat_contents)
print("nfaces from .mat:", mat_contents['nfaces'])


y = np.zeros((faces.shape[1],)) # Creating labels
j = 0
classes = list(range(len(nfaces)))
for i in nfaces:
  y[j:j+i] = classes.pop(0)
  j = j + i

print("Total dataset size:")
print(f"n_samples: {faces.shape[1]}")
print(f"n_features: {m*n}")
print(f"n_classes: {len(nfaces)}")

allPersons = np.zeros((n * 6, m * 6))
count = 0

for j in range(6):
    for k in range(6):
        allPersons[j * n: (j + 1) * n, k * m: (k + 1) * m] = np.reshape(faces[:, np.sum(nfaces[:count])], (m, n)).T
        count += 1

img = plt.imshow(allPersons)
img.set_cmap('gray')
plt.axis('off')
plt.show()

print(f"nfaces: {nfaces}")
cumulative_sum = 0
for i, count in enumerate(nfaces):
    cumulative_sum += count
    print(f"After person {i + 1}: cumulative_sum = {cumulative_sum}")

r = [64, 62, 64, 64, 62, 64, 64, 64, 64, 64, 60, 59, 60, 63, 62, 63, 63, 64, 64, 64, 64, 64, 64, 64,
     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]

for person in range(len(r)):
    start_idx = sum(r[:person])
    end_idx = sum(r[:person + 1])
    print(f"Person {person + 1}: start_idx={start_idx}, end_idx={end_idx}")
    subset = faces[:, start_idx:end_idx]
    allFaces = np.zeros((n * 8, m * 8))

    if subset.shape[1] == 0:
        print(f"Skipping person {person + 1} due to empty subset.")
        continue

    count = 0

    for j in range(8):
        for k in range(8):
            if count < r[person]:
                allFaces[j * n:(j + 1) * n, k * m:(k + 1) * m] = np.reshape(subset[:, count], (m, n)).T
                count += 1

    img = plt.imshow(allFaces)
    img.set_cmap('gray')
    plt.axis('off')

    # Save files
    fileName = f"P{person + 1:02d}"
    # plt.savefig(f'../data/previews/{fileName}.pdf')
    plt.show()

persons = [f'Person {i + 1}' for i in range(len(r))]

plt.figure(figsize=(18, 8))
plt.bar(persons, r)
plt.title('Number of Photos per Person')
plt.xlabel('Persons')
plt.ylabel('Number of Photos')
plt.xticks(rotation=90)
plt.yticks(np.arange(0, max(r) + 1, 5))
plt.tight_layout()
plt.show()