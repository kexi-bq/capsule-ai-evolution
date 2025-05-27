import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

# Загрузка данных
df = pd.read_csv("Semantic_Test_Dataset.csv")
sentences = df['text'].tolist()
labels = df['label'].tolist()

# Кодировка меток
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Загрузка модели
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# Capsule shift (настраиваемый)
def apply_capsule_shift(embedding, dims, alpha):
    shifted = np.copy(embedding)
    for dim in dims:
        shifted[dim] += alpha
    return shifted

# Настройки
max_rounds = 500
best_acc = 0
score = 0
success_streak = 0
history = []
best_config = None

for i in range(1, max_rounds + 1):
    print(f"\n=== \U0001f9ea Раунд {i} ===")

    # Рандомные параметры
    dims = sorted(random.sample(range(len(embeddings[0])), k=10))
    alpha = round(np.random.uniform(0.1, 2.0), 2)

    # Делим на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.3, stratify=y, random_state=42
    )

    # Капсульный сдвиг test-набора
    X_test_shifted = [apply_capsule_shift(e, dims, alpha) for e in X_test]

    # Классификатор обучается на обычных эмбеддингах
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    acc_plain = accuracy_score(y_test, clf.predict(X_test))
    acc_capsule = accuracy_score(y_test, clf.predict(X_test_shifted))

    print(f"\U0001f3af Accuracy обычные (baseline): {acc_plain:.4f}")
    print(f"\U0001f3af Accuracy capsule: {acc_capsule:.4f}  | dims={dims}, alpha={alpha}")

    history.append((acc_plain, acc_capsule))

    if acc_capsule > best_acc:
        best_acc = acc_capsule
        best_config = (dims, alpha)
        print("✅ Capsule улучшило себя — +бал")
        score += 1
        success_streak += 1
    else:
        print("❌ Capsule не улучшило себя — -бал")
        score -= 1
        success_streak = 0

    print(f"\U0001f4ca Счёт AI: {score} (успешных подряд: {success_streak})")

    if success_streak >= 5:
        print("\U0001f3c1 Capsule AI стабильно побеждает. Успех.")
        break

# Финальный вывод
print("\n\u2728 Лучшая конфигурация:")
print(f"Dims: {best_config[0]}")
print(f"Alpha: {best_config[1]}")
print(f"Достигнутая точность: {best_acc:.4f}")

# График прогресса
rounds = list(range(1, len(history)+1))
plain, capsule = zip(*history)

plt.plot(rounds, plain, label="Обычные")
plt.plot(rounds, capsule, label="Capsule")
plt.xlabel("Раунд")
plt.ylabel("Точность")
plt.title("AI прогресс (Capsule vs Plain)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
