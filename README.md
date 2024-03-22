import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Muat dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Pra-pemrosesan data
df_filtered = df[(df['Age'] <= 25) & (df['Overall'] >= 80) & (df['Potential'] >= 80)]

# Pembagian data
X = df_filtered[['Age', 'Overall', 'Potential']]
y = df_filtered['Target'] # Asumsikan 'Target' adalah kolom yang menunjukkan apakah pemain patut direkrut atau tidak
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pemilihan model
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Support Vector Machine', SVC()),
    ('Random Forest', RandomForestClassifier())
]

# Pelatihan dan evaluasi model
best_model = None
best_accuracy = 0
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy}")
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

# Prediksi
# Misalkan kita memiliki data pemain yang ingin kita prediksi
new_player = pd.DataFrame({'Age': [23], 'Overall': [85], 'Potential': [85]})
prediction = best_model.predict(new_player)
print(f"Prediksi: {prediction}")
