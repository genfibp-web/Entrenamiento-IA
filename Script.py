# %% [markdown]
# Entrenamiento de una Red Neuronal para el Dataset Wine Quality

# %% 1. Importaciones y carga del dataset
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import numpy as np
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
import os

# Cargar y combinar los datos de vino tinto y blanco
raw = load_dataset("codesignal/wine-quality")
ds = concatenate_datasets([raw["red"], raw["white"]])
print("Total de muestras:", len(ds))

# %% 2. Preparación de los datos
feature_cols = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]
target_col = "quality"

# Convertir a arrays de NumPy
X = np.array([ds[col] for col in feature_cols], dtype=np.float32).T
y = np.array(ds[target_col], dtype=np.int64)

# Normalización Z-score
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-8
X_norm = (X - mean) / std

# Mapeo de etiquetas para que empiecen desde 0
y_shifted = y - 3
y_shifted[y_shifted >= 6] = 6 
NUM_CLASSES = 7
print(f"Clases a predecir: {NUM_CLASSES}")

# División de datos (80% entrenamiento, 20% validación)
X_train, X_val, y_train, y_val = train_test_split(
    X_norm, y_shifted, test_size=0.2, random_state=42, stratify=y_shifted
)

# %% 3. Dataset y DataLoaders
class WineQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32 
train_loader = DataLoader(WineQualityDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(WineQualityDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

# %% 4. Definición de la Arquitectura (MLP)
class WineQualityMLP(nn.Module):
    def __init__(self, input_dim=11, num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo en uso:", device)
model = WineQualityMLP(num_classes=NUM_CLASSES).to(device)

# %% 5. Ciclo de Entrenamiento
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
epochs = 300

def accuracy(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true).float().mean()

best_val_acc = 0.0

for epoch in range(epochs):
    # Entrenamiento
    model.train()
    t_loss, t_acc = 0.0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_acc += accuracy(out, yb).item()
        
    t_loss /= len(train_loader)
    t_acc /= len(train_loader)

    # Validación
    model.eval()
    v_loss, v_acc = 0.0, 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            v_loss += loss_fn(out, yb).item()
            v_acc  += accuracy(out, yb).item()
            
    v_loss /= len(val_loader)
    v_acc /= len(val_loader)

    # Guardar los pesos si mejora la precisión en validación
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        os.makedirs("modelos", exist_ok=True)
        save_file(model.state_dict(), "modelos/wine_quality_model.safetensors")

    # Mostrar progreso
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

print(f"\nMejor precision alcanzada en validacion: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print("Modelo guardado exitosamente en 'modelos/wine_quality_model.safetensors'")