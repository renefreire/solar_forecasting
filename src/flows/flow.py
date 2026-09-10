"""
Script: solar_forecast_mlp_lm.py
Descrição:
    - Lê arquivos CSV do INMET (estação A606 - Arraial do Cabo)
    - Filtra e limpa dados de RADIACAO GLOBAL (KJ/m²)
    - Agrega dados horários em diários
    - Treina MLP (Perceptron Multicamadas) com algoritmo Levenberg–Marquardt
    - Calcula MAE, RAE e RMSE
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 🟡 CONFIGURAÇÕES DO USUÁRIO
# =========================================================
CSV_DIR = "src/dataset"  # ⬅️ ajuste para o seu caminho local
LAGS = 14       # número de dias anteriores usados como entrada
N_HIDDEN = 20   # neurônios na camada oculta
EPOCHS = 200    # épocas de treinamento
MU_INIT = 1e-3  # parâmetro inicial do LM

# =========================================================
# 🟢 FUNÇÃO PARA LOCALIZAR TODOS OS ARQUIVOS CSV
# =========================================================
def find_csv_files(base_dir):
    csvs = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return sorted(csvs)

csv_paths = find_csv_files(CSV_DIR)
if not csv_paths:
    raise RuntimeError(f"Nenhum arquivo CSV encontrado em: {CSV_DIR}")

print(f"✅ {len(csv_paths)} arquivos CSV encontrados em {CSV_DIR}")

# =========================================================
# 🧹 LEITURA E AGREGAÇÃO DIÁRIA DOS DADOS
# =========================================================
daily_frames = []
for p in csv_paths:
    with open(p, "r", encoding="latin1", errors="replace") as f:
        lines = f.readlines()

    # localizar linha de cabeçalho
    header_idx = None
    for i, line in enumerate(lines[:80]):
        if "DATA (YYYY" in line.upper() or line.strip().upper().startswith("DATA;"):
            header_idx = i
            break
    if header_idx is None:
        continue

    content = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(content), sep=";", decimal=",", encoding="latin1", dayfirst=True, low_memory=True)
    df.columns = [c.strip() for c in df.columns]

    # identificar colunas
    date_col = next((c for c in df.columns if "DATA" in c.upper()), None)
    hour_col = next((c for c in df.columns if "HORA" in c.upper()), None)
    rad_col = next((c for c in df.columns if "RADIACAO" in c.upper()), None)
    if not rad_col:
        continue

    # tratar e converter
    if hour_col and hour_col in df.columns:
        df["dt"] = pd.to_datetime(df[date_col].astype(str).str.strip() + " " + df[hour_col].astype(str).str.strip(),
                                  dayfirst=True, errors="coerce")
    else:
        df["dt"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    df[rad_col] = pd.to_numeric(df[rad_col], errors="coerce")
    df.loc[df[rad_col] <= -9000, rad_col] = np.nan

    # agregação diária
    df["d"] = df["dt"].dt.date
    daily = df.groupby("d")[rad_col].sum(min_count=1).reset_index()
    daily["date"] = pd.to_datetime(daily["d"])
    daily = daily[["date", rad_col]].rename(columns={rad_col: "radiation"}).dropna()
    daily_frames.append(daily)

if not daily_frames:
    raise RuntimeError("Nenhum dado válido encontrado nos CSVs.")

daily_all = (
    pd.concat(daily_frames, ignore_index=True)
    .groupby("date")["radiation"]
    .mean()
    .reset_index()
    .set_index("date")
    .sort_index()
)
print(f"📈 Série diária construída: {len(daily_all)} registros válidos")

# =========================================================
# 🧩 FUNÇÃO PARA GERAR FEATURES COM LAGS
# =========================================================
def make_lags(series, lags=7):
    vals = series.values
    X, y, dates = [], [], []
    for i in range(lags, len(vals)):
        w, t = vals[i - lags:i], vals[i]
        if np.any(np.isnan(w)) or np.isnan(t):
            continue
        X.append(w)
        y.append(t)
        dates.append(series.index[i])
    return np.array(X), np.array(y), np.array(dates)

X, y, dates = make_lags(daily_all["radiation"], lags=LAGS)
split = int(0.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = dates[split:]

# normalização
scX, scY = StandardScaler(), StandardScaler()
X_train_s = scX.fit_transform(X_train)
X_test_s = scX.transform(X_test)
y_train_s = scY.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_s = scY.transform(y_test.reshape(-1, 1)).ravel()

# =========================================================
# 🧠 CLASSE MLP COM TREINAMENTO LEVENBERG–MARQUARDT
# =========================================================
class MLP_LM:
    def __init__(self, n_in, n_hidden=10, mu=1e-3, epochs=100):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.mu = mu
        self.epochs = epochs
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.1, (n_hidden, n_in))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.1, (1, n_hidden))
        self.b2 = np.zeros(1)

    def _fwd(self, X):
        Z1 = X @ self.W1.T + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2.T + self.b2
        return Z1, A1, Z2.ravel()

    def predict(self, X):
        _, _, y = self._fwd(X)
        return y

    def pack(self):
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def unpack(self, v):
        idx = 0
        s1 = self.n_hidden * self.n_in
        self.W1 = v[idx:idx + s1].reshape((self.n_hidden, self.n_in))
        idx += s1
        s2 = self.n_hidden
        self.b1 = v[idx:idx + s2]
        idx += s2
        s3 = self.n_hidden * 1
        self.W2 = v[idx:idx + s3].reshape((1, self.n_hidden))
        idx += s3
        s4 = 1
        self.b2 = v[idx:idx + s4]
        idx += s4

    def jacobian(self, X):
        n = X.shape[0]
        Z1, A1, _ = self._fwd(X)
        sech2 = 1.0 - np.tanh(Z1) ** 2
        n_w = self.n_hidden * self.n_in + self.n_hidden + self.n_hidden + 1
        J = np.zeros((n, n_w))
        for i in range(n):
            base = 0
            for h in range(self.n_hidden):
                J[i, base + h * self.n_in: base + (h + 1) * self.n_in] = (self.W2[0, h] * sech2[i, h]) * X[i]
            base += self.n_hidden * self.n_in
            J[i, base: base + self.n_hidden] = self.W2[0, :] * sech2[i, :]
            base += self.n_hidden
            J[i, base: base + self.n_hidden] = A1[i, :]
            base += self.n_hidden
            J[i, base] = 1.0
        return J

    def fit(self, X, y):
        w = self.pack()
        for ep in range(self.epochs):
            y_pred = self.predict(X)
            e = y - y_pred
            sse = np.sum(e ** 2)
            J = self.jacobian(X)
            JTJ = J.T @ J
            g = J.T @ e
            I = np.eye(JTJ.shape[0])
            try:
                delta = np.linalg.solve(JTJ + self.mu * I, g)
            except np.linalg.LinAlgError:
                self.mu *= 10
                continue
            w_new = w + delta
            self.unpack(w_new)
            y_new = self.predict(X)
            sse_new = np.sum((y - y_new) ** 2)
            if sse_new < sse:
                w = w_new
                self.mu *= 0.7
            else:
                self.unpack(w)
                self.mu *= 5
        self.unpack(w)

# =========================================================
# 🚀 TREINAMENTO E AVALIAÇÃO
# =========================================================
model = MLP_LM(n_in=LAGS, n_hidden=N_HIDDEN, mu=MU_INIT, epochs=EPOCHS)
model.fit(X_train_s, y_train_s)

y_pred_train_s = model.predict(X_train_s)
y_pred_test_s = model.predict(X_test_s)
y_pred_train = scY.inverse_transform(y_pred_train_s.reshape(-1, 1)).ravel()
y_pred_test = scY.inverse_transform(y_pred_test_s.reshape(-1, 1)).ravel()

def mae(a, b): return np.mean(np.abs(a - b))
def rae(a, b): return np.sum(np.abs(a - b)) / np.sum(np.abs(a - np.mean(a)))
def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))

metrics = {
    "MAE_treino": mae(y_train, y_pred_train),
    "MAE_teste": mae(y_test, y_pred_test),
    "RAE_treino": rae(y_train, y_pred_train),
    "RAE_teste": rae(y_test, y_pred_test),
    "RMSE_treino": rmse(y_train, y_pred_train),
    "RMSE_teste": rmse(y_test, y_pred_test),
}
print("\n📊 Métricas de desempenho:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# =========================================================
# 📈 VISUALIZAÇÃO
# =========================================================
plt.figure(figsize=(10, 5))
nplot = min(200, len(y_test))
plt.plot(range(nplot), y_test[:nplot], label="Real", linewidth=2)
plt.plot(range(nplot), y_pred_test[:nplot], label="Previsto", linestyle="--")
plt.title("Previsão da Irradiação Solar - Arraial do Cabo (Teste)")
plt.xlabel("Amostra (dias)")
plt.ylabel("Radiação Global (KJ/m²)")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 1️⃣ Dispersão (Real x Previsto)
# ----------------------------
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolor='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='Ideal (y=x)')
plt.xlabel('Real (kJ/m²)')
plt.ylabel('Previsto (kJ/m²)')
plt.title('Dispersão: Irradiação Solar - Arraial do Cabo (Teste)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------
# 2️⃣ Histograma dos erros
# ----------------------------
erros = y_test - y_pred_test

plt.figure(figsize=(8,5))
plt.hist(erros, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(erros.mean(), color='red', linestyle='--', label=f'Média do erro = {erros.mean():.2f}')
plt.xlabel('Erro (Real - Previsto) [kJ/m²]')
plt.ylabel('Densidade')
plt.title('Distribuição dos Erros - Irradiação Solar (Teste)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ----------------------------
# 3️⃣ (Opcional) Estatísticas resumidas
# ----------------------------
print("Resumo dos erros:")
print(f"  Média do erro: {erros.mean():.2f}")
print(f"  Desvio-padrão: {erros.std():.2f}")
print(f"  Erro absoluto médio: {np.mean(np.abs(erros)):.2f}")
print(f"  Máximo erro positivo: {erros.max():.2f}")
print(f"  Máximo erro negativo: {erros.min():.2f}")