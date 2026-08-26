# ============================================================
# PYTORCH ENERGY CONSUMPTION PREDICTION
# Steel Industry Dataset
# ============================================================

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. SETTINGS
# ============================================================

FILE_PATH = r"C:\Users\Mert Can Yücedağ\Desktop\MicrosoftIntern\Veriseti\Steel_Industry_data.csv"

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("PYTORCH ENERGY PREDICTION")
print("=" * 70)

print("Device:", device)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("VERİ OKUNUYOR")
print("=" * 70)

df = pd.read_csv(FILE_PATH)

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True
)

df = df.sort_values("date").reset_index(drop=True)

print("Veri boyutu:", df.shape)
print("Başlangıç:", df["date"].min())
print("Bitiş    :", df["date"].max())


# ============================================================
# 4. TIME FEATURES
# ============================================================

df["hour"] = df["date"].dt.hour
df["minute"] = df["date"].dt.minute
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_week_num"] = df["date"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week_num"] >= 5
).astype(int)


# ============================================================
# 5. CYCLIC FEATURES
# ============================================================

minutes_of_day = (
    df["hour"] * 60 +
    df["minute"]
)

df["hour_sin"] = np.sin(
    2 * np.pi * minutes_of_day / 1440
)

df["hour_cos"] = np.cos(
    2 * np.pi * minutes_of_day / 1440
)

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week_num"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week_num"] / 7
)


# ============================================================
# 6. LOAD TYPE ENCODING
# ============================================================

load_dummies = pd.get_dummies(
    df["Load_Type"],
    prefix="Load",
    dtype=int
)

df = pd.concat(
    [df, load_dummies],
    axis=1
)


# ============================================================
# 7. BASIC LAGS
# ============================================================

df["usage_lag_1"] = df["Usage_kWh"].shift(1)
df["usage_lag_2"] = df["Usage_kWh"].shift(2)
df["usage_lag_4"] = df["Usage_kWh"].shift(4)
df["usage_lag_8"] = df["Usage_kWh"].shift(8)

df["usage_lag_96"] = df["Usage_kWh"].shift(96)
df["usage_lag_672"] = df["Usage_kWh"].shift(672)


# ============================================================
# 8. ROLLING FEATURES
# ============================================================

previous_usage = df["Usage_kWh"].shift(1)


# Last 1 hour
df["rolling_mean_4"] = (
    previous_usage
    .rolling(4)
    .mean()
)

df["rolling_std_4"] = (
    previous_usage
    .rolling(4)
    .std()
)

df["rolling_max_4"] = (
    previous_usage
    .rolling(4)
    .max()
)

df["rolling_min_4"] = (
    previous_usage
    .rolling(4)
    .min()
)


# Last 2 hours
df["rolling_mean_8"] = (
    previous_usage
    .rolling(8)
    .mean()
)

df["rolling_std_8"] = (
    previous_usage
    .rolling(8)
    .std()
)

df["rolling_max_8"] = (
    previous_usage
    .rolling(8)
    .max()
)


# Last 24 hours
df["rolling_mean_96"] = (
    previous_usage
    .rolling(96)
    .mean()
)

df["rolling_std_96"] = (
    previous_usage
    .rolling(96)
    .std()
)


# ============================================================
# 9. RAMP FEATURES
# ============================================================

df["ramp_1"] = (
    df["usage_lag_1"]
    - df["usage_lag_2"]
)

df["ramp_4"] = (
    df["usage_lag_1"]
    - df["usage_lag_4"]
)

df["ramp_8"] = (
    df["usage_lag_1"]
    - df["usage_lag_8"]
)

df["abs_ramp_1"] = (
    df["ramp_1"].abs()
)


# ============================================================
# 10. PREVIOUS DAY
# ============================================================

df["previous_day_mean"] = (
    df["Usage_kWh"]
    .shift(96)
    .rolling(4)
    .mean()
)

df["previous_day_std"] = (
    df["Usage_kWh"]
    .shift(96)
    .rolling(4)
    .std()
)


# ============================================================
# 11. PREVIOUS WEEK
# ============================================================

df["previous_week_mean"] = (
    df["Usage_kWh"]
    .shift(672)
    .rolling(4)
    .mean()
)

df["previous_week_std"] = (
    df["Usage_kWh"]
    .shift(672)
    .rolling(4)
    .std()
)


# ============================================================
# 12. TARGET
# ============================================================

df["target"] = (
    df["Usage_kWh"].shift(-1)
)


# ============================================================
# 13. DROP NaN
# ============================================================

df = df.dropna().reset_index(drop=True)


# ============================================================
# 14. FEATURES
# ============================================================

features = [

    # Time
    "hour",
    "minute",
    "day",
    "month",
    "day_of_week_num",
    "is_weekend",

    # Cyclic
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",

    # Load Type
    "Load_Light_Load",
    "Load_Medium_Load",
    "Load_Maximum_Load",

    # Lags
    "usage_lag_1",
    "usage_lag_2",
    "usage_lag_4",
    "usage_lag_8",
    "usage_lag_96",
    "usage_lag_672",

    # Rolling
    "rolling_mean_4",
    "rolling_std_4",
    "rolling_max_4",
    "rolling_min_4",

    "rolling_mean_8",
    "rolling_std_8",
    "rolling_max_8",

    "rolling_mean_96",
    "rolling_std_96",

    # Ramp
    "ramp_1",
    "ramp_4",
    "ramp_8",
    "abs_ramp_1",

    # Previous Day
    "previous_day_mean",
    "previous_day_std",

    # Previous Week
    "previous_week_mean",
    "previous_week_std",

    # Electrical
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor"
]


# ============================================================
# 15. TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(df) * 0.8
)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

X_train = train[features].values
y_train = train["target"].values

X_test = test[features].values
y_test = test["target"].values


print("\n" + "=" * 70)
print("TRAIN / TEST")
print("=" * 70)

print(
    "TRAIN:",
    train.shape,
    train["date"].min(),
    "->",
    train["date"].max()
)

print(
    "TEST :",
    test.shape,
    test["date"].min(),
    "->",
    test["date"].max()
)


# ============================================================
# 16. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 17. TORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# 18. DATA LOADER
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)


# ============================================================
# 19. PYTORCH NEURAL NETWORK
# ============================================================

class EnergyPredictor(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 128),

            nn.ReLU(),

            nn.Dropout(0.10),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.10),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


model = EnergyPredictor(
    X_train_tensor.shape[1]
).to(device)


# ============================================================
# 20. LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)


# ============================================================
# 21. TRAINING
# ============================================================

EPOCHS = 150

print("\n" + "=" * 70)
print("PYTORCH MODEL EĞİTİLİYOR")
print("=" * 70)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(train_loader)
    )

    if (
        (epoch + 1) % 10 == 0
        or epoch == 0
    ):

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} "
            f"- Loss: {average_loss:.6f}"
        )


# ============================================================
# 22. VALIDATION / TEST PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    X_test_device = X_test_tensor.to(device)

    pytorch_predictions = (
        model(X_test_device)
        .cpu()
        .numpy()
        .ravel()
    )


# ============================================================
# 23. METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    pytorch_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pytorch_predictions
    )
)

r2 = r2_score(
    y_test,
    pytorch_predictions
)


print("\n" + "=" * 70)
print("PYTORCH VALIDATION RESULTS")
print("=" * 70)

print(
    f"MAE  : {mae:.3f} kWh"
)

print(
    f"RMSE : {rmse:.3f} kWh"
)

print(
    f"R²   : {r2:.3f}"
)


# ============================================================
# 24. ACTUAL VS PREDICTION
# ============================================================

result_df = test[
    ["date", "target"]
].copy()

result_df["prediction"] = (
    pytorch_predictions
)

result_df["absolute_error"] = (
    result_df["target"]
    - result_df["prediction"]
).abs()


print("\n" + "=" * 70)
print("ACTUAL VS PREDICTION")
print("=" * 70)

print(
    result_df
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 25. WORST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("WORST 20 PREDICTIONS")
print("=" * 70)

worst = (
    result_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)

print(
    worst.to_string(
        index=False
    )
)


# ============================================================
# 26. SAVE PYTORCH MODEL
# ============================================================

MODEL_PATH = (
    r"C:\Users\Mert Can Yücedağ"
    r"\Desktop\MicrosoftIntern"
    r"\pytorch_energy_model.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_size":
            X_train_tensor.shape[1],

        "features":
            features,

        "scaler_mean":
            scaler.mean_,

        "scaler_scale":
            scaler.scale_
    },
    MODEL_PATH
)


print("\n" + "=" * 70)
print("PYTORCH MODEL KAYDEDİLDİ")
print("=" * 70)

print(
    "Model:",
    MODEL_PATH
)

print("\n" + "=" * 70)
print("PROJE TAMAMLANDI")
print("=" * 70)