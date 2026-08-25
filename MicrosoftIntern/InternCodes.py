import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# 1. AYARLAR
# ============================================================

FILE_PATH = r"C:\Users\Mert Can Yücedağ\Desktop\MicrosoftIntern\Veriseti\Steel_Industry_data.csv"

FORECAST_MONTHS = 1


# ============================================================
# 2. VERİYİ OKU
# ============================================================

print("=" * 70)
print("VERİ OKUNUYOR")
print("=" * 70)

df = pd.read_csv(FILE_PATH)

print("Veri boyutu:", df.shape)

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True
)

df = df.sort_values("date").reset_index(drop=True)

print("Başlangıç:", df["date"].min())
print("Bitiş    :", df["date"].max())


# ============================================================
# 3. TIME FEATURES
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
# 4. CYCLIC FEATURES
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
# 5. LOAD TYPE ENCODING
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


# Güvenlik: bütün beklenen sütunlar mevcut olsun

for col in [
    "Load_Light_Load",
    "Load_Medium_Load",
    "Load_Maximum_Load"
]:
    if col not in df.columns:
        df[col] = 0


# ============================================================
# 6. BASIC LAGS
# ============================================================

df["usage_lag_1"] = (
    df["Usage_kWh"].shift(1)
)

df["usage_lag_2"] = (
    df["Usage_kWh"].shift(2)
)

df["usage_lag_4"] = (
    df["Usage_kWh"].shift(4)
)

df["usage_lag_8"] = (
    df["Usage_kWh"].shift(8)
)

df["usage_lag_96"] = (
    df["Usage_kWh"].shift(96)
)

df["usage_lag_672"] = (
    df["Usage_kWh"].shift(672)
)


# ============================================================
# 7. ROLLING FEATURES
# ============================================================

previous_usage = (
    df["Usage_kWh"].shift(1)
)


# Son 1 saat

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


# Son 2 saat

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


# Son 24 saat

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
# 8. RAMP FEATURES
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
# 9. PREVIOUS DAY
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
# 10. PREVIOUS WEEK
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
# 11. TARGET
# ============================================================

df["target"] = (
    df["Usage_kWh"].shift(-1)
)


# ============================================================
# 12. DROP NaN
# ============================================================

df = (
    df
    .dropna()
    .reset_index(drop=True)
)


# ============================================================
# 13. FEATURES
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
# 14. TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(df) * 0.8
)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()


X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]


print()
print("=" * 70)
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
# 15. XGBOOST V6
# ============================================================

model = XGBRegressor(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


print()
print("=" * 70)
print("XGBOOST V6 EĞİTİLİYOR")
print("=" * 70)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 16. VALIDATION PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 17. METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print()
print("=" * 70)
print("XGBOOST V6 VALIDATION RESULTS")
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
# 18. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        model.feature_importances_

})

importance = (
    importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 19. VALIDATION ERROR ANALYSIS
# ============================================================

error_df = test[
    [
        "date",
        "target",
        "hour",
        "is_weekend",
        "Load_Type"
    ]
].copy()

error_df["prediction"] = predictions

error_df["error"] = (
    error_df["target"]
    - error_df["prediction"]
)

error_df["absolute_error"] = (
    error_df["error"].abs()
)


# ============================================================
# 20. HOUR BASED MAE
# ============================================================

print()
print("=" * 70)
print("HOUR BASED MAE")
print("=" * 70)

hour_mae = (
    error_df
    .groupby("hour")["absolute_error"]
    .mean()
)

print(
    hour_mae.to_string()
)


# ============================================================
# 21. WEEKDAY / WEEKEND MAE
# ============================================================

print()
print("=" * 70)
print("WEEKDAY / WEEKEND MAE")
print("=" * 70)

week_mae = (
    error_df
    .groupby("is_weekend")["absolute_error"]
    .mean()
)

print(
    week_mae.to_string()
)


# ============================================================
# 22. LOAD TYPE MAE
# ============================================================

print()
print("=" * 70)
print("LOAD TYPE MAE")
print("=" * 70)

load_mae = (
    error_df
    .groupby("Load_Type")["absolute_error"]
    .mean()
)

print(
    load_mae.to_string()
)


# ============================================================
# 23. WORST 20 PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("WORST 20 PREDICTIONS")
print("=" * 70)

worst = (
    error_df
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


# ################################################################
# ################################################################
#
#             FINAL MODEL + 1 AYLIK GELECEK TAHMİNİ
#
# ################################################################
# ################################################################


# ============================================================
# 24. FINAL MODEL
# ============================================================
#
# Validation tamamlandıktan sonra model,
# geçmişteki bütün kullanılabilir verilerle yeniden eğitilir.
#
# Böylece gelecek tahmininde mümkün olan en fazla
# tarihsel bilgi kullanılmış olur.
# ============================================================

print()
print("=" * 70)
print("FINAL V6 MODELİ EĞİTİLİYOR")
print("=" * 70)


final_model = XGBRegressor(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


final_model.fit(
    df[features],
    df["target"]
)


# ============================================================
# 25. ORİJİNAL VERİYİ TEKRAR HAZIRLA
# ============================================================
#
# Gelecek tahmininde Usage_kWh geçmişinin tamamına
# ihtiyacımız var.
# ============================================================

raw_df = pd.read_csv(
    FILE_PATH
)

raw_df["date"] = pd.to_datetime(
    raw_df["date"],
    dayfirst=True
)

raw_df = (
    raw_df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 26. GELECEK TARİHLERİ OLUŞTUR
# ============================================================

last_date = raw_df["date"].max()

forecast_start = (
    last_date
    + pd.Timedelta(minutes=15)
)

forecast_end = (
    forecast_start
    + pd.DateOffset(months=FORECAST_MONTHS)
    - pd.Timedelta(minutes=15)
)

future_dates = pd.date_range(
    start=forecast_start,
    end=forecast_end,
    freq="15min"
)


print()
print("=" * 70)
print("GELECEK TAHMİN DÖNEMİ")
print("=" * 70)

print(
    "Son gerçek veri:",
    last_date
)

print(
    "Tahmin başlangıcı:",
    forecast_start
)

print(
    "Tahmin bitişi:",
    forecast_end
)

print(
    "Tahmin noktası:",
    len(future_dates)
)


# ============================================================
# 27. RECURSIVE FORECAST HAZIRLIĞI
# ============================================================
#
# Burada model bir sonraki 15 dakikayı tahmin eder.
#
# Tahmin edilen değer daha sonra Usage geçmişine eklenir
# ve bir sonraki tahminde lag olarak kullanılır.
#
# Böylece:
#
# t+1 → tahmin
# t+2 → t+1 tahmini kullanılır
# t+3 → t+1 ve t+2 tahminleri kullanılır
# ...
#
# ============================================================

history_usage = list(
    raw_df["Usage_kWh"].astype(float)
)


# ============================================================
# 28. ELECTRICAL FEATURES
# ============================================================
#
# Gelecekte gerçek elektriksel ölçümler henüz mevcut olmadığı
# için modelin eğitiminde kullanılan son bilinen değerler
# gelecek dönem için başlangıç referansı olarak kullanılır.
#
# Bu durum raporda ayrıca açıklanabilir.
# ============================================================

electrical_columns = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor"
]

last_electrical = {}

for col in electrical_columns:

    last_electrical[col] = (
        raw_df[col]
        .iloc[-1]
    )


# ============================================================
# 29. LOAD TYPE İÇİN TARİHSEL SAAT BAZLI MOD
# ============================================================

raw_df["hour_temp"] = (
    raw_df["date"].dt.hour
)

raw_df["dow_temp"] = (
    raw_df["date"].dt.dayofweek
)


def get_load_type(hour, dow):

    subset = raw_df[
        (raw_df["hour_temp"] == hour) &
        (raw_df["dow_temp"] == dow)
    ]

    if len(subset) == 0:

        subset = raw_df[
            raw_df["hour_temp"] == hour
        ]

    if len(subset) == 0:

        return raw_df["Load_Type"].mode()[0]

    return (
        subset["Load_Type"]
        .mode()[0]
    )


# ============================================================
# 30. FUTURE FEATURE OLUŞTURMA FONKSİYONU
# ============================================================

def create_future_features(
    current_date,
    history
):

    row = {}

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    hour = current_date.hour
    minute = current_date.minute

    day = current_date.day
    month = current_date.month

    dow = current_date.dayofweek

    row["hour"] = hour
    row["minute"] = minute
    row["day"] = day
    row["month"] = month
    row["day_of_week_num"] = dow

    row["is_weekend"] = int(
        dow >= 5
    )


    # --------------------------------------------------------
    # Cyclic
    # --------------------------------------------------------

    minutes_day = (
        hour * 60 +
        minute
    )

    row["hour_sin"] = np.sin(
        2 * np.pi *
        minutes_day /
        1440
    )

    row["hour_cos"] = np.cos(
        2 * np.pi *
        minutes_day /
        1440
    )

    row["dow_sin"] = np.sin(
        2 * np.pi *
        dow /
        7
    )

    row["dow_cos"] = np.cos(
        2 * np.pi *
        dow /
        7
    )


    # --------------------------------------------------------
    # Load Type
    # --------------------------------------------------------

    load_type = get_load_type(
        hour,
        dow
    )

    row["Load_Light_Load"] = int(
        load_type == "Light_Load"
    )

    row["Load_Medium_Load"] = int(
        load_type == "Medium_Load"
    )

    row["Load_Maximum_Load"] = int(
        load_type == "Maximum_Load"
    )


    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    row["usage_lag_1"] = history[-1]

    row["usage_lag_2"] = history[-2]

    row["usage_lag_4"] = history[-4]

    row["usage_lag_8"] = history[-8]

    row["usage_lag_96"] = history[-96]

    row["usage_lag_672"] = history[-672]


    # --------------------------------------------------------
    # ROLLING
    # --------------------------------------------------------

    last_4 = np.array(
        history[-4:],
        dtype=float
    )

    last_8 = np.array(
        history[-8:],
        dtype=float
    )

    last_96 = np.array(
        history[-96:],
        dtype=float
    )


    row["rolling_mean_4"] = (
        np.mean(last_4)
    )

    row["rolling_std_4"] = (
        np.std(
            last_4,
            ddof=1
        )
        if len(last_4) > 1
        else 0
    )

    row["rolling_max_4"] = (
        np.max(last_4)
    )

    row["rolling_min_4"] = (
        np.min(last_4)
    )


    row["rolling_mean_8"] = (
        np.mean(last_8)
    )

    row["rolling_std_8"] = (
        np.std(
            last_8,
            ddof=1
        )
        if len(last_8) > 1
        else 0
    )

    row["rolling_max_8"] = (
        np.max(last_8)
    )


    row["rolling_mean_96"] = (
        np.mean(last_96)
    )

    row["rolling_std_96"] = (
        np.std(
            last_96,
            ddof=1
        )
        if len(last_96) > 1
        else 0
    )


    # --------------------------------------------------------
    # RAMP
    # --------------------------------------------------------

    row["ramp_1"] = (
        row["usage_lag_1"]
        -
        row["usage_lag_2"]
    )

    row["ramp_4"] = (
        row["usage_lag_1"]
        -
        row["usage_lag_4"]
    )

    row["ramp_8"] = (
        row["usage_lag_1"]
        -
        row["usage_lag_8"]
    )

    row["abs_ramp_1"] = abs(
        row["ramp_1"]
    )


    # --------------------------------------------------------
    # PREVIOUS DAY
    # --------------------------------------------------------

    day_values = np.array(
        history[-100:-96],
        dtype=float
    )

    if len(day_values) == 4:

        row["previous_day_mean"] = (
            np.mean(day_values)
        )

        row["previous_day_std"] = (
            np.std(
                day_values,
                ddof=1
            )
        )

    else:

        row["previous_day_mean"] = (
            row["usage_lag_96"]
        )

        row["previous_day_std"] = 0


    # --------------------------------------------------------
    # PREVIOUS WEEK
    # --------------------------------------------------------

    week_values = np.array(
        history[-676:-672],
        dtype=float
    )

    if len(week_values) == 4:

        row["previous_week_mean"] = (
            np.mean(week_values)
        )

        row["previous_week_std"] = (
            np.std(
                week_values,
                ddof=1
            )
        )

    else:

        row["previous_week_mean"] = (
            row["usage_lag_672"]
        )

        row["previous_week_std"] = 0


    # --------------------------------------------------------
    # ELECTRICAL
    # --------------------------------------------------------

    for col in electrical_columns:

        row[col] = (
            last_electrical[col]
        )


    return pd.DataFrame(
        [row],
        columns=features
    )


# ============================================================
# 31. RECURSIVE 1 AYLIK TAHMİN
# ============================================================

print()
print("=" * 70)
print("1 AYLIK GELECEK ENERJİ TAHMİNİ BAŞLIYOR")
print("=" * 70)


forecast_results = []


for i, current_date in enumerate(
    future_dates
):

    X_future = create_future_features(
        current_date,
        history_usage
    )


    prediction = final_model.predict(
        X_future
    )[0]


    # Negatif enerji tüketimi mümkün olmadığı için
    # negatif tahminler sıfıra çekilir.

    prediction = max(
        0,
        float(prediction)
    )


    forecast_results.append({

        "date": current_date,

        "predicted_Usage_kWh":
            prediction

    })


    # Recursive kullanım
    history_usage.append(
        prediction
    )


    # İlerleme göstergesi

    if (
        (i + 1) % 500 == 0
        or
        i == len(future_dates) - 1
    ):

        print(
            f"Tahmin ilerlemesi: "
            f"{i + 1} / "
            f"{len(future_dates)}"
        )


# ============================================================
# 32. TAHMİN DATAFRAME
# ============================================================

forecast_df = pd.DataFrame(
    forecast_results
)

forecast_df["month"] = (
    forecast_df["date"].dt.month
)

forecast_df["hour"] = (
    forecast_df["date"].dt.hour
)

forecast_df["day"] = (
    forecast_df["date"].dt.date
)

forecast_df["weekday"] = (
    forecast_df["date"]
    .dt.day_name()
)


# ============================================================
# 33. AYLIK ENERJİ RAPORU
# ============================================================

monthly_report = (
    forecast_df
    .groupby("month")[
        "predicted_Usage_kWh"
    ]
    .agg([
        ("total_kWh", "sum"),
        ("average_kWh", "mean"),
        ("max_kWh", "max"),
        ("min_kWh", "min")
    ])
)


print()
print("=" * 70)
print("1 AYLIK ENERJİ TAHMİNİ")
print("=" * 70)

print(
    monthly_report
    .round(2)
    .to_string()
)


# ============================================================
# 34. GENEL ENERJİ RAPORU
# ============================================================

total_energy = (
    forecast_df[
        "predicted_Usage_kWh"
    ].sum()
)

average_energy = (
    forecast_df[
        "predicted_Usage_kWh"
    ].mean()
)

peak_index = (
    forecast_df[
        "predicted_Usage_kWh"
    ].idxmax()
)

peak_value = (
    forecast_df.loc[
        peak_index,
        "predicted_Usage_kWh"
    ]
)

peak_time = (
    forecast_df.loc[
        peak_index,
        "date"
    ]
)


print()
print("=" * 70)
print("GELECEK 1 AY ENERJİ RAPORU")
print("=" * 70)

print(
    f"Toplam tahmini tüketim : "
    f"{total_energy:,.2f} kWh"
)

print(
    f"Ortalama 15 dk         : "
    f"{average_energy:.2f} kWh"
)

print(
    f"Peak tüketim           : "
    f"{peak_value:.2f} kWh"
)

print(
    f"Peak zamanı            : "
    f"{peak_time}"
)


# ============================================================
# 35. EN YÜKSEK 20 TAHMİN
# ============================================================

print()
print("=" * 70)
print("EN YÜKSEK 20 GELECEK TÜKETİM TAHMİNİ")
print("=" * 70)

top20 = (
    forecast_df[
        [
            "date",
            "predicted_Usage_kWh",
            "hour"
        ]
    ]
    .sort_values(
        "predicted_Usage_kWh",
        ascending=False
    )
    .head(20)
)

print(
    top20.to_string(
        index=False
    )
)


# ============================================================
# 36. GÜNLÜK TÜKETİM
# ============================================================

daily_consumption = (
    forecast_df
    .groupby("day")[
        "predicted_Usage_kWh"
    ]
    .sum()
)


print()
print("=" * 70)
print("GÜNLÜK TÜKETİM")
print("=" * 70)

print(
    f"En yüksek günlük tüketim : "
    f"{daily_consumption.max():,.2f} kWh"
)

print(
    f"En düşük günlük tüketim  : "
    f"{daily_consumption.min():,.2f} kWh"
)

print(
    f"Ortalama günlük tüketim  : "
    f"{daily_consumption.mean():,.2f} kWh"
)


# ============================================================
# 37. EN YÜKSEK 10 TÜKETİM GÜNÜ
# ============================================================

print()
print("=" * 70)
print("EN YÜKSEK 10 TÜKETİM GÜNÜ")
print("=" * 70)

top10_days = (
    daily_consumption
    .sort_values(
        ascending=False
    )
    .head(10)
)

print(
    top10_days
    .round(2)
    .to_string()
)


# ============================================================
# 38. HAFTANIN GÜNLERİNE GÖRE ORTALAMA
# ============================================================

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_consumption = (
    forecast_df
    .groupby("weekday")[
        "predicted_Usage_kWh"
    ]
    .mean()
    .reindex(
        weekday_order
    )
)


print()
print("=" * 70)
print("HAFTANIN GÜNLERİNE GÖRE ORTALAMA")
print("=" * 70)

print(
    weekday_consumption
    .round(2)
    .to_string()
)


# ============================================================
# 39. SAATLERE GÖRE ORTALAMA
# ============================================================

hour_consumption = (
    forecast_df
    .groupby("hour")[
        "predicted_Usage_kWh"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)


print()
print("=" * 70)
print("SAATLERE GÖRE ORTALAMA TÜKETİM")
print("=" * 70)

print(
    hour_consumption
    .head(10)
    .round(2)
    .to_string()
)


# ============================================================
# 40. CSV KAYDET
# ============================================================

output_path = (
    r"C:\Users\Mert Can Yücedağ"
    r"\Desktop\MicrosoftIntern"
    r"\future_1_month_energy_forecast.csv"
)

forecast_df[
    [
        "date",
        "predicted_Usage_kWh",
        "month",
        "hour"
    ]
].to_csv(
    output_path,
    index=False
)


# ============================================================
# 41. MODELİ KAYDET
# ============================================================

model_path = (
    r"C:\Users\Mert Can Yücedağ"
    r"\Desktop\MicrosoftIntern"
    r"\xgboost_v6_final.json"
)

final_model.save_model(
    model_path
)


# ============================================================
# 42. SONUÇ
# ============================================================

print()
print("=" * 70)
print("TAMAMLANDI")
print("=" * 70)

print(
    "Validation R² : "
    f"{r2:.3f}"
)

print(
    "Validation MAE : "
    f"{mae:.3f} kWh"
)

print(
    "Validation RMSE : "
    f"{rmse:.3f} kWh"
)

print()

print(
    "Tahmin dosyası:"
)

print(
    output_path
)

print()

print(
    "Final model:"
)

print(
    model_path
)

print()
print("=" * 70)
print("PROJE TAMAMLANDI")
print("=" * 70)