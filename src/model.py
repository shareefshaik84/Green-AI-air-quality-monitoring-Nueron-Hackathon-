import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

TIMES = ["timestamp","datetime","date","time","Date","Datetime"]
PMS = ["pm25","pm2.5","PM2.5","PM25","pm_2_5"]

def train_and_predict(raw):
    t = next((c for c in TIMES if c in raw.columns), None)
    p = next((c for c in PMS if c in raw.columns), None)
    if t is None or p is None:
        raise ValueError("CSV needs a timestamp/date column and a PM2.5 column. See README.")
    df = raw[[t,p]].rename(columns={t:"timestamp",p:"pm25"}).copy()
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="coerce")
    df["pm25"] = pd.to_numeric(df.pm25, errors="coerce")
    df = df.dropna().sort_values("timestamp").groupby("timestamp",as_index=False).pm25.mean()
    if len(df) < 40:
        raise ValueError("At least 40 valid timestamp + PM2.5 rows are required.")
    df["hour"] = df.timestamp.dt.hour
    df["dow"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month
    df["lag1"] = df.pm25.shift(1)
    df["lag2"] = df.pm25.shift(2)
    df["roll3"] = df.pm25.shift(1).rolling(3).mean()
    df = df.dropna().reset_index(drop=True)
    feats = ["hour","dow","month","lag1","lag2","roll3"]
    split = int(len(df)*0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    if len(test) < 2:
        raise ValueError("Not enough rows for a chronological holdout test.")
    model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train[feats], train.pm25)
    pred = model.predict(test[feats])
    last, prev = df.iloc[-1], df.iloc[-2]
    next_time = last.timestamp + (last.timestamp-prev.timestamp)
    next_x = pd.DataFrame([{"hour":next_time.hour,"dow":next_time.dayofweek,
                            "month":next_time.month,"lag1":last.pm25,
                            "lag2":prev.pm25,"roll3":df.pm25.tail(3).mean()}])
    forecast = max(0.0, float(model.predict(next_x[feats])[0]))
    plot = pd.DataFrame({"timestamp":test.timestamp.values,"actual":test.pm25.values,"predicted":pred})
    return {"mae":float(mean_absolute_error(test.pm25,pred)),"forecast":forecast,
            "train_rows":len(train),"test_rows":len(test),"plot_df":plot}
