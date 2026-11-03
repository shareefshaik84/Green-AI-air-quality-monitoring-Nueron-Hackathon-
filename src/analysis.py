def summarize_measurements(df):
    return {p: (None if df.loc[df.parameter == p, "value"].empty
                else float(df.loc[df.parameter == p, "value"].mean()))
            for p in ["pm25","pm10","no2","so2","o3","co"]}

def estimate_sources(s):
    flags = []
    pm25, pm10, no2, so2, co = (s.get(k) for k in ["pm25","pm10","no2","so2","co"])
    if no2 is not None and co is not None and no2 > 40 and co > 1:
        flags.append("Traffic/combustion may warrant investigation: NO₂ and CO are both elevated.")
    if pm10 is not None and pm25 is not None and pm10 > 100 and pm10 > pm25:
        flags.append("Coarse-particle/dust contribution may warrant investigation: PM10 is high relative to PM2.5.")
    if so2 is not None and so2 > 20:
        flags.append("Sulfur-containing fuel or industrial combustion may warrant investigation because SO₂ is elevated.")
    if pm25 is not None and pm25 > 60:
        flags.append("Fine-particle pollution is elevated; multiple sources may contribute.")
    if not flags:
        flags.append("Insufficient evidence for a useful source indicator. More pollutants, stations and meteorology are needed.")
    return flags
