# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============ Настройки ============
st.set_page_config(layout="wide", page_title="Кванты глубины — анализ", page_icon="💧")
st.title("💧 Анализ «квант глубины»: IF-ELSE / NN / Эксперт / Эталон")

st.markdown("""
Загрузите четыре Excel-файла:
- **IF-ELSE** — расчёт по условной логике  
- **NN** — расчёт нейросети  
- **Expert** — экспертная оценка  
- **Reference** — эталон  

Формат таблицы:
| well | depth | value |
|------|--------|--------|
| A1 | 1000 | 12.3 |

📘 Можно скачать пример:
""")

# ============ Пример файла ============
example_df = pd.DataFrame({
    "well": ["A1", "A1", "B2", "B2"],
    "depth": [1000, 1010, 1000, 1010],
    "value": [12.3, 13.1, 15.2, 14.9]
})
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    example_df.to_excel(writer, index=False)
buf.seek(0)
st.download_button("📥 Скачать пример Excel-файла", data=buf,
                   file_name="example_quant_depth.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============ Загрузка ============
uploaded_files = {
    "if": st.file_uploader("📂 IF-ELSE Excel", type=["xlsx", "xls"], key="if"),
    "nn": st.file_uploader("📂 NN Excel", type=["xlsx", "xls"], key="nn"),
    "expert": st.file_uploader("📂 Expert Excel", type=["xlsx", "xls"], key="exp"),
    "ref": st.file_uploader("📂 Reference Excel", type=["xlsx", "xls"], key="ref")
}

# ============ Вспомогательные ============
def try_read(file):
    try: return pd.read_excel(file)
    except: return None

def extract_number(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(",", ".")
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def normalize(df):
    if df is None: return None
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    wells = ["well", "скважина"]
    depths = ["depth", "глубина"]
    values = ["value", "значение", "квант"]
    w = next((c for c in df.columns if c in wells), None)
    d = next((c for c in df.columns if c in depths), None)
    v = next((c for c in df.columns if c in values), None)
    if not all([w, d, v]): return None
    df = df[[w, d, v]]
    df.columns = ["well", "depth", "value"]
    df["depth"] = df["depth"].apply(extract_number)
    df["value"] = df["value"].apply(extract_number)
    df.dropna(inplace=True)
    return df

def safe_mape(y_true, y_pred):
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def metrics(df, col):
    mask = df["ref"].notna() & df[col].notna()
    if mask.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    y, yhat = df.loc[mask, "ref"], df.loc[mask, col]
    return {
        "RMSE": np.sqrt(mean_squared_error(y, yhat)),
        "MAE": mean_absolute_error(y, yhat),
        "MAPE": safe_mape(y, yhat)
    }

# ============ Основная логика ============
if all(uploaded_files.values()):
    dfs = {k: normalize(try_read(v)) for k, v in uploaded_files.items()}
    if any(v is None for v in dfs.values()):
        st.error("❌ Проверьте формат таблиц (well, depth, value).")
    else:
        st.success("✅ Все файлы успешно загружены!")

        ref = dfs["ref"].rename(columns={"value": "ref"})
        merged = ref
        for k in ["if", "nn", "expert"]:
            df = dfs[k].rename(columns={"value": k})
            merged = pd.merge(merged, df, on=["well", "depth"], how="outer")

        # Общие метрики
        all_metrics = {m: metrics(merged, m) for m in ["if", "nn", "expert"]}
        st.subheader("🧮 Общие метрики")
        st.dataframe(pd.DataFrame(all_metrics).T.style.background_gradient(cmap="YlGnBu"))

        # Метрики по скважинам
        wells = merged["well"].unique()
        summary = []
        for w in wells:
            sub = merged[merged["well"] == w]
            for m in ["if", "nn", "expert"]:
                summary.append({"well": w, "method": m, **metrics(sub, m)})
        summary_df = pd.DataFrame(summary)

        # Heatmap
        st.subheader("🔥 Тепловая карта MAPE по скважинам и методам")
        pivot = summary_df.pivot(index="well", columns="method", values="MAPE")
        fig_heat = px.imshow(pivot, color_continuous_scale="YlOrRd",
                             labels=dict(x="Метод", y="Скважина", color="MAPE %"),
                             text_auto=".1f", aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)

        # 3D-график ошибок
        st.subheader("🌐 3D-график распределения ошибок по глубине и скважинам")
        mape_points = []
        for w in wells:
            sub = merged[merged["well"] == w]
            for m in ["if", "nn", "expert"]:
                mask = sub["ref"].notna() & sub[m].notna()
                if mask.any():
                    y, yhat = sub.loc[mask, "ref"], sub.loc[mask, m]
                    mape_local = np.abs((y - yhat) / np.maximum(np.abs(y), 1e-8)) * 100
                    mape_points.extend(list(zip([w]*len(mape_local), sub.loc[mask, "depth"], mape_local, [m]*len(mape_local))))

        df3d = pd.DataFrame(mape_points, columns=["well", "depth", "MAPE", "method"])
        fig3d = px.scatter_3d(df3d, x="depth", y="well", z="MAPE",
                              color="method", size="MAPE",
                              color_discrete_map={"if":"blue","nn":"green","expert":"orange"},
                              labels={"depth":"Глубина (м)", "well":"Скважина", "MAPE":"Ошибка %"},
                              height=600)
        fig3d.update_layout(scene=dict(zaxis=dict(range=[0, df3d["MAPE"].max()*1.1])),
                            template="plotly_white")
        st.plotly_chart(fig3d, use_container_width=True)

        # График по выбранной скважине
        st.subheader("📈 Графики по выбранной скважине")
        well_sel = st.selectbox("Выберите скважину:", options=wells)
        sub = merged[merged["well"] == well_sel].sort_values("depth")

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=sub["depth"], y=sub["ref"], mode="lines+markers",
                                      name="Reference", line=dict(color="black", width=3)))
        colors = {"if": "blue", "nn": "green", "expert": "orange"}
        for m in ["if", "nn", "expert"]:
            if sub[m].notna().any():
                fig_line.add_trace(go.Scatter(x=sub["depth"], y=sub[m],
                                              mode="lines+markers", name=m.upper(),
                                              line=dict(color=colors[m], dash="dot")))
        fig_line.update_xaxes(title="Глубина (м)", autorange="reversed")
        fig_line.update_yaxes(title="Значение")
        fig_line.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_line, use_container_width=True)

        # Скачать
        buf_out = io.BytesIO()
        with pd.ExcelWriter(buf_out, engine="openpyxl") as writer:
            merged.to_excel(writer, "merged", index=False)
            summary_df.to_excel(writer, "metrics_per_well", index=False)
            pivot.to_excel(writer, "heatmap", index=True)
            df3d.to_excel(writer, "3D_data", index=False)
        buf_out.seek(0)
        st.download_button("💾 Скачать отчёт (Excel)", data=buf_out,
                           file_name="quant_analysis_full.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("⬆️ Загрузите все 4 файла для начала анализа.")
