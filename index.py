

# Modern UI & Layout

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pytrends.request import TrendReq

st.set_page_config(page_title="Analisis Keuangan Modern", layout="wide")

# --- Menu Navigasi ---
menu = st.sidebar.radio("Menu", ["Money Tracking", "Analisis Google Trending"])

if menu == "Money Tracking":
    csv_export_url = "https://docs.google.com/spreadsheets/d/1c8WS2LmhD_deKvGioRwe0RqXfLfsH3MOeJv94uiLu_s/export?format=csv&gid=1653419658"
    # Mengambil data dari Google Sheets dan mengonversinya ke DataFrame
    df = pd.read_csv(csv_export_url)
    df["tanggal"] = pd.to_datetime(df["tanggal"], format="%m/%d/%Y, %I:%M:%S %p")

    # --- Sidebar: Filter Dinamis & SWOT/Canvas ---
    with st.sidebar:
        st.header("Filter Data")
        tanggal_min = df["tanggal"].min()
        tanggal_max = df["tanggal"].max()
        tanggal_range = st.date_input("Rentang Tanggal", [tanggal_min.date(), tanggal_max.date()])
        kategori = st.multiselect("Kategori", options=df["judul"].unique(), default=list(df["judul"].unique()))
        st.markdown("---")
        st.header("Business Model Canvas & SWOT")
        with st.expander("Business Model Canvas"):
            key_activities = st.text_area("Key Activities")
            value_propositions = st.text_area("Value Propositions")
            customer_segments = st.text_area("Customer Segments")
            channels = st.text_area("Channels")
        with st.expander("SWOT Analysis"):
            strengths = st.text_area("Strengths")
            weaknesses = st.text_area("Weaknesses")
            opportunities = st.text_area("Opportunities")
            threats = st.text_area("Threats")

    df_filtered = df[
        (df["tanggal"].dt.date >= tanggal_range[0]) &
        (df["tanggal"].dt.date <= tanggal_range[1]) &
        (df["judul"].isin(kategori))
    ]

    # --- Layout Modern ---
    st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>📊 Analisis Keuangan Modern</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pengeluaran", f"Rp {df_filtered['total harga'].sum():,}")
    with col2:
        st.metric("Jumlah Transaksi", len(df_filtered))
    with col3:
        st.metric("Rata-rata Harga Satuan", f"Rp {df_filtered['harga satuan'].mean():,.0f}" if len(df_filtered)>0 else "Rp 0")

    st.markdown("---")

    # --- Tabel Data ---
    with st.expander("Lihat Data Transaksi"):
        st.dataframe(df_filtered)

    # --- Analisis Tren & Musiman ---
    st.subheader("Tren Pengeluaran & Moving Average")
    trend_data = df_filtered.groupby(df_filtered["tanggal"].dt.date)["total harga"].sum()
    ma = trend_data.rolling(window=2).mean()
    st.line_chart(trend_data, use_container_width=True)
    st.line_chart(ma, use_container_width=True)
    st.caption("Garis biru: pengeluaran harian, garis oranye: moving average")

    # --- Pareto Analysis (80/20 Rule) ---
    st.subheader("Pareto Analysis: 80/20 Rule")
    pareto = df_filtered.groupby("judul")["total harga"].sum().sort_values(ascending=False)
    pareto_cum = pareto.cumsum()/pareto.sum()*100
    st.bar_chart(pareto, use_container_width=True)
    st.line_chart(pareto_cum, use_container_width=True)
    st.caption("Bar: pengeluaran per barang, Line: kumulatif % pengeluaran")
    if not pareto.empty:
        top_20 = pareto_cum[pareto_cum<=80].index.tolist()
        st.info(f"Barang yang menyumbang 80% pengeluaran: {', '.join(top_20)}")

    # --- Analisis Variansi Harga ---
    st.subheader("Analisis Variansi Harga Satuan")
    harga_per_tanggal = df_filtered.groupby(["judul", df_filtered["tanggal"].dt.date])["harga satuan"].mean().unstack(0)
    st.line_chart(harga_per_tanggal, use_container_width=True)
    deviasi = df_filtered.groupby("judul")["harga satuan"].std()
    st.write("Deviasi harga satuan per barang:")
    st.dataframe(deviasi)

    # --- Risk Analysis Sederhana ---
    st.subheader("Risk Analysis: Fluktuasi Harga")
    fluktuatif = deviasi[deviasi > deviasi.mean()]
    if not fluktuatif.empty:
        st.warning(f"Barang dengan harga satuan paling fluktuatif: {', '.join(fluktuatif.index)}")
    if len(df_filtered)>1:
        kenaikan = df_filtered.sort_values("tanggal").groupby("judul")["harga satuan"].apply(lambda x: x.pct_change().max())
        alert = kenaikan[kenaikan > 0.1]
        if not alert.empty:
            st.error(f"ALERT: Ada kenaikan harga satuan >10% pada: {', '.join(alert.index)}")

    # --- Analisis Per Unit / Unit Economics ---
    st.subheader("Analisis Per Unit")
    unit_summary = df_filtered.groupby("judul").agg({"jumlah": "sum", "harga satuan": "mean", "total harga": "sum"})
    unit_summary["Rata-rata per unit"] = unit_summary["total harga"] / unit_summary["jumlah"]
    st.dataframe(unit_summary)

    # --- Forecasting Sederhana ---
    st.subheader("Forecasting Pengeluaran & Harga")
    if len(trend_data) > 1:
        X = np.arange(len(trend_data)).reshape(-1,1)
        y = trend_data.values
        model = LinearRegression().fit(X, y)
        next_week = model.predict([[len(trend_data)]])[0]
        st.info(f"Prediksi pengeluaran hari berikutnya: Rp {next_week:,.0f}")
        # Forecast harga rata-rata
        harga_avg = df_filtered.groupby(df_filtered["tanggal"].dt.month)["harga satuan"].mean()
        if len(harga_avg)>1:
            Xh = np.arange(len(harga_avg)).reshape(-1,1)
            yh = harga_avg.values
            model_h = LinearRegression().fit(Xh, yh)
            next_month = model_h.predict([[len(harga_avg)]])[0]
            st.info(f"Prediksi harga rata-rata bulan depan: Rp {next_month:,.0f}")

    # --- Insight Otomatis ---
    st.subheader("Insight Otomatis & Anomali")
    if not df_filtered.empty:
        barang_terbesar = df_filtered.loc[df_filtered["total harga"].idxmax()]
        st.write(f"Barang dengan pengeluaran terbesar: **{barang_terbesar['judul']}** (Rp {barang_terbesar['total harga']:,})")
        harian = trend_data
        mingguan = df_filtered.groupby(df_filtered["tanggal"].dt.isocalendar().week)["total harga"].sum()
        bulanan = df_filtered.groupby(df_filtered["tanggal"].dt.month)["total harga"].sum()
        st.write("Jumlah pengeluaran harian:")
        st.dataframe(harian)
        st.write("Jumlah pengeluaran mingguan:")
        st.dataframe(mingguan)
        st.write("Jumlah pengeluaran bulanan:")
        st.dataframe(bulanan)
        # Anomaly detection
        rata2 = trend_data.mean()
        if (trend_data > rata2*1.2).any():
            st.warning("Terdeteksi pengeluaran di atas rata-rata!")
        for judul in df_filtered["judul"].unique():
            rata_judul = df_filtered[df_filtered["judul"]==judul]["total harga"].mean()
            minggu_ini = df_filtered[(df_filtered["judul"]==judul) & (df_filtered["tanggal"]>=pd.Timestamp.now()-pd.Timedelta(days=7))]["total harga"].sum()
            if minggu_ini > rata_judul*1.15:
                st.info(f"Pengeluaran {judul} minggu ini naik 15% dibanding rata-rata.")
    else:
        st.info("Tidak ada data sesuai filter.")

    # --- Business Model Canvas & SWOT Output ---
    st.markdown("---")
    st.header("Business Model Canvas & SWOT Output")
    canvas_col, swot_col = st.columns(2)
    with canvas_col:
        st.subheader("Business Model Canvas")
        st.write(f"**Key Activities:** {key_activities}")
        st.write(f"**Value Propositions:** {value_propositions}")
        st.write(f"**Customer Segments:** {customer_segments}")
        st.write(f"**Channels:** {channels}")
    with swot_col:
        st.subheader("SWOT Analysis")
        st.write(f"**Strengths:** {strengths}")
        st.write(f"**Weaknesses:** {weaknesses}")
        st.write(f"**Opportunities:** {opportunities}")
        st.write(f"**Threats:** {threats}")



elif menu == "Analisis Google Trending":
    st.markdown("<h1 style='text-align: center; color: #F98B4F;'>🔥 Analisis Google Trending</h1>", unsafe_allow_html=True)
    st.write("Cari dan analisis outlook/kata kunci yang sangat trending untuk rekomendasi konten.")
    pytrends = TrendReq(hl='id', tz=360)


    st.subheader("Cari Topik Trending di Indonesia")
    if st.button("Cari Topik Trending Saat Ini"):
        trending_topics = None
        error_msg = None
        # Coba Indonesia
        try:
            trending_topics = pytrends.trending_searches(pn='indonesia')
            wilayah = 'Indonesia'
        except Exception as e:
            error_msg = str(e)
        # Jika gagal, coba global
        if trending_topics is None or trending_topics.empty:
            try:
                trending_topics = pytrends.trending_searches(pn='global')
                wilayah = 'Global'
            except Exception as e:
                error_msg = str(e)
        # Jika gagal, coba united_states
        if trending_topics is None or trending_topics.empty:
            try:
                trending_topics = pytrends.trending_searches(pn='united_states')
                wilayah = 'United States'
            except Exception as e:
                error_msg = str(e)
        # Tampilkan hasil
        if trending_topics is not None and not trending_topics.empty:
            st.write(f"Topik yang sedang trending di Google Trends {wilayah}:")
            st.dataframe(trending_topics)
            st.success(f"Rekomendasi topik trending: {trending_topics.iloc[0,0]}")
        else:
            st.warning(f"Gagal mengambil data trending dari Google Trends. Error: {error_msg}")
            st.info("Coba cari saran topik/kata kunci trending berdasarkan input Anda di bawah ini.")
            search_kw_fallback = st.text_input("Cari saran topik/kata kunci trending (fallback)", "teknologi, bisnis, kesehatan")
            if search_kw_fallback:
                suggestions_fallback = pytrends.suggestions(search_kw_fallback)
                if suggestions_fallback:
                    saran_df_fallback = pd.DataFrame(suggestions_fallback)
                    st.write("Saran topik/kata kunci trending:")
                    st.dataframe(saran_df_fallback[["title", "type"]])
                    st.success(f"Rekomendasi: {saran_df_fallback.iloc[0]['title']}")
                else:
                    st.info("Tidak ada saran trending untuk input tersebut.")

    st.subheader("Cari Outlook/Kata Kunci Trending")
    search_kw = st.text_input("Cari outlook/kata kunci trending", "outlook, teknologi, bisnis, kesehatan")
    trending_keywords = []
    if search_kw:
        suggestions = pytrends.suggestions(search_kw)
        if suggestions:
            saran_df = pd.DataFrame(suggestions)
            st.write("Saran outlook/kata kunci trending:")
            st.dataframe(saran_df[["title", "type"]])
            trending_keywords = saran_df["title"].tolist()
        else:
            st.info("Tidak ada saran trending untuk kata kunci tersebut.")
            trending_keywords = [search_kw]

    st.subheader("Analisis & Rekomendasi Konten Trending")
    if st.button("Analisis Outlook Trending") and trending_keywords:
        pytrends.build_payload(trending_keywords, cat=0, timeframe='today 3-m', geo='ID', gprop='')
        trend_df = pytrends.interest_over_time()
        if not trend_df.empty:
            st.line_chart(trend_df[trending_keywords])
            st.write("Data tren 3 bulan terakhir:")
            st.dataframe(trend_df[trending_keywords])
            top_trend = trend_df[trending_keywords].mean().sort_values(ascending=False)
            st.write("Ranking outlook/kata kunci paling trending:")
            st.dataframe(top_trend)
            st.success(f"Rekomendasi konten: Fokus pada '{top_trend.index[0]}' karena sangat trending (skor rata-rata: {top_trend.iloc[0]:.2f})")
        else:
            st.warning("Tidak ada data tren untuk outlook/kata kunci tersebut.")
