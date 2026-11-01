"""Visualisasi persis dari Midterm_Visdat_Lanjut (3).ipynb.

Tiap blok GRAFIK mempertahankan logika Plotly dan styling notebook.
Hanya fig.show()/fig.write_html() dialihkan menjadi pengembalian Figure.
"""
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

OUTPUT = Path("outputs_story")
OUTPUT.mkdir(exist_ok=True)

DOWNLOAD_DATE = 'YYYY-MM-DD'

# ==== Thresholds ===
# BMKG Official classes
OFFICIAL_BINS   = [0, 3, 4, 5, 6, 7, 8, 10]
OFFICIAL_LABELS = ['Mikro (<3)', 'Kecil (3-3.9)', 'Ringan (4-4.9)', 'Sedang (5-5.9)',
                   'Kuat (6-6.9)', 'Besar (7-7.9)', 'Hebat (>=8)']

# Story classes (merged on purpose)
MAG_BINS   = [0, 4, 6, 10]
MAG_LABELS = ['Lemah (<4)', 'Terasa (4-5.9)', 'Kuat ke atas (>=6)']

# BMKG Official Depth classes (shallow <60, intermediate 60-300, deep >300 km)
DEPTH_BINS   = [0, 60, 300, 1000]
DEPTH_LABELS = ['Dangkal (<60 km)', 'Menengah (60-300 km)', 'Dalam (>300 km)']
SHALLOW_KM   = 60
DAMAGING_MAG = 5.0            # damage to weak buildings starts around M5 (BMKG-style classes)
DEFAULT_DEPTH_KM = 10         # suspected default depth (checked in the audit)

def load_data(csv_path=None):
    """Dataset cleaning identik dengan cell 18-23 notebook; KaggleHub sebagai fallback."""
    if csv_path is None:
        local = Path(__file__).resolve().parent / "data" / "katalog_gempa.csv"
        if local.exists():
            csv_path = local
        else:
            import kagglehub
            dataset_dir = Path(kagglehub.dataset_download("kekavigi/earthquakes-in-indonesia"))
            matches = list(dataset_dir.rglob("katalog_gempa.csv"))
            if not matches:
                raise FileNotFoundError("katalog_gempa.csv tidak ditemukan")
            csv_path = matches[0]
    df_raw = pd.read_csv(csv_path)
    DUP_KEY = ['tgl', 'ot', 'lat', 'lon']
    audit = {}
    df = df_raw.copy()
    df['datetime'] = pd.to_datetime(df['tgl'] + ' ' + df['ot'],
                                    format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
    n0 = len(df)
    df = df.dropna(subset=['datetime'])
    audit['dropped_bad_datetime'] = n0 - len(df)
    df['date'] = df['datetime'].dt.normalize()

    n0 = len(df)
    df = df.drop_duplicates(DUP_KEY, keep='first')
    audit['dropped_duplicates'] = n0 - len(df)

    focal_cols = [c for c in ['strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2'] if c in df.columns]
    df = df.drop(columns=focal_cols)

    FOREIGN = r'Philippine|Mindanao|Sulu|Caroline|Micronesia|Papua New Guinea|PNG|P\.N\.G|Malay Peninsula'
    is_foreign = df['remark'].str.contains(FOREIGN, regex=True, na=False)
    audit['dropped_foreign_label'] = int(is_foreign.sum())
    df = df[~is_foreign].reset_index(drop=True)

    df['year'] = df['datetime'].dt.year
    df['region'] = (df['remark']
                    .str.replace(r'\s*-\s*Indonesia$', '', regex=True)
                    .str.replace(' - ', ', ', regex=False)
                    .str.replace(' Region', '', regex=False)
                    .str.strip())

    df['mag_class_official'] = pd.cut(df['mag'], OFFICIAL_BINS, labels=OFFICIAL_LABELS, right=False)
    df['mag_class']          = pd.cut(df['mag'], MAG_BINS,      labels=MAG_LABELS,      right=False)
    df['depth_class']        = pd.cut(df['depth'], DEPTH_BINS,  labels=DEPTH_LABELS,    right=False)
    df['is_default_depth']   = df['depth'] == DEFAULT_DEPTH_KM

    df.attrs["raw_bounds"] = (float(df_raw["lat"].min()), float(df_raw["lat"].max()), float(df_raw["lon"].min()), float(df_raw["lon"].max()))
    df.attrs["raw_default_pct"] = round((df_raw["depth"] == DEFAULT_DEPTH_KM).mean() * 100, 1)
    return df

def create_figures(df):
    """Mengembalikan 7 Plotly Figure dari kode notebook tanpa mengubah desainnya."""
    figures = []
    lat_min, lat_max, lon_min, lon_max = df.attrs["raw_bounds"]
    df_raw = pd.DataFrame({"lat": [lat_min, lat_max], "lon": [lon_min, lon_max]})
    audit = {"depth_default_pct": df.attrs["raw_default_pct"]}
    # ==== CELL NOTEBOOK 31 ====
    facts = {}
    facts['n_events'] = len(df)
    facts['date_min'] = df['datetime'].min().date().isoformat()
    facts['date_max'] = df['datetime'].max().date().isoformat()
    facts['period_label'] = 'Nov 2008 - Jan 2023'

    # Frequency: whole catalog, worded as "recorded"
    n_days_full = (df['datetime'].max().normalize() - df['datetime'].min().normalize()).days + 1
    facts['n_days_full'] = n_days_full
    facts['events_per_day'] = round(len(df) / n_days_full, 1)
    facts['pct_days_with_recorded_event'] = round(df['datetime'].dt.normalize().nunique() / n_days_full * 100, 1)

    # Magnitude
    mc = df['mag_class'].value_counts()
    facts['mag_class_n']   = {k: int(mc[k]) for k in MAG_LABELS}
    facts['mag_class_pct'] = {k: round(float(mc[k]) / len(df) * 100, 1) for k in MAG_LABELS}
    facts['n_M5plus'] = int((df['mag'] >= 5).sum())
    facts['n_M6plus'] = int((df['mag'] >= 6).sum())
    facts['n_M7plus'] = int((df['mag'] >= 7).sum())
    facts['n_M8plus'] = int((df['mag'] >= 8).sum())   # expected 0
    top = df.loc[df['mag'].idxmax()]
    facts['strongest'] = dict(mag=float(top['mag']), date=top['datetime'].date().isoformat(),
                              depth_km=float(top['depth']), region=str(top['region']))

    # Depth (plus a robustness check without the suspected default depth)
    facts['pct_shallow'] = round(float((df['depth'] < SHALLOW_KM).mean()) * 100, 1)
    facts['pct_shallow_excl_default'] = round(float((df.loc[~df['is_default_depth'], 'depth'] < SHALLOW_KM).mean()) * 100, 1)
    facts['pct_M6plus_shallow'] = round(float((df.loc[df['mag'] >= 6, 'depth'] < SHALLOW_KM).mean()) * 100, 1)
    facts['n_shallow_damaging'] = int(((df['mag'] >= DAMAGING_MAG) & (df['depth'] < SHALLOW_KM)).sum())

    # Regions
    rc = df['region'].value_counts()
    facts['top10_share_pct'] = round(float(rc.head(10).sum()) / len(df) * 100, 1)
    facts['top3_regions'] = {k: int(v) for k, v in rc.head(3).items()}
    facts['n_regions_with_M6plus'] = int(df.loc[df['mag'] >= 6, 'region'].nunique())



    # ==== CELL NOTEBOOK 34 ====
    # === Pengaturan visual untuk semua grafik ===

    # Palet
    INK, MUTED, FAINT, GRID = "#18263C", "#60748A", "#8A9AAB", "#E7EDF2"
    TEAL, ORANGE, RED, BLUE = "#168C87", "#F0A44B", "#D65A4A", "#2A6F97"
    MAG_COLORS = dict(zip(MAG_LABELS, [TEAL, ORANGE, RED]))

    # Peta memakai latar laut gelap agar sabuk gempa "menyala"
    OCEAN, LAND, BORDER, COAST = "#0F1E30", "#1D3047", "#34506D", "#4A6A8A"
    DEPTH_COLORS = dict(zip(DEPTH_LABELS, ["#FF6B4A", "#F2C14E", "#7FB8E0"]))  # terang di latar gelap
    HEAT = ["#3B0F70", "#8C2981", "#DE4968", "#FE9F6D", "#FCFDBF"]  # magma, tanpa ujung hitam

    FONT = "Source Sans Pro, Segoe UI, Helvetica Neue, Arial, Liberation Sans, sans-serif"
    SOURCE = None  # diisi setelah fungsi angka() didefinisikan

    pio.templates["cerita"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, size=13, color=INK),
        title=dict(x=0, xref="paper", xanchor="left", font=dict(size=21, color=INK)),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#B8C4CF", tickcolor="#B8C4CF",
                   title=dict(font=dict(color=MUTED))),
        yaxis=dict(gridcolor=GRID, zeroline=False, title=dict(font=dict(color=MUTED))),
        legend=dict(font=dict(color=INK)),
        hoverlabel=dict(font=dict(family=FONT, size=13)),
        separators=",.",   # desimal koma, ribuan titik
    ))
    pio.templates.default = "plotly_white+cerita"
    px.defaults.template = "plotly_white+cerita"


    def angka(x, desimal=0):
        """Format angka gaya Indonesia: 12345.6 -> '12.345,6'."""
        s = f"{x:,.{desimal}f}"
        return s.replace(",", "_").replace(".", ",").replace("_", ".")


    SOURCE = ("Sumber: katalog gempa BMKG (Kaggle: kekavigi/earthquakes-in-indonesia), "
              f"{facts['period_label'].replace(' - ', ' – ')}; {angka(len(df))} kejadian setelah pembersihan.")


    def kerangka(fig, subjudul, tinggi=620, lebar=1000, atas=120, bawah=70, kiri=80, kanan=30,
                 catatan=None, jarak_sub=16):
        """Judul, subjudul (temuan kunci) dan catatan sumber rata kiri pada tepi yang sama.

        Margin dibuat tetap (automargin sumbu-y dimatikan) supaya posisi teks bisa dihitung
        dan tidak bergeser oleh label sumbu yang panjang.
        """
        catatan = SOURCE if catatan is None else catatan
        pad = 14  # jarak dari tepi kiri gambar
        fig.update_yaxes(automargin=False)
        fig.update_layout(height=tinggi, width=lebar,
                          margin=dict(t=atas, b=bawah, l=kiri, r=kanan),
                          title=dict(xref="container", x=pad / lebar, xanchor="left",
                                     y=1 - 14 / tinggi, yanchor="top"))
        x0 = (pad - kiri) / (lebar - kiri - kanan)  # tepi kiri gambar dalam koordinat 'paper'
        fig.add_annotation(text=subjudul, xref="paper", yref="paper", x=x0, y=1, xanchor="left",
                           yanchor="bottom", yshift=jarak_sub, showarrow=False, align="left",
                           font=dict(size=14, color=MUTED))
        fig.add_annotation(text=catatan, xref="paper", yref="paper", x=x0, y=0, xanchor="left",
                           yanchor="bottom", yshift=-bawah + 8, showarrow=False, align="left",
                           font=dict(size=11, color=FAINT))
        return fig


    def peta_gelap(fig, lat=(-12.5, 7.5), lon=(93, 143), **geo):
        """Basemap gelap yang sama untuk semua peta."""
        fig.update_geos(projection_type="mercator", lataxis_range=list(lat), lonaxis_range=list(lon),
                        showland=True, landcolor=LAND, showocean=True, oceancolor=OCEAN,
                        showcountries=True, countrycolor=BORDER, showcoastlines=True,
                        coastlinecolor=COAST, coastlinewidth=0.6, showlakes=False, showframe=False,
                        bgcolor=OCEAN, resolution=50, **geo)
        return fig


    # Batas lempeng tektonik (Bird, 2003: PB2002). Butuh internet; jika gagal, peta tetap jalan.
    PB_URL = ("https://raw.githubusercontent.com/fraxen/tectonicplates/master/"
              "GeoJSON/PB2002_boundaries.json")


    def batas_lempeng(lon_rng=(85, 150), lat_rng=(-20, 15)):
        try:
            with urllib.request.urlopen(PB_URL, timeout=20) as r:
                gj = json.load(r)
        except Exception as e:
            print("Batas lempeng tidak dimuat:", e)
            return None
        lons, lats = [], []
        for f in gj["features"]:
            g = f["geometry"]
            lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
            for line in lines:
                a = np.asarray(line)
                if (a[:, 0].max() < lon_rng[0] or a[:, 0].min() > lon_rng[1]
                        or a[:, 1].max() < lat_rng[0] or a[:, 1].min() > lat_rng[1]):
                    continue
                lons += a[:, 0].tolist() + [None]
                lats += a[:, 1].tolist() + [None]
        return go.Scattergeo(lon=lons, lat=lats, mode="lines", hoverinfo="skip",
                             line=dict(color="#9FB8D0", width=1.1, dash="dot"),
                             name="Batas lempeng (Bird, 2003)")


    def jarak_km(lat1, lon1, lat2, lon2):
        """Jarak haversine (km)."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


    PLATES = batas_lempeng()
    print("Jumlah data untuk visualisasi:", angka(len(df)))


    # ==== CELL NOTEBOOK 36 ====
    # ===== Grafik 1: peta kepadatan berbasis sel (menggantikan bubble map) =====
    CELL = 0.5
    grid = (df.assign(lat0=np.floor(df["lat"] / CELL) * CELL, lon0=np.floor(df["lon"] / CELL) * CELL)
              .groupby(["lat0", "lon0"]).size().reset_index(name="jumlah"))
    grid["id"] = grid["lat0"].map("{:.2f}".format) + "_" + grid["lon0"].map("{:.2f}".format)

    geo_grid = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": r.id, "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [r.lon0, r.lat0], [r.lon0, r.lat0 + CELL], [r.lon0 + CELL, r.lat0 + CELL],
            [r.lon0 + CELL, r.lat0], [r.lon0, r.lat0]]]}}
        for r in grid.itertuples()]}

    # 11°LS–6°LU, 94°–142°BT (3.264 sel 0,5°)
    n_cells_box = int(np.ceil((df_raw["lat"].max() - df_raw["lat"].min()) / CELL)
                      * np.ceil((df_raw["lon"].max() - df_raw["lon"].min()) / CELL))
    cum = grid["jumlah"].sort_values(ascending=False).cumsum() / len(df)
    pct_area_50 = ((cum < 0.5).sum() + 1) / n_cells_box * 100
    pct_borneo = df["remark"].str.contains("Borneo", na=False).mean() * 100

    zmax = np.log10(grid["jumlah"].max())
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=geo_grid, locations=grid["id"], featureidkey="id", z=np.log10(grid["jumlah"]),
        zmin=0, zmax=zmax, colorscale=HEAT, marker_line_width=0,
        customdata=np.c_[grid["jumlah"], grid["lat0"], grid["lon0"]],
        hovertemplate=("Sel %{customdata[1]:.1f}° s/d %{customdata[1]:.1f}°+0,5° LU/LS, "
                       "%{customdata[2]:.1f}° BT<br><b>%{customdata[0]:,.0f} kejadian</b><extra></extra>"),
        colorbar=dict(title=dict(text="Kejadian<br>per sel", font=dict(size=12, color=MUTED)),
                      tickvals=[0, 1, 2, 3], ticktext=["1", "10", "100", "1.000"],
                      thickness=12, len=0.62, y=0.5, x=0.995, xanchor="right", outlinewidth=0,
                      tickfont=dict(color=MUTED)),
    ))
    if PLATES is not None:
        fig.add_trace(PLATES)

    labels = pd.DataFrame([
        ("Sumatra", 2.2, 98.6), ("Jawa", -5.4, 110.2), ("Sulawesi", -2.6, 120.7),
        ("Laut Banda", -5.6, 127.3), ("Papua", -4.6, 138.3),
        (f"Kalimantan<br>hanya {angka(pct_borneo, 1)}% catatan", 0.6, 113.4),
    ], columns=["teks", "lat", "lon"])
    fig.add_trace(go.Scattergeo(lat=labels["lat"], lon=labels["lon"], text=labels["teks"],
                                mode="text", hoverinfo="skip", showlegend=False,
                                textfont=dict(size=12, color="#DCE6F0", family=FONT)))

    peta_gelap(fig)
    fig.update_layout(
        title="Persebaran episentrum gempa dalam katalog Indonesia",
        legend=dict(x=0.012, y=0.03, xanchor="left", yanchor="bottom", bgcolor="rgba(15,30,48,0.8)",
                    font=dict(color="#DCE6F0", size=12)),
    )
    kerangka(fig, (f"Tidak merata: separuh catatan berada di hanya {angka(pct_area_50, 1)}% sel peta, membentuk sabuk "
                   "yang mengikuti batas lempeng.<br>Setiap kotak = sel 0,5° (±55 km); "
                   "warna = jumlah kejadian pada skala logaritmik."),
             tinggi=500, lebar=1000, atas=106, bawah=30, kiri=8, kanan=8)
    fig.update_geos(domain=dict(x=[0, 0.915], y=[0, 1]))
    figures.append(fig)


    # ==== CELL NOTEBOOK 39 ====
    top_regions = df["remark"].value_counts().head(10).sort_values()
    region_plot = top_regions.rename_axis("kawasan").reset_index(name="jumlah")

    fig = px.bar(
        region_plot, x="jumlah", y="kawasan", orientation="h",
        text="jumlah", color="jumlah", color_continuous_scale="Blues",
        title="10 Kawasan dengan Kejadian Gempa Terbanyak"
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        xaxis_title="Jumlah kejadian", yaxis_title="",
        margin=dict(l=30, r=90, t=85, b=55)
    )
    figures.append(fig)


    # ==== CELL NOTEBOOK 41 ====
    mag_counts = (df["mag_class"].value_counts()
                  .reindex(MAG_LABELS, fill_value=0)
                  .rename_axis("kategori").reset_index(name="jumlah"))
    mag_counts["persen"] = mag_counts["jumlah"] / len(df) * 100
    mag_counts["label"] = [f"{angka(n)} ({angka(p, 2 if p < 1 else 1)}%)"
                           for n, p in zip(mag_counts["jumlah"], mag_counts["persen"])]
    per_tahun = mag_counts["jumlah"].iloc[2] / (facts["n_days_full"] / 365.25)

    fig = go.Figure(go.Bar(
        x=mag_counts["kategori"], y=mag_counts["jumlah"],
        marker_color=[MAG_COLORS[k] for k in mag_counts["kategori"]],
        text=mag_counts["label"], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:,.0f} kejadian<extra></extra>"))
    fig.add_annotation(x=MAG_LABELS[2], y=mag_counts["jumlah"].iloc[2], ax=0, ay=-150,
                       showarrow=True, arrowhead=2, arrowcolor=RED, font=dict(size=13, color=RED),
                       text=(f"<b>Gempa kuat (M≥6)</b><br>{angka(mag_counts['jumlah'].iloc[2])} kejadian,"
                             f"<br>±{angka(per_tahun)} per tahun"))
    fig.update_layout(title="Sebagian besar kejadian tercatat bermagnitudo rendah", showlegend=False,
                      xaxis_title="Kategori magnitudo", bargap=0.35,
                      yaxis=dict(title="Jumlah kejadian", tickformat=",d",
                                 range=[0, mag_counts["jumlah"].max() * 1.15]))
    kerangka(fig, ("Hanya ±3 dari setiap 1.000 catatan berkekuatan M≥6, tetapi jumlahnya tetap "
                   f"{angka(mag_counts['jumlah'].iloc[2])} kejadian (±{angka(per_tahun)} per tahun)."),
             tinggi=560, lebar=1000, atas=110, bawah=78, kiri=80, kanan=30)
    figures.append(fig)


    # ==== CELL NOTEBOOK 43 ====
    # ===== heatmap 2D magnitudo x kedalaman ====
    mag_edges = np.arange(0.95, 8.15, 0.2)          # tepi .x5 agar nilai 0,1-an tidak salah bin
    dep_edges = np.arange(0, 760, 10)
    H, _, _ = np.histogram2d(df["mag"], df["depth"], bins=[mag_edges, dep_edges])
    H = H.T                                          # baris = kedalaman, kolom = magnitudo
    with np.errstate(divide="ignore"):
        Z = np.where(H > 0, np.log10(H), np.nan)
    INK_SEQ = ["#E9EFF5", "#B9C9D9", "#7F98B2", "#4A6584", "#18263C"]

    fig = go.Figure(go.Heatmap(
        x=(mag_edges[:-1] + mag_edges[1:]) / 2, y=(dep_edges[:-1] + dep_edges[1:]) / 2, z=Z,
        zmin=0, zmax=np.nanmax(Z), colorscale=INK_SEQ, customdata=H,
        hovertemplate="M %{x:.1f} ± 0,1<br>Kedalaman %{y:.0f} ± 5 km<br>%{customdata:,.0f} kejadian<extra></extra>",
        colorbar=dict(title=dict(text="Kejadian<br>per sel", font=dict(size=12, color=MUTED)),
                      tickvals=[0, 1, 2, 3, 4], ticktext=["1", "10", "100", "1.000", "10.000"],
                      thickness=12, len=0.55, x=1.0, xanchor="left", y=0.62, outlinewidth=0,
                      tickfont=dict(color=MUTED)),
        name="Semua catatan", showlegend=False,
    ))

    # Semua gempa M>=6 (tidak disampel)
    m6 = df[df["mag"] >= 6].copy()
    rng = np.random.default_rng(7)
    fig.add_trace(go.Scatter(
        x=m6["mag"] + rng.uniform(-0.04, 0.04, len(m6)), y=m6["depth"], mode="markers",
        marker=dict(size=7, color=RED, opacity=0.85, line=dict(color="white", width=0.8)),
        name=f"Gempa M≥6 (semua {len(m6)} kejadian)",
        customdata=np.c_[m6["mag"], m6["region"], m6["datetime"].dt.strftime("%d %b %Y")],
        hovertemplate="M %{customdata[0]:.1f}, %{y} km<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))

    # Pita kelas kedalaman (BMKG) + porsi kejadiannya
    pct_depth = df["depth_class"].value_counts(normalize=True).reindex(DEPTH_LABELS) * 100
    for batas in (60, 300):
        fig.add_hline(y=batas, line=dict(color=MUTED, width=1, dash="dash"))
    for lab, y in zip(DEPTH_LABELS, (30, 180, 520)):
        fig.add_annotation(x=1.15, y=y, xref="x", yref="y", xanchor="left", showarrow=False,
                           text=f"<b>{lab}</b>: {angka(pct_depth[lab], 1)}% catatan",
                           font=dict(size=12, color=INK), bgcolor="rgba(255,255,255,0.85)")
    fig.add_annotation(x=1.1, y=735, xref="x", yref="y", xanchor="left", yanchor="bottom", showarrow=False,
                       align="left", font=dict(size=11, color=MUTED),
                       text=(f"{angka(audit['depth_default_pct'], 1)}% catatan memakai kedalaman tepat 10 km "
                             "(nilai default),<br>sehingga baris teratas tampak paling gelap."))
    fig.add_annotation(x=7.95, y=440, xref="x", yref="y", xanchor="right", showarrow=False, align="right",
                       text=(f"<b>{angka(facts['pct_M6plus_shallow'], 0)}% gempa M≥6</b><br>berpusat dangkal (<60 km)"),
                       font=dict(size=13, color=RED))

    fig.update_layout(
        title="Magnitudo dan kedalaman gempa yang tercatat",
        xaxis=dict(title="Magnitudo", range=[0.95, 8.05], dtick=1, showgrid=False),
        yaxis=dict(title="Kedalaman (km)", range=[760, 0], showgrid=False, ticksuffix=" "),
        showlegend=True,
        legend=dict(x=0.995, y=0.03, xanchor="right", yanchor="bottom", bgcolor="rgba(255,255,255,0.9)"),
    )
    kerangka(fig, (f"Mayoritas catatan kecil dan dangkal (sel tergelap di bagian atas). Gempa M≥6 langka, "
                   f"tetapi sekitar {angka(facts['pct_M6plus_shallow'] / 10, 0)} dari 10<br>di antaranya juga dangkal. "
                   "Warna sel = jumlah kejadian (skala logaritmik); sumbu kedalaman terbalik seperti penampang bumi."),
             tinggi=640, lebar=1000, atas=118, bawah=72, kiri=72, kanan=110)
    figures.append(fig)


    # ==== CELL NOTEBOOK 45 ====
    top10_eq = (
        df.sort_values(["mag", "depth"], ascending=[False, True])
        .head(10)
        .copy()
    )

    top10_eq["rank"] = range(1, len(top10_eq) + 1)
    top10_eq["date_str"] = top10_eq["datetime"].dt.strftime("%d %b %Y")
    top10_eq["mag_str"] = top10_eq["mag"].map(lambda x: f"M {x:.1f}")
    top10_eq["depth_str"] = top10_eq["depth"].map(lambda x: f"{x:.0f} km")

    top10_eq = top10_eq[
        ["rank", "mag", "mag_str", "date_str", "remark", "depth_str", "lat", "lon"]
    ]



    # ==== CELL NOTEBOOK 46 ====
    # ===== Grafik 5: satu peta 10 gempa terbesar (menggabungkan dua peta lama) =====
    top10 = (df.sort_values(["mag", "depth"], ascending=[False, True]).head(10).copy()
               .assign(peringkat=range(1, 11)))
    BLN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    WILAYAH_ID = {"Irian Jaya": "Papua (Irian Jaya)", "Southwest of Sumatra": "Barat daya Sumatra",
                  "Northern Sumatra": "Sumatra bagian utara", "Southern Sumatra": "Sumatra bagian selatan",
                  "Minahassa Peninsula, Sulawesi": "Sulawesi (Palu)", "Tanimbar Islands Reg.": "Kepulauan Tanimbar",
                  "Flores Sea": "Laut Flores", "Banda Sea": "Laut Banda"}
    top10["tanggal"] = [f"{t.day} {BLN[t.month - 1]} {t.year}" for t in top10["datetime"]]
    top10["wilayah"] = top10["region"].map(WILAYAH_ID).fillna(top10["region"])
    top10["ukuran"] = 14 * 10 ** (0.75 * (top10["mag"] - top10["mag"].min()))
    top10["is_palu"] = (top10["datetime"].dt.date == pd.Timestamp("2018-09-28").date()) & \
                       top10["lat"].between(-1.5, 1) & top10["lon"].between(118.5, 121)
    lainnya = df[(df["mag"] >= 6) & ~df.index.isin(top10.index)]

    fig = go.Figure()
    if PLATES is not None:
        fig.add_trace(PLATES)
    for lab in DEPTH_LABELS:                          # menampilkan gempa m>=6
        d = lainnya[lainnya["depth_class"] == lab]
        fig.add_trace(go.Scattergeo(
            lat=d["lat"], lon=d["lon"], mode="markers", legendgroup=lab, name=lab,
            marker=dict(size=5, color=DEPTH_COLORS[lab], opacity=0.45, line=dict(width=0)),
            customdata=np.c_[d["mag"], d["depth"], d["region"]],
            hovertemplate="M %{customdata[0]:.1f}, %{customdata[1]} km<br>%{customdata[2]}<extra>M≥6</extra>"))
    for lab in DEPTH_LABELS:                                    # menghiglight 10 terbesar
        d = top10[top10["depth_class"] == lab]
        fig.add_trace(go.Scattergeo(
            lat=d["lat"], lon=d["lon"], mode="markers+text", name=lab, legendgroup=lab, showlegend=False,
            text=d["peringkat"].astype(str),
            # peringkat 6 tumpang tindih dengan nomor 1, sehingga dibuat sedikit label perbedaan untuk membedakannya
            textposition=["bottom right" if p == 6 else "middle center" for p in d["peringkat"]],
            textfont=dict(size=11, family=FONT,
                          color=["white" if p == 6 else OCEAN for p in d["peringkat"]]),
            marker=dict(size=d["ukuran"], color=DEPTH_COLORS[lab], opacity=0.95,
                        line=dict(color=np.where(d["is_palu"], "white", OCEAN).tolist(),
                                  width=np.where(d["is_palu"], 3, 1).tolist())),
            customdata=np.c_[d["peringkat"], d["mag"], d["depth"], d["wilayah"], d["tanggal"]],
            hovertemplate=("<b>#%{customdata[0]} M %{customdata[1]:.1f}</b><br>%{customdata[3]}<br>"
                           "%{customdata[4]}, kedalaman %{customdata[2]} km<extra></extra>")))
    palu = top10[top10["is_palu"]]
    if not palu.empty:
        fig.add_trace(go.Scattergeo(lat=palu["lat"] + 1.3, lon=palu["lon"] - 0.6, mode="text",
                                    text=["Palu 2018<br>(lihat grafik 6)"], showlegend=False, hoverinfo="skip",
                                    textfont=dict(size=12, color="white", family=FONT)))
    peta_gelap(fig)
    fig.update_geos(domain=dict(x=[0, 0.72], y=[0, 1]))

    # Daftar peringkat di kanan peta
    baris = []
    for r in top10.itertuples():
        tebal = ("<b>", "</b>") if r.is_palu else ("", "")
        baris.append(f"<span style='color:{DEPTH_COLORS[r.depth_class].replace('#F2C14E', '#C99A1E')}'>●</span> "
                     f"{tebal[0]}{r.peringkat}. M{angka(r.mag, 1)}  {r.wilayah}{tebal[1]}"
                     f"<span style='color:{FAINT}'>, {r.tanggal}, {r.depth} km</span>")
    fig.add_annotation(xref="paper", yref="paper", x=0.735, y=0.98, xanchor="left", yanchor="top",
                       align="left", showarrow=False, font=dict(size=12, color=INK),
                       text="<b>Peringkat</b> (warna = kelas kedalaman)<br><br>" + "<br>".join(baris))
    n_dangkal = int((top10["depth"] < SHALLOW_KM).sum())
    fig.update_layout(
        title="Persebaran 10 gempa dengan magnitudo terbesar dalam katalog Indonesia",
        legend=dict(title=dict(text="Kelas kedalaman", font=dict(color="#DCE6F0", size=11)),
                    x=0.01, y=0.03, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(15,30,48,0.85)", font=dict(color="#DCE6F0", size=11), itemsizing="constant",
                    tracegroupgap=0),
    )
    kerangka(fig, (f"Sepuluh gempa terkuat (M{angka(top10['mag'].min(), 1)}–{angka(top10['mag'].max(), 1)}) "
                   f"tersebar di sepanjang sabuk, dari Sumatra hingga Papua; {n_dangkal} dari 10 berpusat dangkal.<br>"),
             tinggi=500, lebar=1240, atas=106, bawah=30, kiri=8, kanan=8)
    figures.append(fig)


    # ==== CELL NOTEBOOK 48 ====
    # ===== Grafik 6: Palu, 14 hari setelah kejadian utama =====
    RADIUS_KM, N_HARI = 150, 14
    palu_candidates = df[(df["date"] == pd.Timestamp("2018-09-28")) & df["lat"].between(-1.5, 1)
                         & df["lon"].between(118.5, 121) & (df["mag"] >= 7)]
    if palu_candidates.empty:
        raise ValueError("Gempa utama Palu tidak ditemukan. Periksa versi dataset.")
    main = palu_candidates.sort_values("mag", ascending=False).iloc[0]

    d = df.assign(dist_km=jarak_km(main["lat"], main["lon"], df["lat"], df["lon"]),
                  t_hari=(df["datetime"] - main["datetime"]) / pd.Timedelta(days=1))
    dekat = d[d["dist_km"] <= RADIUS_KM]
    after = dekat[(dekat["t_hari"] > 0) & (dekat["t_hari"] <= N_HARI)].copy()
    after["hari_ke"] = np.floor(after["t_hari"]).astype(int) + 1
    daily = after.groupby("hari_ke").size().reindex(range(1, N_HARI + 1), fill_value=0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.42, 0.58], vertical_spacing=0.07)
    kecil = after[after["mag"] < 5]
    besar = after[after["mag"] >= 5]

    fig.add_trace(go.Scatter(x=kecil["t_hari"], y=kecil["mag"], mode="markers", name="Gempa susulan M<5",
                             marker=dict(size=6, color=TEAL, opacity=0.5),
                             hovertemplate="Hari %{x:.2f}: M %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=besar["t_hari"], y=besar["mag"], mode="markers", name="Gempa susulan M≥5",
                             marker=dict(size=9, color=RED, line=dict(color="white", width=1)),
                             hovertemplate="Hari %{x:.2f}: M %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[main["mag"]], mode="markers+text", showlegend=False,
                             marker=dict(symbol="star", size=20, color=RED, line=dict(color=INK, width=1)),
                             text=[f"  Kejadian utama M{angka(main['mag'], 1)}"], textposition="middle right",
                             textfont=dict(size=13, color=INK), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Bar(x=daily.index - 0.5, y=daily.values, width=0.82, marker_color=TEAL, name="Kejadian per hari",
                         showlegend=False, text=daily.values, textposition="outside", cliponaxis=False,
                         textfont=dict(size=12, color=INK),
                         hovertemplate="Hari ke-%{x:.0f}: %{y} kejadian<extra></extra>"),
                  row=2, col=1)

    fig.update_xaxes(range=[-0.4, N_HARI + 0.2], dtick=1, showgrid=False)
    fig.update_xaxes(title=f"Hari sejak kejadian utama ({main['datetime']:%d %b %Y, %H:%M} UTC)", row=2, col=1)
    fig.update_yaxes(title="Magnitudo", range=[1.8, 8.2], dtick=1, row=1, col=1)
    fig.update_yaxes(title="Kejadian per hari", range=[0, daily.max() * 1.18], row=2, col=1)
    fig.update_layout(title="Catatan gempa sekitar Palu selama 14 hari setelah kejadian utama",
                      bargap=0, legend=dict(orientation="h", x=1, y=1.0, xanchor="right", yanchor="bottom"))

    figures.append(fig)


    # ==== CELL NOTEBOOK 50 ====
    # === heatmap bulanan gempa dangkal M>=5 ===
    # bulan yang didominasi rangkaian susulan satu gempa besar (M>=6,5) diberi bingkai dan keterangan.
    BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    selected = df[(df["mag"] >= 5) & (df["depth"] < SHALLOW_KM)].copy()
    selected["month"] = selected["datetime"].dt.month
    years = sorted(df["year"].unique())
    heat = (selected.groupby(["year", "month"]).size().unstack()
                    .reindex(index=years, columns=range(1, 13)))

    observed = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    obs = pd.Series(1, index=observed)
    days_in = obs.groupby([observed.year, observed.month]).size()               # hari dalam periode
    days_rec = (df.groupby([df["date"].dt.year, df["date"].dt.month])["date"].nunique()
                  .reindex(days_in.index, fill_value=0))                        # hari yang tercatat
    coverage = (days_rec / days_in).unstack().reindex(index=years, columns=range(1, 13))
    heat = heat.where(coverage.isna(), heat.fillna(0))   # bulan dalam periode tanpa kejadian = 0

    # Kelas warna
    EDGES = [0, 1, 6, 11, 16, 26, 41, np.inf]
    KELAS = ["0", "1–5", "6–10", "11–15", "16–25", "26–40", ">40"]
    WARNA = ["#F1F3F5", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#E31A1C", "#800026"]
    kode = pd.DataFrame(np.digitize(heat.values, EDGES[1:-1]), index=heat.index, columns=heat.columns)
    kode = kode.where(heat.notna())
    n = len(KELAS)
    cs = [c for i, w in enumerate(WARNA) for c in ([i / n, w], [(i + 1) / n, w])]

    fig = go.Figure(go.Heatmap(
        z=kode.values, x=BULAN, y=[str(y) for y in years], zmin=-0.5, zmax=n - 0.5, colorscale=cs,
        xgap=2, ygap=2, customdata=np.dstack([heat.values, coverage.values * 100]),
        hovertemplate="%{x} %{y}: %{customdata[0]:.0f} gempa<br>cakupan pencatatan %{customdata[1]:.0f}% hari<extra></extra>",
        colorbar=dict(title=dict(text="Gempa per<br>bulan", font=dict(size=12, color=MUTED)),
                      tickvals=list(range(n)), ticktext=KELAS, thickness=12, len=0.7, outlinewidth=0,
                      x=1.01, xanchor="left", tickfont=dict(color=MUTED)),
    ))
    yy, xx = np.where(heat.notna().values & (heat.values > 0))
    vals = heat.values[yy, xx].astype(int)
    gelap = kode.values[yy, xx] >= 5
    for mask, warna in ((gelap, "white"), (~gelap, INK)):
        fig.add_trace(go.Scatter(x=[BULAN[i] for i in xx[mask]], y=[str(years[j]) for j in yy[mask]],
                                 text=vals[mask].astype(str), mode="text", hoverinfo="skip", showlegend=False,
                                 textfont=dict(size=11, color=warna)))

    # Kabut untuk bulan dengan cakupan < 50% hari
    for j, y in enumerate(years):
        for i in range(12):
            c = coverage.iloc[j, i]
            if pd.notna(c) and c < 0.5:
                fig.add_shape(type="rect", xref="x", yref="y", x0=i - 0.5, x1=i + 0.5, y0=j - 0.5, y1=j + 0.5,
                              fillcolor="rgba(255,255,255,0.62)", line=dict(width=0), layer="above")

    # Bulan yang didominasi rangkaian susulan satu gempa M>=6,5 (30 hari, radius 150 km)
    NAMA_SINGKAT = {"Irian Jaya": "Papua", "Sumbawa": "Lombok–Sumbawa", "Talaud Islands": "Talaud",
                    "Southern Sumatra": "Sumatra bag. selatan", "Northern Sumatra": "Sumatra bag. utara",
                    "Northern Molucca Sea": "Laut Maluku utara", "Minahassa Peninsula, Sulawesi": "Sulawesi"}
    owner = pd.Series(-1, index=selected.index)
    for idx, b in df[df["mag"] >= 6.5].sort_values("mag").iterrows():   # yang lebih besar menimpa
        dt = (selected["datetime"] - b["datetime"]) / pd.Timedelta(days=1)
        m = (dt > 0) & (dt <= 30) & (jarak_km(b["lat"], b["lon"], selected["lat"], selected["lon"]) <= 150)
        owner[m] = idx
    selected["owner"] = owner
    label_bulan = []
    for (y, mth), g in selected.groupby(["year", "month"]):
        share = (g["owner"] != -1).mean()
        if len(g) >= 20 and share >= 0.5:
            b = df.loc[g.loc[g["owner"] != -1, "owner"].value_counts().index[0]]  # penyumbang terbanyak
            label_bulan.append((y, mth, len(g), share * 100,
                                f"M{angka(b['mag'], 1)} {NAMA_SINGKAT.get(b['region'], b['region'])}"))
    for y, mth, nn, share, nama in label_bulan:
        j, i = years.index(y), mth - 1
        fig.add_shape(type="rect", xref="x", yref="y", x0=i - 0.5, x1=i + 0.5, y0=j - 0.5, y1=j + 0.5,
                      line=dict(color=INK, width=2.2), layer="above")
    daftar = "<br>".join(f"{BULAN[m - 1]} {y} ({nn}): {nama}" for y, m, nn, s, nama in
                         sorted(label_bulan, key=lambda t: -t[2]))
    fig.add_annotation(xref="paper", yref="paper", x=1.15, y=0.0, xanchor="left", yanchor="bottom", align="left",
                       showarrow=False, font=dict(size=12, color=INK),
                       text=f"<b>Bingkai hitam</b>: ≥50% gempa bulan itu<br>adalah susulan gempa M≥6,5<br>(30 hari, radius 150 km)<br><br>{daftar}")

    fig.update_layout(title=f"Sebaran bulanan gempa dangkal (<{SHALLOW_KM} km) berkekuatan M≥5",
                      xaxis=dict(side="top", showgrid=False, ticks=""),
                      yaxis=dict(autorange="reversed", showgrid=False, ticks="", ticksuffix=" "))
    # kerangka(fig, ("Bulan-bulan paling padat umumnya adalah bulan setelah gempa besar dangkal, saat susulan berlangsung "
    #                "berhari-hari.<br>Sel berkabut = pencatatan <50% hari dalam bulan itu, sehingga angkanya kemungkinan terlalu rendah."),
    #          tinggi=720, lebar=1260, atas=150, bawah=40, kiri=60, kanan=430, jarak_sub=36)
    figures.append(fig)


    return figures, facts


# Nama fungsi kompatibel dengan app.py yang sudah berfungsi.
def load_and_clean(csv_path):
    return load_data(csv_path)


def make_figures(df):
    return create_figures(df)