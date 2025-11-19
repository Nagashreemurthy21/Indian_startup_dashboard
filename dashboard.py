import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import re
from pathlib import Path

DATA_PATH = "/mnt/data/merged_startup_data.csv"

st.set_page_config(
    page_title="Indian Startup Funding Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root{ --bg-start:#fff7f0; --bg-end:#f9efe0; --muted:#6b532f; --panel:#fffaf3; }
.stApp{ background: linear-gradient(180deg,var(--bg-start) 0%,var(--bg-end) 100%); font-family: "Merriweather","Poppins",serif; color:#2f1f15; }
.header-title { font-family:"Merriweather", serif; font-size:48px; font-weight:700; text-align:center; margin-bottom:6px; margin-top:8px; }
.header-sub { text-align:center; color:#4b3621; margin-top:0; margin-bottom:18px; font-size:16px; }
.results-text { font-size:20px; color:#3b2f2f; margin-left:40px; margin-bottom:14px; }
.chart-panel { background: var(--panel); padding:18px 20px; border-radius:10px; box-shadow:0 8px 24px rgba(90,64,35,0.06); margin:18px 40px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#f3e5ab 0%,#e9d8a8 100%); padding-top:18px; }
.kpi-small { background:#fff; border-radius:20px; padding:16px 18px; box-shadow:0 10px 30px rgba(75,54,33,0.08); text-align:center; min-width:220px; max-width:360px; }
.kpi-label { font-size:16px; color:#3b2f2f; font-weight:700; display:flex; gap:8px; align-items:center; justify-content:center; }
.kpi-value { font-size:32px; font-weight:800; color:#352713; margin-top:6px; }
.kpi-note { color:#A66B3A; font-size:13px; margin-top:6px; }
footer{ text-align:center; color:var(--muted); padding:10px; }
@media (max-width:980px){
  .results-text{ margin-left:20px; }
  .chart-panel{ margin:12px 12px; padding:14px; }
  .kpi-small{ min-width:180px; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv_try_paths():
    candidates = [
        DATA_PATH,
        "/mnt/data/startup_data.csv",
        "startup_data.csv",
        "merged_startup_data.csv"
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                try:
                    df = pd.read_csv(p, encoding="ISO-8859-1", header=0)
                except Exception:
                    df = pd.read_csv(p, encoding="ISO-8859-1", header=4)
                return df
        except Exception:
            continue
    return pd.DataFrame()

def infer_standard_columns(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date','Startup Name','Sector','City','Investment Type','Amount_raw'])
    col_map = {c.lower(): c for c in df.columns}
    def find(tokens):
        for token in tokens:
            for lower, orig in col_map.items():
                if token in lower:
                    return orig
        return None
    date_col = find(["date","transaction","deal","investment","year"])
    name_col = find(["startup","name of the startup","company","company name","name"])
    sector_col = find(["sector","industry","company profile"])
    city_col = find(["city","location of company","location","state","states/ut","place","address"])
    invest_col = find(["investment type","allocation","funding type","fund type","funds approved"])
    amount_col = find(["amount","funds","funding","allocation of funds","funds approved","amount (in rs)"])
    out = pd.DataFrame()
    out['Date'] = df[date_col] if (date_col and date_col in df.columns) else df.iloc[:,0]
    out['Startup Name'] = df[name_col] if (name_col and name_col in df.columns) else (df.iloc[:,1] if df.shape[1]>1 else "")
    out['Sector'] = df[sector_col] if (sector_col and sector_col in df.columns) else ""
    if city_col and city_col in df.columns:
        out['City'] = df[city_col]
    else:
        fallback = None
        for guess in ['states/ut','state','location','location of company']:
            for lower, orig in col_map.items():
                if guess in lower:
                    fallback = orig; break
            if fallback: break
        out['City'] = df[fallback] if fallback else ""
    out['Investment Type'] = df[invest_col] if (invest_col and invest_col in df.columns) else ""
    out['Amount_raw'] = df[amount_col] if (amount_col and amount_col in df.columns) else df.iloc[:,-1]
    return out

def parse_amount_to_millions(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int,float,np.integer,np.floating)): return float(x)/1_000_000
    s = str(x).strip().lower()
    s = s.replace('₹','').replace('inr','').replace('rs.','').replace('rs','').replace('$','').replace('usd','')
    s = s.replace('\xa0',' ').replace(',','').replace('–','-')
    try:
        token = re.split(r'\s|-|to', s)[0]
        m = re.search(r'([0-9]*\.?[0-9]+)', token)
        if m:
            val = float(m.group(1))
        else:
            found = None
            for t in s.split():
                if re.match(r'^[0-9]*\.?[0-9]+$', t):
                    found = float(t); break
            if found is None: return np.nan
            val = found
    except Exception:
        return np.nan
    multiplier = 1.0
    if any(k in s for k in ['crore','cr']): multiplier = 1e7
    elif any(k in s for k in ['lakh','lac','lakhs','lacs']): multiplier = 1e5
    elif any(k in s for k in ['mn','million','millions','m.']): multiplier = 1e6
    elif any(k in s for k in ['k','thousand']): multiplier = 1e3
    absolute = val * multiplier
    return absolute / 1_000_000

def clean_dataframe(std_df, keep_blanks=False):
    df = std_df.copy()
    if df.empty: return pd.DataFrame()
    def safe_date(d):
        if pd.isna(d): return pd.NaT
        s = str(d).strip()
        for fmt in ("%d/%m/%Y","%Y-%m-%d","%d-%m-%Y","%d %b %Y","%d %B %Y","%m/%d/%Y"):
            try: return datetime.strptime(s, fmt)
            except: continue
        return pd.to_datetime(s, errors='coerce')
    df['Date'] = df['Date'].apply(safe_date)
    df = df.dropna(subset=['Date'])
    if df.empty: return pd.DataFrame()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    for col in ['Sector','City','Investment Type','Startup Name']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].astype(str).str.strip().replace({'nan':'','n/a':''})
    df['City'] = df['City'].astype(str).str.title()
    df['City'] = df['City'].replace({'Bengaluru':'Bangalore','Gurugram':'Gurgaon','Hydrabad':'Hyderabad','New Delhi':'Delhi'})
    if not keep_blanks:
        df = df[df['City'].notna() & (df['City'].str.len()>0)]
    df['Amount'] = df['Amount_raw'].apply(parse_amount_to_millions)
    df = df.drop(columns=['Amount_raw'], errors='ignore')
    df = df.dropna(subset=['Amount'])
    df = df[df['Amount']>0]
    df = df[df['Amount']<100000]
    for col in ['Sector','City','Investment Type','Startup Name']:
        df[col] = df[col].astype(str).str.title().replace({'Nan':'Other','N/A':'Other'})
    return df

def convert_unit(series, unit):
    if unit == "M": return series, "M USD"
    if unit == "Cr": return series * 0.01, "Cr INR"
    return series, "M USD"

BAR_PALETTE = ["#7B3F00","#A66B3A","#C18F60","#D7B48A","#E8D9C4","#B07A3B","#C68642","#A0522D","#8B4513","#5D2A00"]

def show_kpis(total_funding, unit_label, total_startups, avg_invest):
    st.markdown(f"""
    <div style="display:flex; gap:18px; justify-content:center; margin-bottom:18px; margin-left:40px; margin-right:40px;">
      <div class="kpi-small"><div class="kpi-label">💰 Total Funding</div><div class="kpi-value">{total_funding:,.2f} {unit_label}</div><div class="kpi-note">Sum of displayed rows</div></div>
      <div class="kpi-small"><div class="kpi-label">🚀 Total Startups</div><div class="kpi-value">{total_startups:,}</div><div class="kpi-note">Unique startups</div></div>
      <div class="kpi-small"><div class="kpi-label">📈 Avg. Investment</div><div class="kpi-value">{avg_invest:,.2f} {unit_label}</div><div class="kpi-note">Mean per deal (displayed)</div></div>
    </div>
    """, unsafe_allow_html=True)

def plot_trend(df, agg_by, unit):
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.subheader("📊 Funding Trend Over Time")
    d = df.copy()
    if agg_by == "Month":
        d['Period'] = pd.to_datetime(d['Month'])
    elif agg_by == "Quarter":
        d['Period'] = pd.to_datetime(d['Date']).dt.to_period('Q').dt.start_time
    else:
        d['Period'] = pd.to_datetime(d['Date']).dt.to_period('Y').dt.start_time
    series = d.groupby('Period', as_index=False)['Amount'].sum().sort_values('Period')
    if series.empty:
        st.info("No trend data to show.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    y, unit_label = convert_unit(series['Amount'], unit)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series['Period'], y=y, mode='lines+markers', line=dict(color=BAR_PALETTE[1], width=3),
                             marker=dict(size=8, color=BAR_PALETTE[4], line=dict(width=1.5, color=BAR_PALETTE[1]))))
    fig.add_shape(type="line", x0=series['Period'].min(), x1=series['Period'].max(), y0=0, y1=0, line=dict(color="black", width=1))
    fig.update_layout(margin=dict(t=12,b=30,l=40,r=20), plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                     xaxis=dict(title_text='Month' if agg_by=="Month" else agg_by, tickformat="%b\n%Y" if agg_by=="Month" else None, showgrid=False, showline=True, linecolor='#4b3621'),
                     yaxis=dict(title_text=f'Funding ({unit_label})', showgrid=True, gridcolor='#e9d8c0', zeroline=False), height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_top_cities_by_deals(df, top_n=6):
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.subheader(f"🏙 Top {top_n} Cities by Deal Count")
    counts = df['City'].value_counts().reset_index().head(top_n)
    if counts.empty:
        st.info("No city data.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    counts.columns = ['City','Deals']
    colors = BAR_PALETTE[:len(counts)]
    fig = go.Figure(go.Bar(x=counts['City'], y=counts['Deals'], marker=dict(color=colors, line=dict(color='#4b3621', width=2)), text=counts['Deals'], textposition='outside', width=0.55))
    fig.update_layout(margin=dict(t=28,b=80,l=60,r=40), xaxis=dict(title_text='City', tickangle=-20, tickfont=dict(size=15), automargin=True), yaxis=dict(title_text='Deal Count', showgrid=True, gridcolor='#e9d8c0', tickfont=dict(size=14)), height=440, plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_investment_pie_like_screenshot(df, top_n=8, unit="M"):
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.subheader("🍊 Funding Distribution by Investment Type")

    agg = df.groupby('Investment Type')['Amount'].sum().reset_index().sort_values('Amount', ascending=False)
    if agg.empty or agg['Amount'].sum() == 0:
        fig = go.Figure(go.Pie(labels=["No Data"], values=[1], marker=dict(colors=["#A66B3A"])))
        fig.update_traces(textinfo='none')
        fig.update_layout(showlegend=False, annotations=[dict(text="No data", x=0.5, y=0.5, font_size=20, showarrow=False)], margin=dict(t=20,b=0,l=0,r=0), height=520, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Group small slices into 'Others'
    if len(agg) > top_n:
        top = agg.head(top_n).copy()
        others_sum = agg['Amount'].iloc[top_n:].sum()
        top = pd.concat([top, pd.DataFrame({'Investment Type': ['Others'], 'Amount': [others_sum]})], ignore_index=True)
        plot_df = top
    else:
        plot_df = agg.copy()

    plot_df['Disp'], unit_label = convert_unit(plot_df['Amount'], unit)
    colors = ["#7B3F00"] + BAR_PALETTE[1:len(plot_df)]

    # Stacked outside annotations
    total = plot_df['Disp'].sum()
    y_step = 0.040 if len(plot_df) > 10 else 0.052
    start_y = 0.89
    y_positions = [start_y - y_step * i for i in range(len(plot_df))]
    annotations = []
    for i, (lab, v) in enumerate(zip(plot_df['Investment Type'], plot_df['Disp'])):
        pct = (v / total) if total > 0 else 0
        txt = f"{lab} {pct:.4f}" if pct > 0 else lab
        annotations.append(dict(
            x=0.31, y=y_positions[i], xref='paper', yref='paper',
            text=txt, showarrow=False, font=dict(size=11, color='#2f1f15')
        ))

    fig = go.Figure(go.Pie(
        labels=plot_df['Investment Type'],
        values=plot_df['Disp'],
        marker=dict(colors=colors, line=dict(color='#fff', width=1)),
        textinfo='none',
        sort=False,
        direction='clockwise',
        domain={'x': [0.06, 0.58], 'y': [0.08, 0.92]}
    ))

    fig.update_layout(
        annotations=annotations,
        height=520,
        margin=dict(t=12, b=15, l=20, r=180),
        legend=dict(orientation='v', x=0.93, y=0.9, xanchor='right', font=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_top_startups(df, top_n=10, unit="M"):
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.subheader(f"🏆 Top {top_n} Startups by Funding")
    agg = df.groupby('Startup Name')['Amount'].sum().reset_index().sort_values('Amount', ascending=False).head(top_n)
    if agg.empty:
        st.info("No startup data.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    agg['Disp'], unit_label = convert_unit(agg['Amount'], unit)
    agg_display = agg[['Startup Name','Disp']].rename(columns={'Disp':f'Funding ({unit_label})'})
    st.dataframe(agg_display.style.format({f'Funding ({unit_label})': '{:,.2f}'}), height=320)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    raw = load_csv_try_paths()
    st.markdown('<h1 class="header-title">Indian Startup Funding Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Gain insights into India\'s startup ecosystem with funding trends, top cities, and investment patterns.</div>', unsafe_allow_html=True)

    if raw is None or raw.empty:
        show_kpis(0.0, "M USD", 0, 0.0)
        _ = st.file_uploader("Upload CSV", type=['csv'])
        return

    std = infer_standard_columns(raw)
    df = clean_dataframe(std, keep_blanks=False)
    if df.empty:
        st.error("No usable rows after cleaning. Check CSV columns & amount formats.")
        return

    st.sidebar.header("🔎 Filter Data")
    all_cities = sorted(df['City'].dropna().unique().tolist())
    default_cities = ["Bangalore","Mumbai","Delhi","Chennai","Hyderabad","Pune"]
    default_cities = [c for c in default_cities if c in all_cities]
    if not default_cities:
        default_cities = all_cities[:6] if len(all_cities) >= 6 else all_cities

    year_range = st.sidebar.slider("📅 Year Range", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=(int(df['Year'].min()), int(df['Year'].max())), step=1)
    selected_cities = st.sidebar.multiselect("📍 Select Cities", options=all_cities, default=default_cities)
    min_amt, max_amt = float(df['Amount'].min()), float(df['Amount'].max())
    funding_range = st.sidebar.slider("💸 Funding Amount (M)", min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt))
    st.sidebar.markdown("---")
    unit = st.sidebar.selectbox("Display unit", options=["M","Cr"], index=0)
    agg_by = st.sidebar.selectbox("Trend aggregation", options=["Month","Quarter","Year"], index=0)
    top_cities_by = st.sidebar.selectbox("Top cities by", options=["Deal Count","Funding"], index=0)
    top_n_cities = st.sidebar.slider("Top cities to show", min_value=4, max_value=10, value=6)
    top_n_sectors = st.sidebar.slider("Top investment types (pie)", min_value=3, max_value=12, value=8)
    top_n_startups = st.sidebar.slider("Top startups table rows", min_value=5, max_value=50, value=10)

    city_condition = (df['City'].isin(selected_cities)) if (len(selected_cities) > 0 and len(selected_cities) != len(all_cities)) else True
    filtered = df[
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1]) &
        (city_condition) &
        (df['Amount'].between(funding_range[0], funding_range[1]))
    ].copy()

    st.markdown(f"<div class='results-text'>Showing results for <strong>{filtered['Startup Name'].nunique():,}</strong> startups.</div>", unsafe_allow_html=True)
    st.markdown("---")

    amounts_conv, unit_label = convert_unit(filtered['Amount'], unit)
    total_funding = amounts_conv.sum()
    total_startups = filtered['Startup Name'].nunique()
    avg_invest = amounts_conv.mean() if len(amounts_conv)>0 else 0.0
    show_kpis(total_funding, unit_label, total_startups, avg_invest)

    plot_trend(filtered, agg_by, unit)
    if top_cities_by == "Deal Count":
        plot_top_cities_by_deals(filtered, top_n=top_n_cities)
    else:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.subheader(f"🏙 Top {top_n_cities} Cities by Funding")
        agg = filtered.groupby('City')['Amount'].sum().reset_index().sort_values('Amount', ascending=False).head(top_n_cities)
        if not agg.empty:
            colors = BAR_PALETTE[:len(agg)]
            fig = go.Figure(go.Bar(x=agg['City'], y=agg['Amount'], marker=dict(color=colors, line=dict(color='#4b3621', width=1)), text=agg['Amount'].round(2), textposition='outside'))
            fig.update_layout(margin=dict(t=12,b=60,l=60,r=20), xaxis=dict(title_text='City', tickangle=-25), yaxis=dict(title_text=f'Funding (M USD)', showgrid=True, gridcolor='#e9d8c0'), height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No city funding data.")
        st.markdown('</div>', unsafe_allow_html=True)

    plot_investment_pie_like_screenshot(filtered, top_n=top_n_sectors, unit=unit)
    show_top_startups(filtered, top_n=top_n_startups, unit=unit)
    st.markdown("<hr><footer>© 2025 | Crafted using Streamlit & Plotly</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
