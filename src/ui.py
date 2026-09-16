from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from .data_loader import load_bundle, validate_bundle, image_url, sync_from_drive
from .intention_engine import taxonomy, add_intention_label, demand_supply_gap, catalog_supply
from .customer_engine import profile, top_intentions, customer_ids
from .recommendation_engine import artifact_status, can_infer, explain_unavailable

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Intention-Aware Fashion BI", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

@st.cache_data(show_spinner=False)
def data():
    sync_from_drive()
    return load_bundle()

@st.cache_data(show_spinner=False)
def labels(summary):
    label_file = ROOT / "data" / "intention_labels.json"
    if label_file.exists():
        raw = json.loads(label_file.read_text(encoding="utf-8"))
        return {int(k): v.get("name", f"Intention {k}") for k, v in raw.items()}
    tax = taxonomy(summary)
    return dict(zip(tax.intention_id.astype(int), tax.intention_name))

def header():
    st.sidebar.markdown("# INTENTION-AWARE\n# FASHION BI")
    st.sidebar.caption("From purchase prediction to purchase intention intelligence")
    st.sidebar.divider()
    st.sidebar.caption("H&M analytical sample")
    st.sidebar.caption("105,542 articles · 1.37M customers · 31.8M transactions")

def load_or_stop():
    try: b = data()
    except Exception as e:
        st.error(f"Required Google Drive data could not be loaded: {e}")
        st.stop()
    errors = validate_bundle(b)
    if errors:
        st.error("Data contract validation failed")
        for e in errors: st.write(f"- {e}")
        st.stop()
    return b

def product_card(row, score=None):
    article = str(row.article_id)
    image_id = None
    if "image_id" in row.index and pd.notna(row.image_id): image_id = str(row.image_id)
    path = image_url(image_id) if image_id else None
    if path: st.image(path, use_container_width=True)
    else: st.info("Image asset not available in the supplied local image folder.")
    st.markdown(f"**{row.get('prod_name', 'Unnamed product')}**")
    st.caption(f"Article {article} · {row.get('product_type_name', 'Product type unavailable')}")
    st.caption(f"{row.get('section_name', 'Section unavailable')} · {row.get('colour_group_name', 'Colour unavailable')}")
    st.caption("Price unavailable in supplied Google Drive data")
    if "intention_name" in row.index: st.caption(f"Intention: {row.intention_name}")
    if "dominant_prob" in row.index: st.caption(f"Intention fit: {float(row.dominant_prob):.1%}")
    if score is not None: st.metric("Verified model score", f"{score:.4f}")

def overview(b):
    st.title("Intention-Aware Fashion BI")
    st.subheader("From purchase prediction to purchase intention intelligence")
    st.write("A decision-support platform built from the supplied H&M thesis outputs. It connects customer intention, product intelligence, recommendation evidence and assortment signals.")
    s = b["summary"]
    cols = st.columns(5)
    metrics = [("Analytical products", len(b["catalog"])), ("Analytical customers", len(b["customers"])), ("Test interactions", len(b["interactions"])), ("Purchase intentions", s.get("intentions", {}).get("total_intentions", 0)), ("Three-Tower AUC", s.get("model_performance", {}).get("three_tower_auc", "n/a"))]
    for col,(label,value) in zip(cols,metrics): col.metric(label, value)
    st.divider()
    gap = demand_supply_gap(b["user_dist"], b["catalog"], taxonomy(s))
    st.subheader("Executive insight: demand versus catalogue representation")
    st.plotly_chart(px.bar(gap, x="intention_name", y=["demand_share","supply_share"], barmode="group", labels={"value":"Share","intention_name":"Intention"}), use_container_width=True)
    st.dataframe(gap[["intention_id","intention_name","demand_share","supply_share","gap_pp","signal"]].style.format({"demand_share":"{:.1%}","supply_share":"{:.1%}","gap_pp":"{:.2f} pp"}), use_container_width=True, hide_index=True)
    st.caption("Gap is a relative catalogue-representation signal, not a measured inventory shortage or financial impact.")

def customer_page(b):
    st.title("Customer Intelligence")
    tax, lab = taxonomy(b["summary"]), labels(b["summary"])
    dist = b["user_dist"].copy(); dist["user_share"] = pd.to_numeric(dist["user_share"])
    st.plotly_chart(px.bar(dist.sort_values("user_share"), x="user_share", y="intention_name", orientation="h", labels={"user_share":"Customer share"}), use_container_width=True)
    cid = st.selectbox("Customer ID from supplied Drive data", customer_ids(b["user_weights"]), key="customer")
    row = profile(b["user_weights"], b["confidence"], cid)
    c = st.columns(4); c[0].metric("Purchases", row.get("n_purchases", "n/a")); c[1].metric("Confidence", f"{float(row.get('confidence',0)):.3f}"); c[2].metric("Cold-start", str(row.get("is_cold_start", False))); c[3].metric("Dominant intention", lab.get(int(row.get("dominant_intention",-1)), "Unknown"))
    dist2 = top_intentions(row, lab)
    st.plotly_chart(px.bar(dist2.sort_values("weight"), x="weight", y="intention_name", orientation="h", labels={"weight":"Profile weight"}), use_container_width=True)

def intention_page(b):
    st.title("Purchase Intention Analytics")
    tax = taxonomy(b["summary"])
    gap = demand_supply_gap(b["user_dist"], b["catalog"], tax)
    selected = st.selectbox("Select intention", tax.intention_id.tolist(), format_func=lambda x: f"T{x} — {tax.loc[tax.intention_id==x,'intention_name'].iloc[0]}")
    row = gap[gap.intention_id == selected].iloc[0]
    c=st.columns(4); c[0].metric("Demand share", f"{row.demand_share:.1%}"); c[1].metric("Catalogue share", f"{row.supply_share:.1%}"); c[2].metric("Gap", f"{row.gap_pp:+.2f} pp"); c[3].metric("Signal", row.signal)
    catalog_cols = [c for c in b["catalog"].columns if c != "dominant_intention"]
    prof = b["article_intentions"].merge(b["catalog"][catalog_cols], on="article_id", how="left").merge(b["images"], on="article_id", how="left")
    prof = prof[prof.dominant_intention == selected].sort_values("dominant_prob", ascending=False).head(12)
    st.subheader("Products associated with this intention")
    for start in range(0, len(prof), 4):
        cs=st.columns(4)
        for col,(_,r) in zip(cs,prof.iloc[start:start+4].iterrows()):
            with col: product_card(r)

def assortment_page(b):
    st.title("Product & Assortment Intelligence")
    tax=taxonomy(b["summary"]); supply=catalog_supply(b["catalog"],tax)
    st.dataframe(supply, use_container_width=True, hide_index=True)
    st.subheader("Explore the supplied catalogue")
    names=sorted(b["catalog"].product_group_name.dropna().unique().tolist())
    group=st.selectbox("Product group", ["All"]+names)
    df=b["catalog"].merge(b["article_intentions"][["article_id","dominant_prob"]],on="article_id").merge(b["images"],on="article_id",how="left")
    if group != "All": df=df[df.product_group_name==group]
    st.dataframe(df[["article_id","prod_name","product_type_name","product_group_name","section_name","colour_group_name","dominant_intention","dominant_prob"]].head(100), use_container_width=True, hide_index=True)

def recommendation_page(b):
    st.title("AI Personalised Recommendation")
    st.write("This page is reserved for inference from the exact trained Three-Tower checkpoint and thesis feature artifacts.")
    statuses=artifact_status(); st.table(pd.DataFrame({"Artifact":list(statuses),"Available":list(statuses.values())}))
    if not can_infer():
        st.warning(explain_unavailable())
        st.info("No recommendation score is generated here. This fail-closed behavior protects the thesis claim: the app will not substitute a new ranking algorithm for the trained Three-Tower model.")
        return
    st.error("Checkpoint detected, but inference is disabled until the training preprocessing contract is verified against the notebook.")

def performance_page(b):
    st.title("Model Performance")
    m=b["summary"].get("model_performance",{})
    c=st.columns(3); c[0].metric("Three-Tower AUC",m.get("three_tower_auc","n/a")); c[1].metric("Two-Tower AUC",m.get("two_tower_auc","n/a")); c[2].metric("Reported improvement",m.get("improvement","n/a"))
    st.subheader("What the model evidence supports")
    st.write("The supplied summary supports comparative offline evaluation. It does not by itself establish revenue uplift, conversion uplift, margin improvement or inventory reduction.")
    st.subheader("From AI prediction to business action")
    st.code("Customer → Intention profile → Product alignment → Recommendation → Catalogue representation → Decision support", language="text")

def render(page="Executive Overview"):
    header(); b=load_or_stop()
    if page=="Executive Overview": overview(b)
    elif page=="Customer Intelligence": customer_page(b)
    elif page=="Intention Analytics": intention_page(b)
    elif page=="Product & Assortment": assortment_page(b)
    elif page=="AI Recommendation": recommendation_page(b)
    elif page=="Model Performance": performance_page(b)
