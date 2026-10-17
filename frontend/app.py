import streamlit as st
import json
import os
import subprocess
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Multi-Probe LSH Visualizer", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ Multi-Probe LSH: High-Dimensional Vector Search Engine")
st.markdown("CS G526 | Term Project: Locality-Sensitive Hashing & Multi-Probe Boundary Diagnostics")

# Sidebar: Live Run Controller
st.sidebar.header("⚙️ Live Benchmark Controls")
num_vecs = st.sidebar.select_slider("Number of Vectors (N)", options=[5000, 10000, 25000, 50000], value=10000)
dim = st.sidebar.select_slider("Vector Dimension (D)", options=[64, 128, 256], value=128)
k_bits = st.sidebar.slider("Hyperplanes per Table (K)", min_value=4, max_value=14, value=8)
l_tables = st.sidebar.slider("Hash Tables (L)", min_value=2, max_value=12, value=6)
probes = st.sidebar.slider("Probes per Table", min_value=1, max_value=24, value=6)

if st.sidebar.button("🚀 Run C++ Benchmark"):
    with st.spinner("Executing C++ SIMD engine..."):
        cmd = f".\\build\\lsh_bench.exe -n {num_vecs} -d {dim} -k {k_bits} -l {l_tables} -p {probes}"
        subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.sidebar.success("Benchmark completed!")
        st.rerun()

# Load telemetry JSON
json_path = os.path.join("benchmarks", "results.json")
if not os.path.exists(json_path):
    st.error("Telemetry file 'benchmarks/results.json' not found. Please click 'Run C++ Benchmark' in the sidebar.")
    st.stop()

with open(json_path, "r") as f:
    data = json.load(f)

# Top KPI metrics (5 Columns)
col1, col2, col3, col4, col5 = st.columns(5)
bf_lat = data["brute_force"]["latency_us"]
std_lat = data["standard_lsh"]["latency_us"]
mp_lat = data["multiprobe_lsh"]["latency_us"]

std_speedup = bf_lat / std_lat if std_lat > 0 else 0
mp_speedup = bf_lat / mp_lat if mp_lat > 0 else 0

col1.metric("Brute-Force Latency", f"{bf_lat:.1f} μs", "Baseline")
col2.metric("Standard LSH", f"{std_lat:.1f} μs", f"{std_speedup:.1f}x Speedup")
col3.metric("Multi-Probe LSH", f"{mp_lat:.1f} μs", f"{mp_speedup:.1f}x Speedup")
col4.metric("Multi-Probe Recall", f"{data['multiprobe_lsh']['recall'] * 100:.1f}%", f"+{(data['multiprobe_lsh']['recall'] - data['standard_lsh']['recall']) * 100:.1f}% vs Std")
col5.metric("Index RAM", f"{data.get('index_memory_mb', 0.45):.2f} MB", f"Dataset: {data.get('raw_data_mb', 4.88):.2f} MB")

st.divider()

tab1, tab2, tab3 = st.tabs(["🚀 Latency vs Recall Frontier", "🎯 3D Hyperplane Slicing Playground", "📐 Theoretical S-Curve Simulator"])

with tab1:
    st.subheader("Empirical Search Efficiency & Candidate Filtering")
    c1, c2 = st.columns(2)
    
    methods = ["Brute Force", "Standard LSH (1-Probe)", "Multi-Probe LSH"]
    latencies = [bf_lat, std_lat, mp_lat]
    recalls = [100.0, data["standard_lsh"]["recall"] * 100, data["multiprobe_lsh"]["recall"] * 100]
    
    fig_bar = go.Figure(data=[
        go.Bar(name="Latency (μs)", x=methods, y=latencies, marker_color="#3b82f6", yaxis="y"),
        go.Scatter(name="Recall (%)", x=methods, y=recalls, mode="lines+markers", marker=dict(size=10, color="#10b981"), yaxis="y2")
    ])
    fig_bar.update_layout(
        title="Speedup vs Recall Trade-Off",
        yaxis=dict(title="Latency (μs)"),
        yaxis2=dict(title="Recall (%)", overlaying="y", side="right", range=[0, 105]),
        legend=dict(x=0.01, y=0.99)
    )
    c1.plotly_chart(fig_bar, use_container_width=True)

    candidates = [data["num_vectors"], data["standard_lsh"]["candidates"], data["multiprobe_lsh"]["candidates"]]
    fig_cand = px.bar(
        x=methods, y=candidates, 
        labels={"x": "Method", "y": "Candidates Scored via AVX2"},
        title=f"Search Space Reduction (Total Vectors: {data['num_vectors']:,})",
        color=methods, color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"]
    )
    c2.plotly_chart(fig_cand, use_container_width=True)

with tab2:
    st.subheader("3D Vector Space & Hyperplane Boundary Partitioning")
    st.markdown("Visualizing synthetic vector clusters partitioned by random hyperplanes. Demonstrates how **Multi-Probe** recovers boundary neighbors that single-bucket LSH drops.")
    
    np.random.seed(42)
    c1_pts = np.random.normal(loc=[1.5, 1.5, 1.5], scale=0.35, size=(40, 3))
    c2_pts = np.random.normal(loc=[-1.5, -1.0, 1.0], scale=0.4, size=(40, 3))
    c3_pts = np.random.normal(loc=[0.0, -1.5, -1.2], scale=0.35, size=(40, 3))
    query_pt = np.array([[1.1, 1.1, 1.1]])
    boundary_miss_pt = np.array([[0.85, 0.9, 0.9]])

    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(x=c1_pts[:,0], y=c1_pts[:,1], z=c1_pts[:,2], mode='markers', name='Cluster 1 (Bucket 1)', marker=dict(size=4, color='#38bdf8')))
    fig_3d.add_trace(go.Scatter3d(x=c2_pts[:,0], y=c2_pts[:,1], z=c2_pts[:,2], mode='markers', name='Cluster 2 (Bucket 2)', marker=dict(size=4, color='#c084fc')))
    fig_3d.add_trace(go.Scatter3d(x=c3_pts[:,0], y=c3_pts[:,1], z=c3_pts[:,2], mode='markers', name='Cluster 3 (Bucket 3)', marker=dict(size=4, color='#fbbf24')))
    
    fig_3d.add_trace(go.Scatter3d(x=query_pt[:,0], y=query_pt[:,1], z=query_pt[:,2], mode='markers+text', name='Query Vector', text=['Query'], marker=dict(size=9, color='#ef4444', symbol='diamond')))
    fig_3d.add_trace(go.Scatter3d(x=boundary_miss_pt[:,0], y=boundary_miss_pt[:,1], z=boundary_miss_pt[:,2], mode='markers+text', name='Multi-Probe Recovered Neighbor', text=['Recovered Neighbor'], marker=dict(size=7, color='#22c55e', symbol='cross')))

    x_grid, y_grid = np.meshgrid(np.linspace(-2.5, 2.5, 10), np.linspace(-2.5, 2.5, 10))
    z_grid = 0.3 * x_grid - 0.2 * y_grid
    fig_3d.add_trace(go.Surface(x=x_grid, y=y_grid, z=z_grid, opacity=0.3, colorscale='Blues', showscale=False, name='Hyperplane Boundary'))

    fig_3d.update_layout(height=650, scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.subheader("Interactive Theoretical S-Curve (Collision Probability)")
    st.markdown(r"Formula: $P(\text{Collision}) = 1 - (1 - s^K)^L$, where $s$ is cosine similarity, $K$ is hash bits, and $L$ is number of tables.")
    
    col_k, col_l = st.columns(2)
    k_sim = col_k.slider("Hash Bits (K)", min_value=2, max_value=24, value=k_bits, key="sim_k")
    l_sim = col_l.slider("Hash Tables (L)", min_value=1, max_value=32, value=l_tables, key="sim_l")

    s_range = np.linspace(0.0, 1.0, 100)
    p_collision = 1.0 - (1.0 - s_range**k_sim)**l_sim

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=s_range, y=p_collision, mode='lines', name=f'L={l_sim}, K={k_sim}', line=dict(color='#8b5cf6', width=3)))
    fig_s.update_layout(
        title="Theoretical LSH S-Curve Filter Function",
        xaxis_title="True Cosine Similarity (s)",
        yaxis_title="Collision Probability P(Collision)",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05])
    )
    st.plotly_chart(fig_s, use_container_width=True)