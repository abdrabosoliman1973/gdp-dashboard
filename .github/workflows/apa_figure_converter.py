diff --git a//dev/null b/dash_app.py
index 0000000000000000000000000000000000000000..ea58dc3ff3988dc1af7a5082ebbb724977113679 100644
--- a//dev/null
+++ b/dash_app.py
@@ -0,0 +1,404 @@
+import dash
+from dash import dcc, html
+from dash.dependencies import Input, Output
+import plotly.graph_objects as go
+import plotly.express as px
+import numpy as np
+import pandas as pd
+from sklearn.decomposition import PCA
+from sklearn.manifold import TSNE
+
+# Initialize the Dash app
+app = dash.Dash(__name__)
+
+# ======================
+# Data Preparation
+# ======================
+
+# Generate synthetic cultural embedding data (replace with real data)
+def generate_cultural_data():
+    np.random.seed(42)
+    n_samples = 500
+    countries = ['Qatar', 'Oman']
+    groups = ['National', 'Expat']
+
+    data = {
+        'PowerDistance': np.concatenate([
+            np.random.normal(80, 5, n_samples // 2),  # Qatar
+            np.random.normal(65, 7, n_samples // 2)   # Oman
+        ]),
+        'UncertaintyAvoidance': np.concatenate([
+            np.random.normal(70, 8, n_samples // 2),
+            np.random.normal(85, 6, n_samples // 2)
+        ]),
+        'Individualism': np.concatenate([
+            np.random.normal(35, 10, n_samples // 2),
+            np.random.normal(45, 8, n_samples // 2)
+        ]),
+        'Country': np.repeat(countries, n_samples // 2),
+        'Group': np.tile(np.repeat(groups, n_samples // 4), 2)
+    }
+
+    df = pd.DataFrame(data)
+
+    # Add dimensionality reduction
+    cultural_features = df[['PowerDistance', 'UncertaintyAvoidance', 'Individualism']]
+    pca = PCA(n_components=2)
+    tsne = TSNE(n_components=2, perplexity=30)
+
+    df['PCA1'] = pca.fit_transform(cultural_features)[:, 0]
+    df['PCA2'] = pca.fit_transform(cultural_features)[:, 1]
+    tsne_results = tsne.fit_transform(cultural_features)
+    df['TSNE1'] = tsne_results[:, 0]
+    df['TSNE2'] = tsne_results[:, 1]
+
+    return df
+
+# Pre-compute the dataset for the dashboard
+cultural_df = generate_cultural_data()
+
+# ======================
+# Visualization Components
+# ======================
+
+def create_phase_diagram():
+    """Build a simple diagram showing the three research phases."""
+    fig = go.Figure()
+
+    # Add phases as nodes
+    phases = [
+        {"label": "Cultural Modeling", "x": 0.1, "y": 0.5, "color": "#1f77b4"},
+        {"label": "Hybrid AI Development", "x": 0.5, "y": 0.5, "color": "#ff7f0e"},
+        {"label": "Validation & Testing", "x": 0.9, "y": 0.5, "color": "#2ca02c"}
+    ]
+
+    # Add nodes for each phase
+    for phase in phases:
+        fig.add_trace(
+            go.Scatter(
+                x=[phase["x"]],
+                y=[phase["y"]],
+                mode="markers+text",
+                marker=dict(
+                    size=40,
+                    color=phase["color"],
+                    opacity=0.8,
+                    line=dict(width=2, color='DarkSlateGrey')
+                ),
+                text=phase["label"],
+                textposition="bottom center",
+                hoverinfo="text",
+                textfont=dict(size=14, family="Arial", color="black"),
+                name=phase["label"]
+            )
+        )
+
+    # Add connecting arrows
+    fig.add_annotation(
+        x=0.3, y=0.5,
+        ax=0.2, ay=0,
+        xref="x", yref="y",
+        axref="x", ayref="y",
+        showarrow=True,
+        arrowhead=2,
+        arrowsize=1,
+        arrowwidth=2,
+        arrowcolor="#636363"
+    )
+
+    fig.add_annotation(
+        x=0.7, y=0.5,
+        ax=0.6, ay=0,
+        xref="x", yref="y",
+        axref="x", ayref="y",
+        showarrow=True,
+        arrowhead=2,
+        arrowsize=1,
+        arrowwidth=2,
+        arrowcolor="#636363"
+    )
+
+    # Add feedback loop arrow
+    fig.add_annotation(
+        x=0.5, y=0.3,
+        ax=0.5, ay=0.4,
+        xref="x", yref="y",
+        axref="x", ayref="y",
+        showarrow=True,
+        arrowhead=2,
+        arrowsize=1,
+        arrowwidth=2,
+        arrowcolor="#636363",
+        startstandoff=15,
+        standoff=15
+    )
+
+    fig.update_layout(
+        title="Three-Phase Research Methodology",
+        showlegend=False,
+        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
+        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
+        margin=dict(l=20, r=20, t=60, b=20),
+        height=400,
+        plot_bgcolor='white'
+    )
+
+    return fig
+
+def create_cultural_embedding(reduction_method='PCA'):
+    """Create a scatter plot of the embedding using PCA or t-SNE."""
+    if reduction_method == 'PCA':
+        x_col, y_col = 'PCA1', 'PCA2'
+    else:
+        x_col, y_col = 'TSNE1', 'TSNE2'
+
+    # Build scatter plot
+    fig = px.scatter(
+        cultural_df,
+        x=x_col,
+        y=y_col,
+        color='Country',
+        symbol='Group',
+        hover_data=['PowerDistance', 'UncertaintyAvoidance', 'Individualism'],
+        title=f"Cultural Embedding Space ({reduction_method} Reduction)",
+        width=800,
+        height=600
+    )
+
+    fig.update_traces(
+        marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
+        selector=dict(mode='markers')
+    )
+
+    fig.update_layout(
+        plot_bgcolor='white',
+        paper_bgcolor='white',
+        font=dict(family="Arial", size=12),
+        legend=dict(
+            orientation="h",
+            yanchor="bottom",
+            y=1.02,
+            xanchor="right",
+            x=1
+        )
+    )
+
+    # Add cultural dimension arrows for PCA case
+    if reduction_method == 'PCA':
+        fig.add_annotation(
+            x=0.5, y=1.05,
+            xref="paper", yref="paper",
+            text="Higher Power Distance →",
+            showarrow=False,
+            font=dict(size=12)
+        )
+
+        fig.add_annotation(
+            x=0.05, y=0.5,
+            xref="paper", yref="paper",
+            text="↑ Higher Uncertainty Avoidance",
+            showarrow=False,
+            textangle=90,
+            font=dict(size=12)
+        )
+
+    return fig
+
+def create_ai_architecture():
+    """Create a simple neural network style diagram for the AI architecture."""
+    fig = go.Figure()
+
+    # Define layers and their neurons
+    layers = [
+        {"name": "Sensor Input", "x": 0.1, "neurons": ["LiDAR", "Camera", "Cultural Cues"]},
+        {"name": "Perception", "x": 0.3, "neurons": ["CNN"]},
+        {"name": "Temporal", "x": 0.5, "neurons": ["LSTM"]},
+        {"name": "Decision", "x": 0.7, "neurons": ["Transformer", "Cultural Embedding"]},
+        {"name": "Control", "x": 0.9, "neurons": ["RL Policy"]}
+    ]
+
+    # Add neuron nodes
+    for layer in layers:
+        y_positions = np.linspace(0.1, 0.9, len(layer["neurons"]))
+        for neuron, y in zip(layer["neurons"], y_positions):
+            color = "#ff7f0e" if "Cultural" in neuron else "#1f77b4"
+            fig.add_trace(
+                go.Scatter(
+                    x=[layer["x"]],
+                    y=[y],
+                    mode="markers+text",
+                    marker=dict(size=25, color=color),
+                    text=neuron,
+                    textposition="middle center",
+                    hoverinfo="text",
+                    textfont=dict(size=10),
+                    name=neuron
+                )
+            )
+
+    # Draw connections between layers
+    for i in range(len(layers) - 1):
+        current_layer = layers[i]
+        next_layer = layers[i + 1]
+
+        for j, neuron in enumerate(current_layer["neurons"]):
+            for k, next_neuron in enumerate(next_layer["neurons"]):
+                fig.add_trace(
+                    go.Scatter(
+                        x=[current_layer["x"], next_layer["x"]],
+                        y=[
+                            np.linspace(0.1, 0.9, len(current_layer["neurons"]))[j],
+                            np.linspace(0.1, 0.9, len(next_layer["neurons"]))[k],
+                        ],
+                        mode="lines",
+                        line=dict(width=1, color="#7f7f7f"),
+                        hoverinfo="none",
+                        showlegend=False,
+                    )
+                )
+
+    fig.update_layout(
+        title="Hybrid AI Architecture with Cultural Integration",
+        showlegend=False,
+        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
+        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
+        margin=dict(l=20, r=20, t=60, b=20),
+        height=500,
+        plot_bgcolor='white'
+    )
+
+    return fig
+
+# ======================
+# Dash App Layout
+# ======================
+
+app.layout = html.Div([
+    # Title and description
+    html.Div([
+        html.H1(
+            "Cultural-Aware Autonomous Driving Methodology",
+            style={"textAlign": "center", "color": "#2a3f5f"},
+        ),
+        html.P(
+            "Interactive visualization of the multi-layered, interdisciplinary research methodology",
+            style={"textAlign": "center", "marginBottom": 30},
+        ),
+    ], style={"backgroundColor": "white", "padding": "20px"}),
+
+    # Main tab container
+    html.Div([
+        dcc.Tabs([
+            # Overview tab
+            dcc.Tab(label="Methodology Overview", children=[
+                html.Div([
+                    dcc.Graph(
+                        id="phase-diagram",
+                        figure=create_phase_diagram(),
+                        style={"height": "500px"},
+                    )
+                ], style={"padding": "20px"}),
+            ]),
+
+            # Cultural modeling tab
+            dcc.Tab(label="Cultural Modeling", children=[
+                html.Div([
+                    html.Div([
+                        html.Label("Dimensionality Reduction Method:"),
+                        dcc.RadioItems(
+                            id="reduction-method",
+                            options=[
+                                {"label": "PCA", "value": "PCA"},
+                                {"label": "t-SNE", "value": "TSNE"},
+                            ],
+                            value="PCA",
+                            labelStyle={"display": "inline-block", "marginRight": "20px"},
+                        ),
+                    ], style={"marginBottom": "20px"}),
+                    dcc.Graph(id="cultural-embedding", style={"height": "700px"}),
+                ], style={"padding": "20px"}),
+            ]),
+
+            # Architecture visualization tab
+            dcc.Tab(label="AI Architecture", children=[
+                html.Div([
+                    dcc.Graph(
+                        id="ai-architecture",
+                        figure=create_ai_architecture(),
+                        style={"height": "600px"},
+                    )
+                ], style={"padding": "20px"}),
+            ]),
+
+            # Validation and testing tab
+            dcc.Tab(label="Validation Process", children=[
+                html.Div([
+                    html.H3("Validation Workflow"),
+                    html.Img(
+                        src="https://raw.githubusercontent.com/plotly/dash-sample-apps/main/apps/dash-tsne/assets/validation_workflow.png",
+                        style={"width": "100%", "border": "1px solid #ddd"},
+                    ),
+                    html.Hr(),
+                    html.H3("Uncertainty Quantification"),
+                    dcc.Graph(
+                        figure={
+                            "data": [
+                                go.Scatter(
+                                    x=np.linspace(0, 10, 100),
+                                    y=np.sin(np.linspace(0, 10, 100))
+                                    + np.random.normal(0, 0.1, 100),
+                                    mode="lines",
+                                    name="Behavior Pattern",
+                                    line=dict(color="blue"),
+                                )
+                            ],
+                            "layout": {
+                                "title": "Cultural Uncertainty Detection",
+                                "xaxis": {"title": "Time"},
+                                "yaxis": {"title": "Behavior Metric"},
+                                "shapes": [
+                                    {
+                                        "type": "rect",
+                                        "xref": "paper",
+                                        "yref": "y",
+                                        "x0": 0,
+                                        "y0": -0.5,
+                                        "x1": 1,
+                                        "y1": 0.5,
+                                        "fillcolor": "rgba(255, 0, 0, 0.1)",
+                                        "line": {"width": 0},
+                                    }
+                                ],
+                                "annotations": [
+                                    {
+                                        "x": 5,
+                                        "y": 1.2,
+                                        "text": "Safety Protocol Trigger Zone",
+                                        "showarrow": False,
+                                    }
+                                ],
+                            },
+                        },
+                        style={"height": "400px"},
+                    ),
+                ], style={"padding": "20px"}),
+            ]),
+        ])
+    ])
+])
+
+# ======================
+# Callbacks
+# ======================
+
+@app.callback(Output("cultural-embedding", "figure"), [Input("reduction-method", "value")])
+def update_embedding(method):
+    """Update the embedding plot when the radio button changes."""
+    return create_cultural_embedding(method)
+
+# ======================
+# Run the App
+# ======================
+
+if __name__ == "__main__":
+    app.run_server(debug=True, port=8050)
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import io

st.set_page_config(page_title="APA Figure Converter", layout="centered")

st.title("🖼️ APA-Style Figure Converter")
st.markdown("Upload any figure file (PNG, JPG, PDF) and get a **grayscale, APA-compliant** version in TIFF and PDF (1200 DPI).")

uploaded_file = st.file_uploader("📂 Upload your figure", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = ImageOps.autocontrast(image)
    image = ImageOps.expand(image, border=20, fill='white')

    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"  # fallback serif font
        font = ImageFont.truetype(font_path, 40)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    # Optional caption
    # draw.text((30, image.height - 60), "Figure Note. Example caption text.", font=font, fill="black")

    st.image(image, caption="Preview (Grayscale, APA Style)", use_column_width=True)

    tiff_buffer = io.BytesIO()
    pdf_buffer = io.BytesIO()

    image.save(tiff_buffer, format="TIFF", dpi=(1200, 1200))
    image.save(pdf_buffer, format="PDF")

    st.download_button("⬇️ Download TIFF (1200 DPI)", tiff_buffer.getvalue(), file_name="figure_APA.tiff", mime="image/tiff")
    st.download_button("⬇️ Download PDF", pdf_buffer.getvalue(), file_name="figure_APA.pdf", mime="application/pdf")
