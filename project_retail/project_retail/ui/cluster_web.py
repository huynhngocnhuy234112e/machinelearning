from flask import Flask, request, render_template_string, send_file, jsonify
from project_retail.project_retail.connectors.connector import Connector
from project_retail.project_retail.models.customer_clustering import (
    fetch_customer_features,
    kmeans_cluster,
    get_cluster_details,
    export_clusters_to_excel,
)

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Customer Clusters</title>
    <style>
        :root{
            --c0:#4F46E5; /* indigo */
            --c1:#EF4444; /* red */
            --c2:#10B981; /* emerald */
            --c3:#F59E0B; /* amber */
            --c4:#3B82F6; /* blue */
            --c5:#8B5CF6; /* violet */
            --c6:#14B8A6; /* teal */
            --c7:#D946EF; /* fuchsia */
            --text:#0f172a; /* slate-900 */
            --muted:#334155; /* slate-700 */
            --border:#e5e7eb; /* gray-200 */
            --bg:#f7f8fb; /* light bg */
            --header:#f1f5f9; /* slate-100 */
        }
        *{box-sizing:border-box}
        body { margin:0; font-family: Arial, sans-serif; color:var(--text); background: var(--bg); line-height: 1.5; }
        .header { background: linear-gradient(90deg, var(--c0), var(--c4)); color: #fff; padding: 16px 24px; box-shadow: 0 6px 16px rgba(0,0,0,.08); }
        .header h1 { margin: 0; font-size: 22px; letter-spacing: .3px; }
        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }

        .meta { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom: 16px; color: var(--muted); }
        .badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; background:#eef2ff; color:#4338ca; border:1px solid #e0e7ff; }
        .badge.secondary { background:#ecfeff; color:#0e7490; border-color:#cffafe; }
        .badge.warn { background:#fff7ed; color:#b45309; border-color:#fde68a; }
        .spacer { flex:1 1 auto }
        .btn { display:inline-block; text-decoration:none; background: var(--c4); color:#fff; padding:8px 14px; border-radius:8px; font-weight:600; box-shadow:0 2px 8px rgba(59,130,246,.35); border:1px solid rgba(255,255,255,.3); }
        .btn:hover{ background: #2563eb }

        .cluster-card { background:#fff; border:1px solid var(--border); border-left-width:8px; border-radius:14px; margin-bottom: 24px; box-shadow: 0 2px 14px rgba(0,0,0,.06); }
        .cluster-0{ border-left-color: var(--c0) }
        .cluster-1{ border-left-color: var(--c1) }
        .cluster-2{ border-left-color: var(--c2) }
        .cluster-3{ border-left-color: var(--c3) }
        .cluster-4{ border-left-color: var(--c4) }
        .cluster-5{ border-left-color: var(--c5) }
        .cluster-6{ border-left-color: var(--c6) }
        .cluster-7{ border-left-color: var(--c7) }
        .card-inner{ padding: 16px 16px 8px 16px; }
        .cluster-header { display:flex; align-items:center; gap:10px; margin-bottom: 10px; }
        .cluster-title { margin:0; font-size:18px; }
        .count { color: var(--muted); font-weight: 700; }
        .cluster-badge{ padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; color:#fff; }
        .cluster-0 .cluster-badge{ background: var(--c0) }
        .cluster-1 .cluster-badge{ background: var(--c1) }
        .cluster-2 .cluster-badge{ background: var(--c2) }
        .cluster-3 .cluster-badge{ background: var(--c3) }
        .cluster-4 .cluster-badge{ background: var(--c4) }
        .cluster-5 .cluster-badge{ background: var(--c5) }
        .cluster-6 .cluster-badge{ background: var(--c6) }
        .cluster-7 .cluster-badge{ background: var(--c7) }

        .table-wrap{ border:1px solid var(--border); border-radius: 10px; overflow: auto; max-height: 52vh; }
        table { border-collapse: separate; border-spacing: 0; width: 100%; }
        thead th { position: sticky; top: 0; background: var(--header); border-bottom:1px solid var(--border); text-align:left; padding: 10px; font-size:13px; }
        tbody td { padding: 8px 10px; border-bottom:1px solid var(--border); font-size: 13px; }
        tbody tr:nth-child(even){ background: #fbfbfd }
        tbody tr:hover{ background: #f0f9ff }
    </style>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
    <div class="header">
        <h1>Customer Clusters</h1>
    </div>
    <div class="container">
        <div class="meta">
            <span class="badge">scenario: {{ scenario }}</span>
            <span class="badge secondary">features: {{ features|join(', ') }}</span>
            <span class="badge warn">k: {{ k }}</span>
            <span class="badge secondary">scale: {{ scale }}</span>
            <span class="spacer"></span>
            <a class="btn" href="/download?k={{ k }}&scenario={{ scenario }}&scale={{ 'true' if scale else 'false' }}">Download Excel</a>
        </div>
        {% for c in clusters %}
            <div class="cluster-card cluster-{{ c.color_index }}">
                <div class="card-inner">
                    <div class="cluster-header">
                        <span class="cluster-badge">Cluster {{ c.label }}</span>
                        <h2 class="cluster-title">{{ c.count }} customers</h2>
                    </div>
                    {% if c.headers %}
                    <div class="table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    {% for h in c.headers %}<th>{{ h }}</th>{% endfor %}
                                </tr>
                            </thead>
                            <tbody>
                                {% for row in c.rows %}
                                    <tr>
                                        {% for h in c.headers %}<td>{{ row[h] }}</td>{% endfor %}
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    {% else %}
                        <p style="color:var(--muted)">(no rows)</p>
                    {% endif %}
                </div>
            </div>
        {% endfor %}
    </div>
    
</body>
</html>
"""


def _assemble_cluster_view(details_dict):
    clusters = []
    for label in sorted(details_dict.keys()):
        df = details_dict[label]
        if df is None or df.empty:
            clusters.append({"label": label, "count": 0, "headers": [], "rows": [], "color_index": int(label) % 8})
        else:
            headers = list(df.columns)
            rows = [dict(zip(headers, map(lambda x: x, row))) for row in df.values]
            clusters.append({"label": label, "count": len(df), "headers": headers, "rows": rows, "color_index": int(label) % 8})
    return clusters


def output_customers_by_cluster_web(k: int = 4, scenario: str = "2d", scale: bool = False) -> str:
    # features by scenario
    if scenario == "3d":
        feature_cols = ['Age', 'Annual Income', 'Spending Score']
    else:
        feature_cols = ['Age', 'Spending Score']

    # db connect and pipeline
    conn = Connector(database="salesdatabase")
    conn.connect()
    df_features = fetch_customer_features(conn)
    labels, _model, _X = kmeans_cluster(df_features, feature_cols, n_clusters=k, scale=scale)
    df_features = df_features.copy()
    df_features['cluster'] = labels
    details = get_cluster_details(conn, df_features)

    clusters = _assemble_cluster_view(details)
    return render_template_string(PAGE_TEMPLATE, clusters=clusters, features=feature_cols, k=k, scale=scale, scenario=scenario)


@app.route("/")
def clusters_page():
    k = int(request.args.get("k", 4))
    scenario = request.args.get("scenario", "2d")
    scale = request.args.get("scale", "false").lower() == "true"
    return output_customers_by_cluster_web(k=k, scenario=scenario, scale=scale)


@app.route("/api/clusters")
def clusters_api():
    # inputs
    k = int(request.args.get("k", 4))
    scenario = request.args.get("scenario", "2d")
    scale = request.args.get("scale", "false").lower() == "true"

    # features by scenario
    if scenario == "3d":
        feature_cols = ['Age', 'Annual Income', 'Spending Score']
    else:
        feature_cols = ['Age', 'Spending Score']

    # db connect and build details
    conn = Connector(database="salesdatabase")
    conn.connect()
    df_features = fetch_customer_features(conn)
    labels, _model, _X = kmeans_cluster(df_features, feature_cols, n_clusters=k, scale=scale)
    df_features = df_features.copy()
    df_features['cluster'] = labels
    details = get_cluster_details(conn, df_features)

    clusters = _assemble_cluster_view(details)
    return jsonify({
        "scenario": scenario,
        "features": feature_cols,
        "k": k,
        "scale": scale,
        "clusters": clusters,
    })


@app.route("/download")
def download_excel():
    # inputs
    k = int(request.args.get("k", 4))
    scenario = request.args.get("scenario", "2d")  # '2d' or '3d'
    scale = request.args.get("scale", "false").lower() == "true"

    # features by scenario
    if scenario == "3d":
        feature_cols = ['Age', 'Annual Income', 'Spending Score']
    else:
        feature_cols = ['Age', 'Spending Score']

    # db connect and build details
    conn = Connector(database="salesdatabase")
    conn.connect()
    df_features = fetch_customer_features(conn)
    labels, _model, _X = kmeans_cluster(df_features, feature_cols, n_clusters=k, scale=scale)
    df_features = df_features.copy()
    df_features['cluster'] = labels
    details = get_cluster_details(conn, df_features)

    # write to memory and return
    from io import BytesIO
    buffer = BytesIO()
    export_clusters_to_excel(details, buffer)
    buffer.seek(0)
    filename = f"customer_clusters_{scenario}_k{k}.xlsx"
    return send_file(
        buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )


if __name__ == "__main__":
    # Default dev server
    app.run(host="127.0.0.1", port=5000, debug=True)