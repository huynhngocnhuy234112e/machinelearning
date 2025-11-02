from project_retail.project_retail.connectors.connector import Connector
from project_retail.project_retail.models.customer_clustering import (
    fetch_customer_features,
    kmeans_cluster,
    get_cluster_details,
    print_customers_by_cluster,
    export_clusters_to_excel,
)


def scenario_2d():
    conn = Connector(database="salesdatabase")
    conn.connect()
    df_features = fetch_customer_features(conn)
    labels, _model, _X = kmeans_cluster(df_features, ['Age', 'Spending Score'], n_clusters=4, scale=False)
    df_features['cluster'] = labels
    details = get_cluster_details(conn, df_features)
    print("\n=== Scenario 2D (Age + Spending Score, k=4) ===")
    print_customers_by_cluster(details)
    export_clusters_to_excel(details, "customer_clusters_2d.xlsx")


def scenario_3d():
    conn = Connector(database="salesdatabase")
    conn.connect()
    df_features = fetch_customer_features(conn)
    labels, _model, _X = kmeans_cluster(df_features, ['Age', 'Annual Income', 'Spending Score'], n_clusters=5, scale=True)
    df_features['cluster'] = labels
    details = get_cluster_details(conn, df_features)
    print("\n=== Scenario 3D (Age + Annual Income + Spending Score, k=5, scaled) ===")
    print_customers_by_cluster(details)
    export_clusters_to_excel(details, "customer_clusters_3d.xlsx")


if __name__ == "__main__":
    scenario_2d()
    scenario_3d()