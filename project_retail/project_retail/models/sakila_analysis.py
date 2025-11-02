import pandas as pd
from typing import Dict, Optional, Tuple

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from project_retail.project_retail.connectors.connector import Connector


def customers_by_film(conn: Connector, film_title: Optional[str] = None) -> Dict[str, pd.DataFrame] | pd.DataFrame:
    """Return customers who rented each film (or a specific film).

    - When film_title is None: returns a dict mapping film title -> DataFrame of unique customers who rented it.
    - When film_title is provided: returns a DataFrame of unique customers who rented that film.
    """
    if film_title:
        sql = (
            """
            SELECT DISTINCT f.title,
                   c.customer_id,
                   c.first_name,
                   c.last_name,
                   c.email
            FROM rental r
            INNER JOIN inventory i ON r.inventory_id = i.inventory_id
            INNER JOIN film f ON i.film_id = f.film_id
            INNER JOIN customer c ON r.customer_id = c.customer_id
            WHERE f.title = %s
            ORDER BY c.last_name, c.first_name
            """
        )
        items = conn.fetchall(sql, (film_title,))
        df = pd.DataFrame(items, columns=["title", "customer_id", "first_name", "last_name", "email"]) if items else pd.DataFrame(columns=["title", "customer_id", "first_name", "last_name", "email"]) 
        # drop duplicates by customer_id
        if not df.empty:
            df = df.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
        return df

    # All films -> unique customers
    sql_all = (
        """
        SELECT f.title,
               c.customer_id,
               c.first_name,
               c.last_name,
               c.email,
               COUNT(*) AS rentals_count
        FROM rental r
        INNER JOIN inventory i ON r.inventory_id = i.inventory_id
        INNER JOIN film f ON i.film_id = f.film_id
        INNER JOIN customer c ON r.customer_id = c.customer_id
        GROUP BY f.title, c.customer_id, c.first_name, c.last_name, c.email
        ORDER BY f.title, c.last_name, c.first_name
        """
    )
    df_all = conn.queryDataset(sql_all)
    if df_all is None or df_all.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for title in sorted(df_all["title"].unique()):
        df_t = df_all[df_all["title"] == title][["customer_id", "first_name", "last_name", "email", "rentals_count"]]
        df_t = df_t.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
        result[title] = df_t
    return result


def customers_by_category(conn: Connector, category_name: Optional[str] = None) -> Dict[str, pd.DataFrame] | pd.DataFrame:
    """Return customers who rented by category. Removes duplicated customers per category.

    - When category_name is None: returns a dict mapping category name -> DataFrame of unique customers.
    - When category_name is provided: returns a DataFrame of unique customers for that category.
    """
    base_sql = (
        """
        SELECT cat.name AS category_name,
               c.customer_id,
               c.first_name,
               c.last_name,
               c.email,
               COUNT(*) AS rentals_count
        FROM rental r
        INNER JOIN inventory i ON r.inventory_id = i.inventory_id
        INNER JOIN film f ON i.film_id = f.film_id
        INNER JOIN film_category fc ON f.film_id = fc.film_id
        INNER JOIN category cat ON fc.category_id = cat.category_id
        INNER JOIN customer c ON r.customer_id = c.customer_id
        {where_clause}
        GROUP BY cat.name, c.customer_id, c.first_name, c.last_name, c.email
        ORDER BY cat.name, c.last_name, c.first_name
        """
    )

    if category_name:
        sql = base_sql.format(where_clause="WHERE cat.name = %s")
        items = conn.fetchall(sql, (category_name,))
        df = pd.DataFrame(items, columns=["category_name", "customer_id", "first_name", "last_name", "email", "rentals_count"]) if items else pd.DataFrame(columns=["category_name", "customer_id", "first_name", "last_name", "email", "rentals_count"]) 
        if not df.empty:
            df = df.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
        return df

    sql_all = base_sql.format(where_clause="")
    df_all = conn.queryDataset(sql_all)
    if df_all is None or df_all.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for cat in sorted(df_all["category_name"].unique()):
        df_c = df_all[df_all["category_name"] == cat][["customer_id", "first_name", "last_name", "email", "rentals_count"]]
        df_c = df_c.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
        result[cat] = df_c
    return result


def build_customer_interest_features(conn: Connector) -> Tuple[pd.DataFrame, list]:
    """Aggregate customer-level interest features from Sakila (film + inventory).

    Features proposed:
    - total_rentals: total transactions.
    - distinct_films: number of different films rented.
    - distinct_categories: number of different categories rented.
    - distinct_stores: number of different stores visited.
    - avg_rental_days: average days per rental (return_date - rental_date).
    - recency_days: days since last rental.
    """
    sql = (
        """
        SELECT c.customer_id,
               CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
               COUNT(r.rental_id) AS total_rentals,
               COUNT(DISTINCT i.film_id) AS distinct_films,
               COUNT(DISTINCT fc.category_id) AS distinct_categories,
               COUNT(DISTINCT i.store_id) AS distinct_stores,
               AVG(DATEDIFF(r.return_date, r.rental_date)) AS avg_rental_days,
               DATEDIFF(CURDATE(), MAX(r.rental_date)) AS recency_days
        FROM customer c
        LEFT JOIN rental r ON r.customer_id = c.customer_id
        LEFT JOIN inventory i ON r.inventory_id = i.inventory_id
        LEFT JOIN film f ON i.film_id = f.film_id
        LEFT JOIN film_category fc ON f.film_id = fc.film_id
        GROUP BY c.customer_id, customer_name
        ORDER BY c.customer_id
        """
    )
    df = conn.queryDataset(sql)
    if df is None:
        df = pd.DataFrame(columns=[
            "customer_id", "customer_name", "total_rentals", "distinct_films",
            "distinct_categories", "distinct_stores", "avg_rental_days", "recency_days"
        ])
    # fill NaNs for math
    for col in ["total_rentals", "distinct_films", "distinct_categories", "distinct_stores", "avg_rental_days", "recency_days"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    feature_cols = [
        "total_rentals", "distinct_films", "distinct_categories",
        "distinct_stores", "avg_rental_days", "recency_days"
    ]
    return df, feature_cols


def kmeans_cluster_customer_interest(conn: Connector, n_clusters: int = 5, scale: bool = True):
    """Run K-Means on proposed interest features and return labels, model, and the feature DataFrame with cluster assignment."""
    df, feature_cols = build_customer_interest_features(conn)
    if df.empty:
        # no data, return empty results
        return pd.Series(dtype=int), None, df

    X = df[feature_cols].values
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels
    return labels, model, df


def get_cluster_details_interest(df_with_labels: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Group customers into clusters, returning a dict[cluster_label] -> DataFrame of customer details + interest metrics."""
    if df_with_labels is None or df_with_labels.empty or "cluster" not in df_with_labels.columns:
        return {}
    result: Dict[int, pd.DataFrame] = {}
    for label in sorted(df_with_labels["cluster"].unique()):
        df_c = df_with_labels[df_with_labels["cluster"] == label].copy()
        df_c = df_c[[
            "customer_id", "customer_name",
            "total_rentals", "distinct_films", "distinct_categories",
            "distinct_stores", "avg_rental_days", "recency_days", "cluster"
        ]]
        result[int(label)] = df_c.reset_index(drop=True)
    return result


def connect_sakila() -> Connector:
    """Helper to create a Connector to the 'sakila' schema with existing credentials."""
    conn = Connector(database="sakila")
    conn.connect()
    return conn