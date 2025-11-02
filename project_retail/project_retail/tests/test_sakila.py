import argparse
from pprint import pprint

from project_retail.project_retail.models.sakila_analysis import (
    connect_sakila,
    customers_by_film,
    customers_by_category,
    kmeans_cluster_customer_interest,
    get_cluster_details_interest,
)


def run_film(conn, film_title: str | None):
    if film_title:
        df = customers_by_film(conn, film_title)
        print(f"\nCustomers who rented film '{film_title}':")
        print(df.to_string(index=False))
    else:
        mapping = customers_by_film(conn)
        count = 0
        for title, df in mapping.items():
            print(f"\nFilm: {title} — {len(df)} unique customers")
            print(df.head(10).to_string(index=False))
            count += 1
            if count >= 5:  # show first 5 films for brevity
                break


def run_category(conn, category_name: str | None):
    if category_name:
        df = customers_by_category(conn, category_name)
        print(f"\nCustomers who rented category '{category_name}':")
        print(df.to_string(index=False))
    else:
        mapping = customers_by_category(conn)
        count = 0
        for cat, df in mapping.items():
            print(f"\nCategory: {cat} — {len(df)} unique customers")
            print(df.head(10).to_string(index=False))
            count += 1
            if count >= 5:  # show first 5 categories
                break


def run_cluster(conn, k: int, scale: bool):
    labels, model, df = kmeans_cluster_customer_interest(conn, n_clusters=k, scale=scale)
    if df.empty:
        print("No data available to cluster.")
        return
    print(f"\nK-Means on customer interest features: k={k}, scale={scale}")
    print("Cluster counts:")
    print(df["cluster"].value_counts().sort_index())

    details = get_cluster_details_interest(df)
    for label, cdf in details.items():
        print(f"\nCluster {label} — {len(cdf)} customers")
        print(cdf.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Sakila analysis: film/category classification and K-Means clustering")
    parser.add_argument("--film", dest="film_title", default=None, help="Film title to list customers for (optional)")
    parser.add_argument("--category", dest="category_name", default=None, help="Category name to list customers for (optional)")
    parser.add_argument("--cluster", action="store_true", help="Run K-Means clustering on customer interest features")
    parser.add_argument("-k", dest="k", type=int, default=5, help="Number of clusters for K-Means")
    parser.add_argument("--scale", dest="scale", choices=["true", "false"], default="true", help="Scale features before K-Means")
    args = parser.parse_args()

    conn = connect_sakila()

    if args.film_title is not None:
        run_film(conn, args.film_title)
    else:
        print("\n[Film Classification] Showing first 5 films and their unique customers:")
        run_film(conn, None)

    if args.category_name is not None:
        run_category(conn, args.category_name)
    else:
        print("\n[Category Classification] Showing first 5 categories and their unique customers:")
        run_category(conn, None)

    if args.cluster:
        run_cluster(conn, args.k, args.scale.lower() == "true")


if __name__ == "__main__":
    main()