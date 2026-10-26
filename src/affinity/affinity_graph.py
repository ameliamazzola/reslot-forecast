# %%
from pathlib import Path
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(r"C:\Users\ameli\projects\reslot-forecast")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

pairs = pd.read_pickle(PROCESSED_DIR / "frequent_pairs.pkl")
N_ORDERS = len(pd.read_pickle(PROCESSED_DIR / "basket_matrix.pkl"))

files = ["yowmspd2_archive_picks_past_365days_ranon9.21.26.csv",
         "yowmspd2__picks_past_365days_ranon9.21.26.csv"]
descr = (pd.concat([pd.read_csv(RAW_DIR / f, usecols=["FROM_ITEM", "FROM_DESCR"],
                                dtype={"FROM_ITEM": str}) for f in files])
         .drop_duplicates("FROM_ITEM").set_index("FROM_ITEM")["FROM_DESCR"])

print("itemsets:", len(pairs), "| orders:", N_ORDERS)
# %%
singles = pairs[pairs["length"] == 1].copy()
singles["sku"] = singles["itemsets"].apply(lambda s: next(iter(s)))
sku_support = singles.set_index("sku")["support"]

edges = pairs[pairs["length"] == 2].copy()
edges[["sku_a", "sku_b"]] = edges["itemsets"].apply(lambda s: pd.Series(sorted(s)))

print("nodes:", len(sku_support), "| edges:", len(edges))
# %%
s_a = edges["sku_a"].map(sku_support)
s_b = edges["sku_b"].map(sku_support)

edges["jaccard"] = edges["support"] / (s_a + s_b - edges["support"])
edges["orders"] = (edges["support"] * N_ORDERS).round().astype(int)

print(edges["jaccard"].describe())
# %%
G = nx.Graph()

for sku, sup in sku_support.items():
    G.add_node(sku, support=sup, orders=round(sup * N_ORDERS), descr=descr.get(sku, "?"))

for row in edges.itertuples():
    G.add_edge(row.sku_a, row.sku_b, weight=row.jaccard, orders=row.orders)

print("nodes:", G.number_of_nodes(), "| edges:", G.number_of_edges())
print("SKUs with no strong partner:", nx.number_of_isolates(G))
# %%
top_sku = edges.sort_values("support", ascending=False).iloc[0]
a, b = top_sku["sku_a"], top_sku["sku_b"]

print(G.nodes[a])
print(G.edges[a, b])

top_connected = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
for sku, deg in top_connected:
    print(deg, sku, G.nodes[sku]["descr"])
# %%
MIN_JACCARD = 0.10

strong_edges = [(u, v) for u, v, w in G.edges.data("weight") if w >= MIN_JACCARD]
H = G.edge_subgraph(strong_edges).copy()

print("nodes:", H.number_of_nodes(), "| edges:", H.number_of_edges())
print("disconnected groups:", nx.number_connected_components(H))
# %%
print([len(c) for c in sorted(nx.connected_components(H), key=len, reverse=True)])
print("Peroxicare before:", G.degree["20018590"], "| after:", H.degree["20018590"])
# %%
components = sorted(nx.connected_components(H), key=len, reverse=True)
for i, comp in enumerate(components):
    top = sorted(comp, key=lambda s: G.nodes[s]["orders"], reverse=True)[:6]
    print(f"\n--- Group {i} ({len(comp)} SKUs) ---")
    for sku in top:
        print(G.nodes[sku]["orders"], G.nodes[sku]["descr"])
# %%
clusters = nx.community.louvain_communities(H, weight="weight", resolution=1.0, seed=42)
clusters = sorted(clusters, key=len, reverse=True)

print("number of clusters:", len(clusters))
print("sizes:", [len(c) for c in clusters])
print("modularity:", round(nx.community.modularity(H, clusters, weight="weight"), 3))

# %%
rows = [{"cluster": i, "sku": sku, "descr": G.nodes[sku]["descr"], "orders": G.nodes[sku]["orders"]}
        for i, cluster in enumerate(clusters) for sku in cluster]
cluster_table = pd.DataFrame(rows).sort_values(["cluster", "orders"], ascending=[True, False])

for i in range(len(clusters)):
    print(f"\n--- Cluster {i} ({len(clusters[i])} SKUs) ---")
    print(cluster_table[cluster_table["cluster"] == i].head(8)[["orders", "descr"]].to_string(index=False))

cluster_table.to_csv(PROCESSED_DIR / "sku_clusters.csv", index=False)

# %%
picks_dt = pd.concat([pd.read_csv(RAW_DIR / f, usecols=["ORD_NO", "FROM_ITEM", "CREATE_DT"],
                                  dtype={"ORD_NO": str, "FROM_ITEM": str}) for f in files])
picks_dt["month"] = pd.to_datetime(picks_dt["CREATE_DT"], format="%m/%d/%Y %I:%M:%S %p").dt.to_period("M")

clusters_saved = pd.read_csv(PROCESSED_DIR / "sku_clusters.csv", dtype={"sku": str})
cluster_skus = clusters_saved.loc[clusters_saved["cluster"] == 1, "sku"]

c1_lines = picks_dt[picks_dt["FROM_ITEM"].isin(cluster_skus)].groupby("month").size()
total_lines = picks_dt.groupby("month").size()
share = (c1_lines / total_lines * 100).round(2)

print(pd.DataFrame({"cluster_1": c1_lines, "all_DC": total_lines, "share_%": share}))

# %%
all_skus = pd.read_pickle(PROCESSED_DIR / "basket_matrix.pkl").columns

labels = pd.DataFrame({"sku": all_skus})
labels["cluster"] = labels["sku"].map(dict(zip(cluster_table["sku"], cluster_table["cluster"])))
labels["group"] = "clustered"
labels.loc[labels["cluster"].isna() & labels["sku"].isin(G.nodes), "group"] = "no strong partner"
labels.loc[~labels["sku"].isin(G.nodes), "group"] = "below min support"
labels["cluster"] = labels["cluster"].fillna(-1).astype(int)
labels["descr"] = labels["sku"].map(descr)

print(labels["group"].value_counts())
labels.to_csv(PROCESSED_DIR / "sku_labels.csv", index=False)


# %%
lines_per_sku = picks_dt.groupby("FROM_ITEM").size()

summary = (cluster_table
           .assign(pick_lines=cluster_table["sku"].map(lines_per_sku))
           .groupby("cluster")
           .agg(skus=("sku", "size"), pick_lines=("pick_lines", "sum"), top_sku=("descr", "first")))

summary["name"] = summary.index.map({
    0: "Walmart personal-care assortment", 1: "Cold & wellness", 2: "Core liquid laundry",
    3: "Batiste", 4: "Laundry boosters / large formats", 5: "Cat litter",
    6: "Trojan (+ Orajel gel)", 7: "Platinum litter 37 lb"})
summary = summary.sort_values("pick_lines", ascending=False)

print(summary.to_string())
summary.to_csv(PROCESSED_DIR / "cluster_summary.csv")
# %%
labels["pick_lines"] = labels["sku"].map(lines_per_sku).fillna(0)
print(labels.groupby("group")["pick_lines"].sum() / labels["pick_lines"].sum())
# %%
picks_uom = pd.concat([pd.read_csv(RAW_DIR / f, usecols=["FROM_ITEM", "FROM_UOM"],
                                   dtype={"FROM_ITEM": str}) for f in files])

sku_to_cluster = dict(zip(cluster_table["sku"], cluster_table["cluster"]))
uom_mix = (picks_uom.assign(cluster=picks_uom["FROM_ITEM"].map(sku_to_cluster))
           .dropna(subset=["cluster"])
           .groupby(["cluster", "FROM_UOM"]).size()
           .unstack(fill_value=0))

print((uom_mix.div(uom_mix.sum(axis=1), axis=0) * 100).round(1))
# %%
from sklearn.metrics import adjusted_rand_score

def run_clustering(min_jaccard, resolution):
    keep = [(u, v) for u, v, w in G.edges.data("weight") if w >= min_jaccard]
    sub = G.edge_subgraph(keep).copy()
    comms = nx.community.louvain_communities(sub, weight="weight", resolution=resolution, seed=42)
    return {sku: i for i, c in enumerate(comms) for sku in c}, len(comms)

baseline, _ = run_clustering(0.10, 1.0)

results = []
for min_j in [0.10, 0.15, 0.20]:
    for res in [0.8, 1.0, 1.5]:
        labels_alt, n = run_clustering(min_j, res)
        common = sorted(set(baseline) & set(labels_alt))
        ari = adjusted_rand_score([baseline[s] for s in common], [labels_alt[s] for s in common])
        results.append({"min_jaccard": min_j, "resolution": res, "clusters": n, "ARI_vs_baseline": round(ari, 3)})

sensitivity = pd.DataFrame(results)
print(sensitivity.to_string(index=False))
sensitivity.to_csv(PROCESSED_DIR / "cluster_sensitivity.csv", index=False)
# %%
import matplotlib.pyplot as plt

pos = nx.spring_layout(H, weight="weight", seed=42)
node_cluster = {sku: i for i, c in enumerate(clusters) for sku in c}
colors = plt.cm.tab10.colors

plt.figure(figsize=(12, 9))
nx.draw_networkx(H, pos, node_color=[colors[node_cluster[n]] for n in H.nodes],
                 node_size=30, width=0.2, with_labels=False)

handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[i],
                      label=f"{i}: {summary.loc[i, 'name']}") for i in range(len(clusters))]
plt.legend(handles=handles, loc="lower right", fontsize=9)
plt.axis("off")
plt.savefig(PROCESSED_DIR / "affinity_graph.png", dpi=200, bbox_inches="tight")
plt.show()


# %%
