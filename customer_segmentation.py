from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_churn.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = ["TenureMonths", "MonthlyCharges", "TotalCharges", "SupportCalls"]
X = df[features].fillna(df[features].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scores = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    scores.append({"k": k, "silhouette_score": silhouette_score(X_scaled, labels)})
score_df = pd.DataFrame(scores)
score_df.to_csv(OUT / "silhouette_scores.csv", index=False)
best_k = int(score_df.loc[score_df["silhouette_score"].idxmax(), "k"])

# Use the best silhouette score from k=2..6
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Segment"] = kmeans.fit_predict(X_scaled)
df.to_csv(OUT / "customer_segments.csv", index=False)

summary = df.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    AvgTenureMonths=("TenureMonths", "mean"),
    AvgMonthlyCharges=("MonthlyCharges", "mean"),
    AvgTotalCharges=("TotalCharges", "mean"),
    AvgSupportCalls=("SupportCalls", "mean"),
    ChurnRate=("Churn", "mean")
).round(2)
summary.to_csv(OUT / "segment_summary.csv")

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Segment": df["Segment"].astype(str)})
fig, ax = plt.subplots(figsize=(9, 6))
for segment, group in plot_df.groupby("Segment"):
    ax.scatter(group["PC1"], group["PC2"], label=f"Segment {segment}", alpha=0.65, s=25)
ax.set_title("Customer Segments (PCA 2D View)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "customer_segments_pca.png", dpi=150)
plt.close(fig)

print("Customer segmentation completed.")
print(f"Selected k: {best_k}")
print("\nSilhouette scores:\n", score_df.to_string(index=False))
print("\nSegment summary:\n", summary)
print(f"\nOutputs saved to: {OUT}")
