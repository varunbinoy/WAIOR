# src/92_course_overlap_network.py
# Elite visual: course overlap network (top N courses by enrollment)
# Nodes=courses, edges weighted by #common students

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

OUT = Path("outputs")
DASH = OUT / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)

TOP_N = 12          # keep 10-15 for readability
MIN_EDGE = 25       # show only strong overlaps (tune: 15/20/25)

def main():
    enroll_path = OUT / "enrollments.csv"
    if not enroll_path.exists():
        raise FileNotFoundError("outputs/enrollments.csv missing. Run your 01_load_data.py first.")

    enroll = pd.read_csv(enroll_path)
    enroll.columns = [c.strip().lower() for c in enroll.columns]
    enroll["student_id"] = enroll["student_id"].astype(str).str.strip()
    enroll["course_id"]  = enroll["course_id"].astype(str).str.strip()
    enroll = enroll.drop_duplicates(["student_id", "course_id"])

    # top N courses by enrollment
    course_sizes = enroll.groupby("course_id")["student_id"].nunique().sort_values(ascending=False)
    top_courses = course_sizes.head(TOP_N).index.tolist()

    e = enroll[enroll["course_id"].isin(top_courses)].copy()

    # Build student -> list of courses
    stu_courses = e.groupby("student_id")["course_id"].apply(list)

    # Pairwise overlaps
    overlap = {}
    for courses in stu_courses:
        courses = sorted(set(courses))
        for i in range(len(courses)):
            for j in range(i+1, len(courses)):
                a, b = courses[i], courses[j]
                overlap[(a, b)] = overlap.get((a, b), 0) + 1

    # Graph
    G = nx.Graph()
    for c in top_courses:
        G.add_node(c, size=int(course_sizes[c]))

    for (a, b), w in overlap.items():
        if w >= MIN_EDGE:
            G.add_edge(a, b, weight=w)

    if G.number_of_edges() == 0:
        print("⚠️ No edges met MIN_EDGE threshold. Lower MIN_EDGE and rerun.")
        print("Try MIN_EDGE=15 or 20.")
        return

    # Positions
    pos = nx.spring_layout(G, seed=7, k=1.0)

    # Node sizes scaled
    node_sizes = [G.nodes[n]["size"] for n in G.nodes]
    node_sizes = [200 + 12*s for s in node_sizes]  # scale up

    # Edge widths scaled by overlap
    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    edge_widths = [0.5 + (w / max(edge_weights)) * 6.0 for w in edge_weights]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.55)

    # Edge labels (only for stronger edges to keep clean)
    strong_edges = {(u, v): G[u][v]["weight"] for u, v in G.edges if G[u][v]["weight"] >= (MIN_EDGE + 10)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=strong_edges, font_size=9)

    title = f"Course Overlap Network (Top {TOP_N} Courses) | Edge>= {MIN_EDGE} common students"
    plt.title(title)
    plt.axis("off")

    out_path = DASH / "course_overlap_network.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    # Print quick ranking of most overlapping pairs
    top_pairs = sorted(overlap.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"✅ Saved: {out_path}")
    print("\nTop 10 overlaps (pair -> common students):")
    for (a, b), w in top_pairs:
        print(f"{a} - {b}: {w}")

if __name__ == "__main__":
    main()
