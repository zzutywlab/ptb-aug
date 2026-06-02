import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

output_tiff = "output/lasso_selected_features_bubble_matrix_upgraded.tiff"
output_pdf = "output/lasso_selected_features_bubble_matrix_upgraded.pdf"


regions = [
    "Inner Mongolia", "Ningxia", "Qinghai", "Sichuan",
    "Xinjiang", "Tibet", "Yunnan", "Guangxi",
    "Guizhou", "Gansu", "Chongqing", "Shaanxi"
]

region_labels = regions

selected_features = {
    "Inner Mongolia": ['PM10', 'SO2', 'O3', 'TEMP', 'NPBMI', 'PGCBA', 'TEP', 'PHT'],
    "Ningxia": ['PM10', 'SO2', 'RH', 'PEP', 'POVSI'],
    "Qinghai": ['NO2', 'O3', 'WS', 'PRE', 'PD', 'NPBMI', 'PGCBA'],
    "Sichuan": ['PM2.5', 'O3', 'WS', 'NO2', 'PEP', 'POVSI'],
    "Xinjiang": ['PEP', 'PD', 'NDVI', 'NMBPC', 'WS', 'POVSI', 'PM10', 'O3'],
    "Tibet": ['NO2', 'O3', 'NDVI', 'PM2.5'],
    "Yunnan": ['O3', 'NDVI', 'PD', 'PGCBA'],
    "Guangxi": ['NDVI', 'PRE', 'PEP', 'TEP', 'O3', 'TEMP'],
    "Guizhou": ['O3', 'NDVI', 'PRE', 'WS', 'SO2', 'TEP', 'NPBMI'],
    "Gansu": ['PM2.5', 'SO2', 'O3', 'PD', 'URBA', 'NPBMI'],
    "Chongqing": ['PM2.5', 'O3', 'WS', 'NDVI', 'PRE', 'NPBMI', 'TEP'],
    "Shaanxi": ['PM10', 'O3', 'WS', 'RH', 'CO', 'TEP', 'PEP']
}

natural_features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'WS', 'NDVI', 'PRE', 'RH']
socioeconomic_features = ['PEP', 'PD', 'PCGDP', 'URBA', 'NPBMI', 'PGCBA', 'NMBPC', 'PHT', 'TEP', 'POVSI']
all_features = natural_features + socioeconomic_features

feature_category = {f: 'Natural' for f in natural_features}
feature_category.update({f: 'Socioeconomic' for f in socioeconomic_features})

feature_freq = {f: 0 for f in all_features}
for region in regions:
    for f in selected_features[region]:
        feature_freq[f] += 1

natural_sorted = sorted(natural_features, key=lambda x: (-feature_freq[x], natural_features.index(x)))
socio_sorted = sorted(socioeconomic_features, key=lambda x: (-feature_freq[x], socioeconomic_features.index(x)))
feature_order = natural_sorted + socio_sorted

region_counts = {r: len(selected_features[r]) for r in regions}

records = []
for r in regions:
    for f in feature_order:
        records.append({
            "Region": r,
            "Feature": f,
            "Selected": int(f in selected_features[r]),
            "Category": feature_category[f],
            "FeatureFreq": feature_freq[f],
            "RegionCount": region_counts[r]
        })
df = pd.DataFrame(records)

x_map = {r: i for i, r in enumerate(regions)}
y_map = {f: i for i, f in enumerate(feature_order)}
df["x"] = df["Region"].map(x_map)
df["y"] = df["Feature"].map(y_map)

color_natural = "#2F6C8F"
color_socio = "#C97B63"
color_grid = "#D9D9D9"
color_text = "#222222"
color_topbar = "#5A6B7A"
bg_natural = "#F4F8FB"
bg_socio = "#FCF6F2"

fig = plt.figure(figsize=(4.2, 4.4))
gs = fig.add_gridspec(
    nrows=2, ncols=2,
    height_ratios=[1.3, 8.8],
    width_ratios=[11.2, 2.3],
    hspace=0.035, wspace=0.035
)

ax_top = fig.add_subplot(gs[0, 0])
ax_main = fig.add_subplot(gs[1, 0])
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

n_nat = len(natural_sorted)
n_soc = len(socio_sorted)
n_feat = len(feature_order)
n_reg = len(regions)

ax_main.add_patch(Rectangle(
    (-0.5, -0.5), n_reg, n_nat,
    facecolor=bg_natural, edgecolor='none', zorder=0
))
ax_main.add_patch(Rectangle(
    (-0.5, n_nat - 0.5), n_reg, n_soc,
    facecolor=bg_socio, edgecolor='none', zorder=0
))

for x in range(n_reg + 1):
    ax_main.axvline(x - 0.5, color="#EFEFEF", lw=0.4, zorder=0)
for y in range(n_feat + 1):
    ax_main.axhline(y - 0.5, color="#EFEFEF", lw=0.4, zorder=0)

ax_main.axhline(n_nat - 0.5, color="#B8B8B8", lw=0.8, zorder=1)

ax_main.scatter(
    df["x"], df["y"],
    s=10, facecolors='none', edgecolors=color_grid,
    linewidths=0.4, zorder=1
)

df_sel = df[df["Selected"] == 1].copy()
sel_colors = df_sel["Category"].map({
    "Natural": color_natural,
    "Socioeconomic": color_socio
})

ax_main.scatter(
    df_sel["x"], df_sel["y"],
    s=45, c=sel_colors,
    edgecolors="white", linewidths=0.5, zorder=3
)

ax_main.scatter(
    df_sel["x"], df_sel["y"],
    s=45, facecolors='none',
    edgecolors="#3A3A3A", linewidths=0.25, zorder=4
)

ax_main.set_xlim(-0.5, n_reg - 0.5)
ax_main.set_ylim(-0.5, n_feat - 0.5)
ax_main.invert_yaxis()

ax_main.set_xticks(range(n_reg))
ax_main.set_xticklabels(region_labels, rotation=45, ha='right', fontsize=6, color=color_text)
ax_main.set_yticks(range(n_feat))
ax_main.set_yticklabels(feature_order, fontsize=6, color=color_text)

ax_main.tick_params(axis='x', length=0, pad=6)
ax_main.tick_params(axis='y', length=0)

for spine in ["top", "right"]:
    ax_main.spines[spine].set_visible(False)
ax_main.spines["left"].set_linewidth(0.5)
ax_main.spines["bottom"].set_linewidth(0.5)

ax_main.set_xlabel("")
ax_main.set_ylabel("")

y_center_nat = (n_nat - 1) / 2
y_center_soc = n_nat + (n_soc - 1) / 2

ax_main.text(
    -2.4, y_center_nat,
    "Natural factors",
    va='center', ha='center',
    rotation=90, fontsize=7.5, color=color_natural, fontweight='bold'
)
ax_main.text(
    -2.4, y_center_soc,
    "Socioeconomic factors",
    va='center', ha='center',
    rotation=90, fontsize=7.5, color=color_socio, fontweight='bold'
)

top_values = [region_counts[r] for r in regions]
ax_top.bar(range(n_reg), top_values, color=color_topbar, width=0.65)

for i, v in enumerate(top_values):
    ax_top.text(i, v + 0.08, str(v), ha='center', va='bottom', fontsize=6, color=color_text)

ax_top.set_xlim(-0.5, n_reg - 0.5)
ax_top.set_ylim(0, max(top_values) + 1.2)
ax_top.set_ylabel("Count", fontsize=7, color=color_text)
ax_top.tick_params(axis='x', bottom=False, labelbottom=False)
ax_top.tick_params(axis='y', labelsize=6, colors=color_text)

for spine in ["top", "right"]:
    ax_top.spines[spine].set_visible(False)
ax_top.spines["left"].set_linewidth(0.5)
ax_top.spines["bottom"].set_linewidth(0.5)

right_values = [feature_freq[f] for f in feature_order]
right_colors = [color_natural if feature_category[f] == "Natural" else color_socio for f in feature_order]

ax_right.barh(range(n_feat), right_values, color=right_colors, height=0.65)

for i, v in enumerate(right_values):
    ax_right.text(v + 0.08, i, str(v), va='center', ha='left', fontsize=6, color=color_text)

ax_right.set_xlim(0, max(right_values) + 1.2)
ax_right.set_xlabel("Count", fontsize=7, color=color_text, labelpad=4)
ax_right.tick_params(axis='y', left=False, labelleft=False)
ax_right.tick_params(axis='x', labelsize=6, colors=color_text)

for spine in ["top", "right"]:
    ax_right.spines[spine].set_visible(False)
ax_right.spines["left"].set_linewidth(0.5)
ax_right.spines["bottom"].set_linewidth(0.5)

plt.subplots_adjust(left=0.20, right=0.96, top=0.96, bottom=0.18)

plt.savefig(output_tiff, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.savefig(output_pdf, dpi=600, bbox_inches='tight')
plt.show()