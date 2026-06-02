import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ============================== 读取数据 ==============================
df = pd.read_csv('output/moxingduibi/cross_validation_r2.csv')

# 原始名称替换为缩写
rename_dict = {
    'LinearRegression': 'LR',
    'DecisionTree': 'DT',
    'SVR': 'SVR',
    'KNN': 'KNN',
    'RandomForest': 'RF',
    'GradientBoosting': 'GBDT',
    'XGBoost': 'XGB',
    'CatBoost': 'CB',
    'MLP': 'MLP'
}

df = df.rename(columns=rename_dict)

# 按替换后的列顺序绘图
model_order = df.columns.tolist()

# ============================== 计算中位数 ==============================
medians = df.median().reindex(model_order)

# ============================== 颜色设计 ==============================
anchor_colors = [
    "#163b65",  # 深蓝
    "#2f6c8f",  # 蓝青
    "#7fb8c9",  # 浅青蓝
    "#d9eef0",  # 极浅青
    "#f5ece3",  # 米色
    "#f2a97f",  # 杏橙
    "#de5d4f",  # 暖红
    "#7d1022"   # 深红
]

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

def rgb_to_hex(rgb):
    rgb = np.clip(rgb, 0, 1)
    return '#%02x%02x%02x' % tuple((rgb * 255).astype(int))

def interpolate_palette(anchor_hex_list, n):
    anchors = [hex_to_rgb(c) for c in anchor_hex_list]
    if n == 1:
        return [anchor_hex_list[0]]
    positions = np.linspace(0, len(anchors) - 1, n)
    colors = []
    for p in positions:
        left = int(np.floor(p))
        right = min(left + 1, len(anchors) - 1)
        t = p - left
        rgb = anchors[left] * (1 - t) + anchors[right] * t
        colors.append(rgb_to_hex(rgb))
    return colors

model_colors = interpolate_palette(anchor_colors, len(model_order))

# ============================== 全局风格（字号适配双栏宽度） ==============================
sns.set_style("white")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9          # 全局基准字号调小
plt.rcParams['axes.labelsize'] = 10    # 轴标签字号
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.linewidth'] = 0.9   # 坐标轴线宽

# ============================== 创建画布（双栏宽度，约17 cm） ==============================
fig, ax = plt.subplots(figsize=(6.7, 4.0), facecolor='white')   # 宽度 6.7 英寸，高度略高于折线图
ax.set_facecolor('white')

# ============================== 绘制箱线图 ==============================
data_list = [df[col].dropna().values for col in model_order]

box_parts = ax.boxplot(
    data_list,
    positions=np.arange(len(model_order)),
    widths=0.55,                   # 宽度略减以留出间距
    patch_artist=True,
    showmeans=False,
    showfliers=False,
    medianprops=dict(color='#3a3a3a', linewidth=1.2),
    whiskerprops=dict(color='#8a8a8a', linewidth=0.9),
    capprops=dict(color='#8a8a8a', linewidth=0.9),
    boxprops=dict(edgecolor='#555555', linewidth=0.9)
)

for patch, color in zip(box_parts['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.38)
    patch.set_edgecolor(color)
    patch.set_linewidth(1.0)

for whisker in box_parts['whiskers']:
    whisker.set_color('#9a9a9a')
for cap in box_parts['caps']:
    cap.set_color('#9a9a9a')

# ============================== 叠加散点 ==============================
np.random.seed(42)

for i, col in enumerate(model_order):
    y = df[col].dropna().values
    x = np.random.normal(loc=i, scale=0.06, size=len(y))  # 抖动幅度略调
    ax.scatter(
        x, y,
        s=12,                       # 原 18，调小
        color=model_colors[i],
        alpha=0.45,
        edgecolors='none',
        zorder=2
    )

# ============================== 中位数连线 ==============================
median_vals = medians.values

ax.plot(
    np.arange(len(model_order)),
    median_vals,
    color='#666666',
    linestyle='-',
    linewidth=1.0,
    zorder=3
)

ax.scatter(
    np.arange(len(model_order)),
    median_vals,
    s=16,                           # 原22，调小
    facecolor='white',
    edgecolor='#666666',
    linewidth=0.8,
    zorder=4
)

# ============================== 标注中位数数值 ==============================
for i, val in enumerate(median_vals):
    ax.text(
        i,
        val + 0.035,
        f'{val:.3f}',
        ha='center',
        va='bottom',
        fontsize=7.5,               # 原9，调小
        color='#4d4d4d'
    )

# ============================== 坐标轴设置 ==============================
ax.set_xticks(np.arange(len(model_order)))
ax.set_xticklabels(model_order, rotation=0, ha='center')

ax.set_xlabel('')
ax.set_ylabel('Cross-validation R²', fontsize=10)

ax.set_ylim(-1, 1)
ax.set_yticks(np.arange(-1.0, 1.01, 0.2))

ax.grid(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.9)
ax.spines['bottom'].set_linewidth(0.9)

ax.tick_params(axis='x', length=5, width=0.9, direction='out', pad=8)
ax.tick_params(axis='y', length=5, width=0.9, direction='out')

# ============================== 图例 ==============================
legend_elements = [
    Line2D([0], [0],
           color='#666666', linestyle='-',
           marker='o', markerfacecolor='white', markeredgecolor='#666666',
           markersize=4, linewidth=1.0, label='Median'),
    Line2D([0], [0],
           marker='o', color='w',
           markerfacecolor='#9a9a9a', alpha=0.45, markersize=5,
           label='Observations')
]

ax.legend(
    handles=legend_elements,
    loc='lower left',
    frameon=False,
    handlelength=1.5,
    fontsize=8
)

# ============================== 布局与保存 ==============================
plt.tight_layout()

# 保存位图（PNG，600 DPI）
plt.savefig('output/boxplot_r2_comparison_pubhealth_style_labeled_abbr.png',
            dpi=600, bbox_inches='tight', facecolor='white')

# 额外保存矢量 PDF 格式
plt.savefig('output/boxplot_r2_comparison_pubhealth_style_labeled_abbr.pdf',
            dpi=300, bbox_inches='tight', format='pdf')

plt.show()