import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as transforms

# =========================
# 1. 全局绘图参数（适配双栏，整体字号缩小）
# =========================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 7                # 基准字号从10降至7
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.8        # 线条变细
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['mathtext.default'] = 'regular'

# =========================
# 2. 数据（保持不变）
# =========================
data = [
    ['PEP',   1.00, 1.00, -5.502176565,  -4.991496952,  'Stable'],
    ['PD',    1.00, 1.00, -0.004417228,  -0.002908508,  'Stable'],
    ['NDVI',  1.00, 0.90, -54.48439574, -10.10520144,   'Stable'],
    ['NMBPC', 1.00, 0.90,  0.737212392,   0.318736742,  'Stable'],
    ['WS',    1.00, 0.60,  3.01601912,    0.539313618,  'Moderately stable'],
    ['POVSI', 1.00, 0.60,  1.464602034,   0.176284284,  'Moderately stable'],
    ['PM10',  0.70, 1.00,  0.006613792,   0.011250546,  'Moderately stable'],
    ['O3',    0.00, 0.84,  0.0,           0.010027468,  'Moderately stable'],
    ['PRE',   1.00, 0.02,  5.072428913,   0.002810457,  'Moderately stable'],
    ['NO2',   1.00, 0.00,  0.20167883,    0.0,          'Moderately stable'],
    ['CO',    1.00, 0.00,  3.167727017,   0.0,          'Moderately stable'],
    ['PCGDP', 1.00, 0.00,  0.000265317,   0.0,          'Moderately stable'],
    ['RH',    0.56, 0.00, -0.077943348,   0.0,          'Moderately stable'],
    ['PM2.5', 0.48, 0.00,  0.014845797,   0.0,          'Unstable'],
    ['SO2',   0.48, 0.00, -0.021971075,   0.0,          'Unstable'],
    ['PGCBA', 0.46, 0.00,  0.060841062,   0.0,          'Unstable'],
    ['NPBMI', 0.00, 0.06,  0.0,           0.0,          'Unstable'],
    ['TEMP',  0.00, 0.00,  0.0,           0.0,          'Unstable'],
    ['URBA',  0.00, 0.00,  0.0,           0.0,          'Unstable'],
    ['PHT',   0.00, 0.00,  0.0,           0.0,          'Unstable'],
    ['TEP',   0.00, 0.00,  0.0,           0.0,          'Unstable'],
]

df = pd.DataFrame(data, columns=[
    'Variable',
    'Frequency_lambda_min',
    'Frequency_lambda_1se',
    'Mean_lambda_min',
    'Mean_lambda_1se',
    'Stability'
])

# =========================
# 3. 分类顺序与配色
# =========================
trend_categories = ['Stable', 'Moderately stable', 'Unstable']

trend_colors = {
    'Stable': '#8E2C2C',
    'Moderately stable': '#1B7F79',
    'Unstable': '#4C5C8A'
}

trend_light_colors = {
    'Stable': '#F4E3E3',
    'Moderately stable': '#DDF1EE',
    'Unstable': '#E4E8F3'
}

# =========================
# 4. 颜色工具函数
# =========================
def blend_with_white(color, blend_factor):
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    blended = rgb * (1 - blend_factor) + white * blend_factor
    return mcolors.to_hex(np.clip(blended, 0, 1))

def create_color_gradient(base_color, n_colors):
    if n_colors == 1:
        return [base_color]
    blend_factors = np.linspace(0.35, 0.0, n_colors)
    return [blend_with_white(base_color, bf) for bf in blend_factors]

# =========================
# 5. 排序
# =========================
grouped_data = {}
for cat in trend_categories:
    sub = df[df['Stability'] == cat].copy()
    sub = sub.sort_values('Mean_lambda_min')
    grouped_data[cat] = sub

# =========================
# 6. 计算共享y位置
# =========================
y_pos = 0
y_ticks = []
y_tick_labels = []
category_positions = []
plot_records = []

for category in trend_categories:
    category_data = grouped_data[category]
    if category_data.empty:
        continue

    n_vars = len(category_data)
    colors = create_color_gradient(trend_colors[category], n_vars)

    start_y = y_pos

    for j, (_, row) in enumerate(category_data.iterrows()):
        plot_records.append({
            'Variable': row['Variable'],
            'Mean_lambda_min': row['Mean_lambda_min'],
            'Mean_lambda_1se': row['Mean_lambda_1se'],
            'Stability': category,
            'y': y_pos,
            'color': colors[j],
            'main_color': trend_colors[category],
            'light_color': trend_light_colors[category]
        })
        y_ticks.append(y_pos)
        y_tick_labels.append(row['Variable'])
        y_pos += 1

    end_y = y_pos - 1

    category_positions.append({
        'category': category,
        'ymin': start_y,
        'ymax': end_y,
        'main_color': trend_colors[category],
        'light_color': trend_light_colors[category]
    })

    y_pos += 0.75

plot_df = pd.DataFrame(plot_records)

# =========================
# 7. 创建双面板图形（整体宽度6.7英寸，双栏）
# =========================
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(4.2, 4.0),          # 宽度6.7英寸（双栏），高度4.8英寸
    sharey=True,
    gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.06}
)

# =========================
# 8. y轴设置（变量名向右靠，pad减小以适配小字号）
# =========================
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels, fontsize=7)   # 字号7pt
ax1.tick_params(axis='y', which='both', left=False, pad=10)  # pad从28减到20

ax2.set_yticks(y_ticks)
ax2.tick_params(axis='y', which='both', left=False, labelleft=False)

# =========================
# 9. 绘制棒棒糖函数（点大小、线宽相应缩小）
# =========================
def draw_lollipop(ax, x_col):
    for _, row in plot_df.iterrows():
        x = row[x_col]
        y = row['y']
        c = row['color']

        ax.plot([0, x], [y, y],
                color=c, linewidth=1.2, alpha=0.95, zorder=2)   # 线宽从2.0降至1.2

        ax.scatter(x, y,
                   s=70,                     # 从118降至70
                   facecolors=c,
                   edgecolors='white',
                   linewidth=0.9,            # 从1.3降至0.9
                   alpha=0.98,
                   zorder=3,
                   clip_on=False)

        ax.scatter(x, y,
                   s=20,                     # 从34降至20
                   facecolors='white',
                   edgecolors='none',
                   zorder=4,
                   clip_on=False)

        if abs(x) < 1e-12:
            ax.scatter(0, y,
                       s=14,                 # 从24降至14
                       facecolors='#444444',
                       edgecolors='white',
                       linewidth=0.6,
                       zorder=5,
                       clip_on=False)

    ax.axvline(x=0, color='#2B2B2B', linestyle='-', linewidth=0.7, alpha=0.9, zorder=1)

draw_lollipop(ax1, 'Mean_lambda_min')
draw_lollipop(ax2, 'Mean_lambda_1se')

# =========================
# 10. x轴：symlog（边距微调）
# =========================
def set_symlog_axis(ax, values):
    ax.set_xscale('symlog', linthresh=0.01, linscale=1.0)

    xticks = [-100, -10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10, 100]
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    valid_xticks = [x for x in xticks if (vmin * 1.6 <= x <= vmax * 1.6) or x == 0]
    if len(valid_xticks) < 5:
        valid_xticks = xticks

    ax.set_xticks(valid_xticks)
    ax.set_xticklabels([str(x) if x != 0 else '0' for x in valid_xticks], fontsize=6.5)  # 刻度字号

    if vmin < 0:
        xmin = vmin * 1.55
    else:
        xmin = -0.02

    if vmax > 0:
        xmax = vmax * 1.55
    else:
        xmax = 0.02

    xmin = min(xmin, -0.03)
    xmax = max(xmax, 0.03)

    ax.set_xlim(xmin, xmax)

set_symlog_axis(ax1, plot_df['Mean_lambda_min'].values)
set_symlog_axis(ax2, plot_df['Mean_lambda_1se'].values)

# =========================
# 11. 左侧稳定性分组块（位置微调，适配更窄的左边距）
# =========================
trans = transforms.blended_transform_factory(ax1.transAxes, ax1.transData)

for item in category_positions:
    y_center = (item['ymin'] + item['ymax']) / 2
    rect_height = (item['ymax'] - item['ymin']) + 0.72

    rect_x = -0.48           # 原 -0.43，因整体左边距增大稍微左移
    rect_w = 0.09

    rect = FancyBboxPatch(
        (rect_x, item['ymin'] - 0.36),
        rect_w,
        rect_height,
        boxstyle="round,pad=0.012",
        transform=trans,
        facecolor=item['light_color'],
        edgecolor=item['main_color'],
        linewidth=0.9,
        alpha=1.0,
        clip_on=False,
        zorder=5
    )
    ax1.add_patch(rect)

    ax1.text(rect_x + rect_w / 2, y_center, item['category'],
             transform=trans,
             fontsize=6.8,
             fontweight='bold',
             color=item['main_color'],
             ha='center',
             va='center',
             rotation=90,
             clip_on=False,
             zorder=6)

# =========================
# 12. 坐标轴标签与美化
# =========================
ax1.set_xlabel(r'Mean coefficient ($\lambda_{min}$)', fontsize=8, labelpad=8)
ax2.set_xlabel(r'Mean coefficient ($\lambda_{1se}$)', fontsize=8, labelpad=8)

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-0.6, y_pos - 0.2)
    ax.grid(False)

# =========================
# 13. 布局调整（左边界增加，为分组块留出空间）
# =========================
plt.subplots_adjust(
    left=0.24,          # 原0.26，稍微收紧
    right=0.97,
    top=0.96,
    bottom=0.12,
    wspace=0.06
)

# =========================
# 14. 保存
# =========================
os.makedirs('output', exist_ok=True)
save_path = 'output/lasso_mean_coefficients_dual_panel_compact.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()

print(f'图片已保存为: {save_path}')