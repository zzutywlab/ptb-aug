import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==================== 基本设置 ====================
csv_path = "data/output_9regions_cv/nine_regions_10models_5fold_mean_r2.csv"
output_path = "data/output_9regions_cv/nine_regions_10models_5fold_lineplot.tiff"

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['figure.dpi'] = 300

df = pd.read_csv(csv_path)

rename_dict = {
    'LinearRegression': 'LR',
    'DecisionTree': 'DT',
    'SVR': 'SVR',
    'KNN': 'KNN',
    'RandomForest': 'RF',
    'GradientBoosting': 'GBDT',
    'XGBoost': 'XGB',
    'LightGBM': 'LGBM',
    'CatBoost': 'CB',
    'MLP': 'MLP'
}

model_order = [
    'LinearRegression', 'DecisionTree', 'SVR', 'KNN', 'RandomForest',
    'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP'
]

model_order = [m for m in model_order if m in df.columns]
model_labels = [rename_dict[m] for m in model_order]

# ==================== 地区名称英文映射 ====================
region_rename = {
    '内蒙古': 'Inner Mongolia',
    '宁夏': 'Ningxia',
    '青海': 'Qinghai',
    '四川': 'Sichuan',
    '新疆': 'Xinjiang',
    '西藏': 'Tibet',
    '云南': 'Yunnan',
    '广西': 'Guangxi',
    '贵州': 'Guizhou',
    '甘肃': 'Gansu',
    '重庆': 'Chongqing',
    '陕西': 'Shaanxi'
}

df['Region_Eng'] = df['Region'].map(region_rename).fillna(df['Region'])

region_colors = {
    'Inner Mongolia': '#1F4E79',
    'Ningxia': '#4C78A8',
    'Qinghai': '#72B7B2',
    'Sichuan': '#54A24B',
    'Xinjiang': '#E39C45',
    'Tibet': '#C76E6E',
    'Yunnan': '#B279A2',
    'Guangxi': '#8C8C8C',
    'Guizhou': '#C0A43A',
    'Gansu': '#8E6C8A',
    'Chongqing': '#D16D5B',
    'Shaanxi': '#5B8C5A'
}

fig, ax = plt.subplots(figsize=(6.7, 4.2))

for _, row in df.iterrows():
    region = row['Region_Eng']
    y = row[model_order].values.astype(float)
    ax.plot(
        model_labels,
        y,
        marker='o',
        markersize=4.5,
        linewidth=1.5,
        color=region_colors.get(region, '#333333'),
        label=region,
        alpha=0.95
    )

ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel(r"R²", fontsize=10, fontweight='normal')

ax.tick_params(axis='x', labelsize=9, width=0.8, length=4)
ax.tick_params(axis='y', labelsize=9, width=0.8, length=4)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('normal')
    label.set_fontname('Arial')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.9)
ax.spines['bottom'].set_linewidth(0.9)

legend = ax.legend(
    loc='lower left',
    frameon=False,
    fontsize=9,
    handlelength=2.0,
    handletextpad=0.6,
    ncol=2
)
for text in legend.get_texts():
    text.set_fontweight('normal')
    text.set_fontname('Arial')

plt.tight_layout()

plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
plt.savefig(output_path.replace('.tiff', '.png'), dpi=600, bbox_inches='tight')

pdf_path = output_path.replace('.tiff', '.pdf')
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')

plt.show()

print(f"图像已保存到: {output_path}")
print(f"PNG版本已保存到: {output_path.replace('.tiff', '.png')}")
print(f"矢量PDF版本已保存到: {pdf_path}")