# Structure-preserving data augmentation for region-sensitive infectious disease prediction under data scarcity

This repository contains the code and supporting analysis files for the manuscript:

**Structure-preserving data augmentation for region-sensitive infectious disease prediction under data scarcity**

## Overview

This study develops a region-sensitive, structure-preserving data augmentation framework for infectious disease risk prediction under data scarcity. The empirical case study uses monthly pulmonary tuberculosis incidence in China together with environmental, ecological, socioeconomic, demographic, and healthcare-resource indicators.

<p align="center">
  <img src="fig.png" alt="Overview of the structure-preserving data augmentation framework" width="850">
</p>

<p align="center">
  <b>Figure 1.</b> Overview of the region-sensitive, structure-preserving data augmentation framework.
</p>

## Repository organization

This repository is organized according to the main analysis components of the manuscript. The regional folders contain region-specific modeling, validation, and interpretation results, while the Xinjiang folder also contains the detailed methodological evaluation of the augmentation framework.

| Folder/File | Description |
|---|---|
| `fig.png` | Overview figure of the structure-preserving data augmentation framework. |
| `lasso/` | Region-specific feature-screening results based on repeated LASSO regression. These results were used to define locally relevant predictor sets before data augmentation and prediction modeling. |
| `xinjiang/` | Detailed methodological evaluation using Xinjiang as the case-study region. This folder includes base-learner selection, synthetic-data fidelity assessment, and downstream predictive utility evaluation of the structure-preserving data augmentation strategy. |
| `chongqing/` | Region-wise validation and model interpretation for Chongqing. |
| `gansu/` | Region-wise validation and model interpretation for Gansu. |
| `guangxi/` | Region-wise validation and model interpretation for Guangxi. |
| `guizhou/` | Region-wise validation and model interpretation for Guizhou. |
| `inner mongolia/` | Region-wise validation and model interpretation for Inner Mongolia. |
| `ningxia/` | Region-wise validation and model interpretation for Ningxia. |
| `qinghai/` | Region-wise validation and model interpretation for Qinghai. |
| `shaanxi/` | Region-wise validation and model interpretation for Shaanxi. |
| `sichuan/` | Region-wise validation and model interpretation for Sichuan. |
| `tibet/` | Region-wise validation and model interpretation for Tibet. |
| `yunnan/` | Region-wise validation and model interpretation for Yunnan. |
| `output/` | Aggregated outputs, tables, or figures generated from the regional analyses. |

## Main analysis components

### 1. Region-specific feature screening

The `lasso/` folder contains the region-specific feature-screening results. Repeated LASSO regression was used to identify locally relevant predictors for each region before model training and data augmentation. This step was designed to reduce redundant variables, stabilize small-sample modeling, and preserve regional heterogeneity in predictor selection.

### 2. Base-learner selection

The base-learner selection analysis is provided in the `xinjiang/` folder. This analysis was used to compare candidate machine-learning models and select the base learner for subsequent prediction tasks. Xinjiang was used as the detailed methodological case-study region because it provides a representative setting for evaluating model performance under data scarcity and regional heterogeneity.

### 3. Fidelity and predictive utility of structure-preserving data augmentation

The `xinjiang/` folder also contains the detailed evaluation of the proposed structure-preserving data augmentation strategy. This component assesses whether the generated synthetic samples preserve the statistical structure of the observed training data while improving downstream prediction on held-out observed data.

The evaluation includes:

- synthetic-data fidelity assessment;
- preservation of marginal distributions;
- preservation of variable dependencies;
- preservation of multivariate data structure;
- downstream predictive utility on observed data;
- comparison with benchmark generative models and ablation variants.

This part of the repository corresponds to the methodological evaluation of the proposed augmentation framework.

### 4. Region-wise validation and model interpretation across heterogeneous settings

The region-wise validation and model interpretation analyses are provided in each regional folder:

- `chongqing/`
- `gansu/`
- `guangxi/`
- `guizhou/`
- `inner mongolia/`
- `ningxia/`
- `qinghai/`
- `shaanxi/`
- `sichuan/`
- `tibet/`
- `xinjiang/`
- `yunnan/`

Each folder contains the corresponding region-specific analysis files for model validation and interpretation. These analyses were used to evaluate whether the proposed framework improves regional pulmonary tuberculosis risk prediction under data scarcity and to identify shared and region-specific predictors associated with model-predicted incidence.

## How to navigate the repository

A suggested order for reviewing the repository is:

1. Start with `lasso/` to examine the region-specific feature-screening results.
2. Review `xinjiang/` for base-learner selection and detailed evaluation of synthetic-data fidelity and downstream predictive utility.
3. Review each regional folder for region-wise validation and model interpretation results.
4. Review `output/` for aggregated outputs, tables, or figures generated from the analyses.

## Data note

This repository is intended to support reproducibility of the computational workflow described in the manuscript. No individual-level records or personal identifiers are included.

Some raw datasets used in the study may be third-party datasets and should be obtained from the original data providers described in the manuscript. Processed or derived files are included where redistribution is permitted.

## Code availability

The repository provides the code and supporting analysis files for the main computational components of the study, including feature screening, data augmentation, model selection, predictive validation, synthetic-data evaluation, and model interpretation.

## Citation

If you use this repository, please cite the associated manuscript:

**Structure-preserving data augmentation for region-sensitive infectious disease prediction under data scarcity**

A full citation will be added after publication.
