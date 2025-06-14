from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import numpy as np

dataset = load_dataset("sumuks/UFE-CLEAN-10B", split="train")

fasttext_scores = dataset["openbmb-fasttext-classifier-score"]
finewebedu_classifier_scores = dataset["fineweb-edu-classifier-score"]

# Convert to numpy arrays for normalization
fasttext_scores = np.array(fasttext_scores)
finewebedu_classifier_scores = np.array(finewebedu_classifier_scores)

# Min-max normalize to [0, 1]
ft_min, ft_max = fasttext_scores.min(), fasttext_scores.max()
fasttext_scores = (fasttext_scores - ft_min) / (ft_max - ft_min)

fw_min, fw_max = finewebedu_classifier_scores.min(), finewebedu_classifier_scores.max()
finewebedu_classifier_scores = (finewebedu_classifier_scores - fw_min) / (fw_max - fw_min)


# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

# Left: overlapping normalized histograms
ax1.hist(fasttext_scores, bins=50, density=True, alpha=0.5, label='OpenBMB FastText')
ax1.hist(finewebedu_classifier_scores, bins=50, density=True, alpha=0.5, label='Hugging Face FineWebEdu')
ax1.set_xlabel('Normalized Score')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Classifier Scores')
ax1.legend(loc='best')

# Right: CDF curves
ft_sorted = np.sort(fasttext_scores)
ft_cdf = np.arange(1, len(ft_sorted) + 1) / len(ft_sorted)
fw_sorted = np.sort(finewebedu_classifier_scores)
fw_cdf = np.arange(1, len(fw_sorted) + 1) / len(fw_sorted)
ax2.plot(ft_sorted, ft_cdf, linestyle='--', label='FastText CDF')
ax2.plot(fw_sorted, fw_cdf, linestyle='--', label='FineWebEdu CDF')
ax2.set_xlabel('Normalized Score')
ax2.set_ylabel('CDF')
ax2.set_title('CDF of Classifier Scores')
ax2.legend(loc='best')

fig.tight_layout()

plt.savefig(os.path.join('plots', 'distribution.png'), dpi=300)

# Combined score line plots: density and CDF side by side
combined_scores = (fasttext_scores + finewebedu_classifier_scores) / 2

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

# Density line plot
counts, bins = np.histogram(combined_scores, bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax1.plot(bin_centers, counts, label='Combined Score')
ax1.set_xlabel('Normalized Combined Score')
ax1.set_ylabel('Density')
ax1.set_title('Density of Combined Scores')
ax1.legend()

# CDF line plot
sorted_scores = np.sort(combined_scores)
cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
ax2.plot(sorted_scores, cdf, label='Combined Score')
ax2.set_xlabel('Normalized Combined Score')
ax2.set_ylabel('CDF')
ax2.set_title('CDF of Combined Scores')
ax2.legend()
fig.tight_layout()
plt.savefig(os.path.join('plots', 'combined_score.png'), dpi=300)

# Correlation scatter‑plot between the two classifiers
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(fasttext_scores, finewebedu_classifier_scores, s=4, alpha=0.4)
rho = np.corrcoef(fasttext_scores, finewebedu_classifier_scores)[0, 1]
ax.set_xlabel('FastText Normalized Score')
ax.set_ylabel('FineWebEdu Normalized Score')
ax.set_title(f'Score Correlation (ρ = {rho:.2f})')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'score_correlation.png'), dpi=300)
