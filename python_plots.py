# Cognitive Cluster plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("ally.csv")

# Keep only valid A-D responses
df = df[df["option_selected"].isin(list("ABCD"))].copy()

# Majority vote option per (model, question_code)
def majority_vote(series):
    vc = series.value_counts()
    # 3 runs: 3-0, 2-1, or 1-1-1
    if len(vc) == 3 and vc.iloc[0] == 1:  # 1-1-1
        return np.nan
    return vc.index[0]

maj = (
    df.groupby(["model", "question_code"])["option_selected"]
      .apply(majority_vote)
      .reset_index(name="maj_option")
)

# Correct key per question
key = df.groupby("question_code")["correct_option"].first().reset_index()
maj = maj.merge(key, on="question_code", how="left")

# Majority-vote correctness (ties counted as incorrect)
maj["maj_correct"] = (maj["maj_option"] == maj["correct_option"]).astype(float)
maj.loc[maj["maj_option"].isna(), "maj_correct"] = 0.0

# Order models by majority-vote accuracy
acc = maj.groupby("model")["maj_correct"].mean().sort_values(ascending=False)
ordered_models = acc.index.tolist()

# Build question x model matrix of majority options
mat = maj.pivot(index="question_code", columns="model", values="maj_option").reindex(columns=ordered_models)

# Pairwise similarity (agreement rate on overlapping non-missing)
sim = pd.DataFrame(index=ordered_models, columns=ordered_models, dtype=float)
for m1 in ordered_models:
    a = mat[m1]
    for m2 in ordered_models:
        b = mat[m2]
        mask = a.notna() & b.notna()
        sim.loc[m1, m2] = np.nan if mask.sum() == 0 else (a[mask].values == b[mask].values).mean()

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 8), dpi=220)
im = ax.imshow(sim.values, vmin=0, vmax=1, cmap="coolwarm", origin="upper")

ax.set_xticks(np.arange(len(ordered_models)))
ax.set_yticks(np.arange(len(ordered_models)))
ax.set_xticklabels(ordered_models, rotation=90, ha="center")
ax.set_yticklabels(ordered_models)

# Annotate cells
for i in range(len(ordered_models)):
    for j in range(len(ordered_models)):
        val = sim.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Response similarity (agreement rate)")

ax.set_title('The "Cognitive Cluster" (Model Response Similarity)\nOrdered by majority-vote accuracy (highest → lowest)')

plt.tight_layout()

out_path = "cognitive_cluster_ordered_by_accuracy.png"
plt.savefig(out_path, bbox_inches="tight")
plt.show()

out_path

# Ensemble Plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ally.csv")

# Keep only standard A-D outputs for voting
df = df[df["option_selected"].isin(list("ABCD"))].copy()

# Raw (unadjusted) accuracy per model across all runs
acc_raw = df.groupby("model")["marks"].mean().sort_values(ascending=False)

# Majority option per (question_code, model) across 3 runs
def majority_vote(series):
    vc = series.value_counts()
    # 3 runs: 3-0, 2-1, or 1-1-1
    if len(vc) == 3 and vc.iloc[0] == 1:  # 1-1-1
        return np.nan
    return vc.index[0]

maj = (
    df.groupby(["question_code", "model"])["option_selected"]
      .apply(majority_vote)
      .reset_index(name="maj_option")
)

# Correct answer per question_code
key = df.groupby("question_code")["correct_option"].first()

# Choose top-N models by raw accuracy for the ensemble
TOP_N = 5
top_models = acc_raw.index.tolist()[:TOP_N]

# Ensemble majority vote across selected models (ties -> NA)
def ensemble_vote_for_question(group):
    opts = group["maj_option"].dropna()
    if len(opts) == 0:
        return np.nan
    vc = opts.value_counts()
    # tie among top choices
    if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
        return np.nan
    return vc.index[0]

ens = (
    maj[maj["model"].isin(top_models)]
    .groupby("question_code")
    .apply(ensemble_vote_for_question)
    .reset_index(name="ens_option")
)
ens["correct_option"] = ens["question_code"].map(key)
ens["ens_correct"] = (ens["ens_option"] == ens["correct_option"]).astype(float)

# Treat NA ensemble choices as incorrect (0)
ens.loc[ens["ens_option"].isna(), "ens_correct"] = 0.0
ensemble_acc = ens["ens_correct"].mean()

# Build plotting dataframe: models + ensemble row
plot_df = (
    acc_raw.reset_index()
    .rename(columns={"model": "Model", "marks": "Accuracy"})
)

plot_df = pd.concat([
    pd.DataFrame({"Model": ["Ensemble (Majority Vote)"], "Accuracy": [ensemble_acc]}),
    plot_df
], ignore_index=True)

# Sort by accuracy (descending) so best is on top
plot_df = plot_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

# Colors: grey for models, red for ensemble
colors = ["#d62728" if m == "Ensemble (Majority Vote)" else "#7f7f7f" for m in plot_df["Model"]]

# Plot
fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=220)
ax.barh(plot_df["Model"], plot_df["Accuracy"], color=colors, edgecolor="none")
ax.invert_yaxis()

ax.set_xlabel("Accuracy")
ax.set_ylabel("Model")

title = f'Figure: The "Ensemble Effect" (Accuracy = {ensemble_acc*100:.2f}%)'
ax.set_title(title)

# tidy look
ax.set_xlim(0, min(1.0, plot_df["Accuracy"].max() + 0.05))
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()

out_path = "ensemble_effect.png"
plt.savefig(out_path, bbox_inches="tight")
plt.show()

out_path

# Option Bias Plot

import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Load model option distribution
df_opt = pd.read_csv("opt_bias.csv")

# Ground-truth distribution (unique questions)
ally = pd.read_csv("ally.csv")
truth = ally[["question_code", "correct_option"]].drop_duplicates()
truth_p = truth["correct_option"].value_counts(normalize=True).reindex(list("ABCD")).fillna(0)

# Build bias matrix: (model preference - ground truth)
models_in_order = df_opt["model"].drop_duplicates().tolist()
bias_mat = (
    df_opt.pivot(index="model", columns="option_selected", values="p")
      .reindex(index=models_in_order, columns=list("ABCD"))
      .fillna(0)
)

bias = bias_mat.copy()
for opt in list("ABCD"):
    bias[opt] = bias[opt] - truth_p[opt]

# Plot heatmap (matplotlib)
fig, ax = plt.subplots(figsize=(12.5, 6.5), dpi=220)

# Symmetric color limits around 0 for a balanced diverging map
max_abs = float(np.nanmax(np.abs(bias.values)))
vlim = max(0.01, max_abs)

im = ax.imshow(bias.values, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")

# Ticks and labels
ax.set_xticks(np.arange(len(bias.columns)))
ax.set_xticklabels(bias.columns)
ax.set_yticks(np.arange(len(bias.index)))
ax.set_yticklabels(bias.index)

ax.set_xlabel("Option Choice")
ax.set_ylabel("Model")
ax.set_title("Figure: Option Bias (Model Preference vs Ground Truth)")

# Annotate cells as percentages
for i in range(bias.shape[0]):
    for j in range(bias.shape[1]):
        val = bias.values[i, j]
        txt = f"{val*100:.1f}%"
        # choose text color for readability
        color = "white" if abs(val) > 0.06 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Bias (model proportion − ground-truth proportion)")

plt.tight_layout()

out_path = "option_bias_heatmap.png"
plt.savefig(out_path, bbox_inches="tight")
plt.show()

out_path

# Family/year Wise Accuracy Plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("ally.csv")

# Basic cleaning
df = df.copy()
df["neet_year"] = df["neet_year"].astype(int)
df = df[df["marks"].isin([0, 1])]

def add_family(model: str) -> str:
    m = str(model).strip()
    if m.lower().startswith("claude"):
        return "Claude"
    if m.lower().startswith("deepseek"):
        return "DeepSeek"
    if m.lower().startswith("gemini"):
        return "Gemini"
    if m.lower().startswith("kimi"):
        return "Kimi"
    if m.lower().startswith("llama"):
        return "Llama"
    if m.lower().startswith("gpt"):
        return "OpenAI"
    return "Other"

df["family"] = df["model"].apply(add_family)

# Unadjusted accuracy by model x year (mean across all runs and questions that year)
acc = (
    df.groupby(["family", "model", "neet_year"], as_index=False)["marks"]
      .mean()
      .rename(columns={"marks": "accuracy"})
)

# Facet order like the example
family_order = ["Claude", "DeepSeek", "OpenAI", "Gemini", "Kimi", "Llama"]
acc["family"] = pd.Categorical(acc["family"], categories=family_order, ordered=True)

# Plot
fig, axes = plt.subplots(2, 3, figsize=(14, 5), dpi=220, sharex=True, sharey=True)
axes = axes.flatten()

all_handles = []
all_labels = []

years = sorted(df["neet_year"].unique())

for idx, fam in enumerate(family_order):
    ax = axes[idx]
    sub = acc[acc["family"] == fam]
    ax.set_title(f"Family = {fam}", fontsize=9)
    ax.grid(True, axis="both", linestyle="--", alpha=0.4)
    
    # plot each model in this family
    for model_name, g in sub.groupby("model"):
        g = g.sort_values("neet_year")
        h = ax.plot(g["neet_year"], g["accuracy"], marker="o", linewidth=1)[0]
        if model_name not in all_labels:
            all_handles.append(h)
            all_labels.append(model_name)
    
    ax.set_xticks(years)
    ax.set_xlabel("NEET PG Year")
    if idx % 3 == 0:
        ax.set_ylabel("Accuracy")

# Global title + legend
fig.suptitle("Figure: Family-wise Evolution of Medical AI (2021–2025)", fontsize=12, y=1.02)
fig.legend(all_handles, all_labels, title="model", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

plt.tight_layout()

out_path = "family_wise_accuracy_generated.png"
plt.savefig(out_path, bbox_inches="tight")
plt.show()

out_path

# Miscellaneous Plots

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Re-prepare data just in case
accuracy_df = df.groupby(['model', 'neet_year', 'run'])['marks'].mean().reset_index()
accuracy_df.rename(columns={'marks': 'accuracy'}, inplace=True)
avg_accuracy = accuracy_df.groupby(['model', 'neet_year'])['accuracy'].mean().reset_index()
heatmap_data = avg_accuracy.pivot(index='model', columns='neet_year', values='accuracy')

# Figure 1: Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
plt.title('Figure 1: Model Accuracy Heatmap across NEET PG Years')
plt.ylabel('AI Model')
plt.xlabel('NEET PG Year')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.close()

# Figure 2: Trend Lines
plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_accuracy, x='neet_year', y='accuracy', hue='model', marker='o', palette='tab20')
plt.title('Figure 2: Temporal Performance Trends (2021-2025)')
plt.ylabel('Mean Accuracy')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('trends.png')
plt.close()

# Figure 3: Consistency vs Performance
# Fix: Ensure we are accessing the columns correctly
model_stats = accuracy_df.groupby('model')['accuracy'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 8))
sns.scatterplot(data=model_stats, x='mean', y='std', s=100, hue='model', palette='tab20', legend=False)

# Add labels
# model_stats has columns: 'model', 'mean', 'std'
for i in range(len(model_stats)):
    plt.text(model_stats.iloc[i]['mean']+0.005, model_stats.iloc[i]['std'], model_stats.iloc[i]['model'], fontsize=9)

plt.title('Figure 3: Reliability Map (Performance vs Consistency)')
plt.xlabel('Mean Accuracy (Higher is better)')
plt.ylabel('Standard Deviation across Runs (Lower is better)')
plt.grid(True)
plt.tight_layout()
plt.savefig('consistency.png')
plt.close()

# Figure 4: Difficulty Stratification
# Calculate question difficulty (1 = everyone got it right, 0 = everyone got it wrong)
question_difficulty = df.groupby('question_id')['marks'].mean().reset_index()
question_difficulty.rename(columns={'marks': 'q_difficulty'}, inplace=True)

# Merge back
df_with_diff = pd.merge(df, question_difficulty, on='question_id')

def classify_difficulty(x):
    if x <= 0.5: return 'Hard'
    elif x <= 0.8: return 'Medium'
    else: return 'Easy'

df_with_diff['difficulty_bin'] = df_with_diff['q_difficulty'].apply(classify_difficulty)

# Calculate model accuracy per bin
bin_performance = df_with_diff.groupby(['model', 'difficulty_bin'])['marks'].mean().reset_index()

# Sort models by performance on 'Hard' questions
hard_order_df = bin_performance[bin_performance['difficulty_bin']=='Hard'].sort_values('marks', ascending=False)
hard_order = hard_order_df['model'].tolist()

plt.figure(figsize=(14, 7))
sns.barplot(data=bin_performance, x='model', y='marks', hue='difficulty_bin', 
            order=hard_order, hue_order=['Easy', 'Medium', 'Hard'], palette='RdYlGn_r')
plt.title('Figure 4: Model Performance Stratified by Question Difficulty')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.legend(title='Question Difficulty')
plt.tight_layout()
plt.savefig('difficulty_strat.png')
plt.close()

print("Figures generated successfully.")

