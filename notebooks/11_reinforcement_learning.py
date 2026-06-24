# %% [markdown]
"""
# 🎰 Notebook 11: Reinforcement Learning (Multi-Armed Bandits)
## Healthcare Recommendation System — Phase 5

This notebook simulates a Multi-Armed Bandit (MAB) recommendation policy:
1. Defines drugs as arms, with rewards drawn from ratings normalized to [0, 1].
2. Implements three MAB algorithms: Epsilon-Greedy (eps=0.1), UCB (Upper Confidence Bound), and Thompson Sampling.
3. Simulates 2000 recommendation steps, tracking cumulative reward, regret, and rolling average rewards.
4. Plots performance metrics to `reports/` and demonstrates online recommendation adaptation.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('reports', exist_ok=True)

# Set styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# Load drug reviews
reviews_df = pd.read_csv('data/raw/drug_reviews.tsv', sep='\t')

# Pick the top 20 most frequent drugs to be our bandit arms
n_arms = 20
top_drugs = reviews_df['drugName'].value_counts().head(n_arms).index.tolist()

# Extract ratings for each drug to act as reward distributions
# Scale 1-10 to [0, 1] for reward range
drug_rewards = {}
for drug in top_drugs:
    ratings = reviews_df[reviews_df['drugName'] == drug]['rating'].values / 10.0
    drug_rewards[drug] = ratings

# Calculate expected reward of each arm
expected_rewards = {d: np.mean(r) for d, r in drug_rewards.items()}
best_drug = max(expected_rewards, key=expected_rewards.get)
optimal_reward = expected_rewards[best_drug]

print("\n🎰 MULTI-ARMED BANDIT ENVIRONMENT SETUP")
print("="*60)
print(f"Number of arms (drugs): {n_arms}")
print(f"Optimal drug: {best_drug} (Expected reward: {optimal_reward:.4f})")
print("Top 5 drugs expected rewards:")
for d in top_drugs[:5]:
    print(f" - {d}: {expected_rewards[d]:.4f}")

# %% [markdown]
"""
## 1. Implement MAB Strategies
"""

# %%
n_steps = 2000

# Helper function to sample reward for selected arm
def draw_reward(drug_name):
    # Select a random rating from the historical distributions
    return np.random.choice(drug_rewards[drug_name])

# %%
# Strategy 1: Epsilon-Greedy (eps=0.1)
np.random.seed(42)
eps = 0.1
Q_eg = np.zeros(n_arms)
N_eg = np.zeros(n_arms)
rewards_eg = []
regret_eg = []
chosen_arms_eg = []

for t in range(n_steps):
    if np.random.random() < eps:
        a = np.random.randint(n_arms)
    else:
        a = np.argmax(Q_eg)
        
    r = draw_reward(top_drugs[a])
    N_eg[a] += 1
    Q_eg[a] += (r - Q_eg[a]) / N_eg[a]
    
    rewards_eg.append(r)
    regret_eg.append(optimal_reward - expected_rewards[top_drugs[a]])
    chosen_arms_eg.append(a)

# %%
# Strategy 2: UCB (Upper Confidence Bound)
np.random.seed(42)
Q_ucb = np.zeros(n_arms)
N_ucb = np.zeros(n_arms)
rewards_ucb = []
regret_ucb = []
chosen_arms_ucb = []

for t in range(n_steps):
    if t < n_arms:
        # Initialize by pulling each arm once
        a = t
    else:
        # UCB index calculation
        ucb_values = Q_ucb + np.sqrt(2 * np.log(t) / (N_ucb + 1e-8))
        a = np.argmax(ucb_values)
        
    r = draw_reward(top_drugs[a])
    N_ucb[a] += 1
    Q_ucb[a] += (r - Q_ucb[a]) / N_ucb[a]
    
    rewards_ucb.append(r)
    regret_ucb.append(optimal_reward - expected_rewards[top_drugs[a]])
    chosen_arms_ucb.append(a)

# %%
# Strategy 3: Thompson Sampling
np.random.seed(42)
# Using Beta distribution priors (alpha=1, beta=1)
alpha_ts = np.ones(n_arms)
beta_ts = np.ones(n_arms)
rewards_ts = []
regret_ts = []
chosen_arms_ts = []

for t in range(n_steps):
    # Sample from Beta distributions
    samples = np.random.beta(alpha_ts, beta_ts)
    a = np.argmax(samples)
    
    r = draw_reward(top_drugs[a])
    # Update priors based on binary feedback (using threshold of 0.7 for positive outcome)
    if r >= 0.7:
        alpha_ts[a] += 1
    else:
        beta_ts[a] += 1
        
    rewards_ts.append(r)
    regret_ts.append(optimal_reward - expected_rewards[top_drugs[a]])
    chosen_arms_ts.append(a)

# %% [markdown]
"""
## 2. Plots & Visualizations
"""

# %%
# Plot A: Cumulative Rewards
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(rewards_eg), label='Epsilon-Greedy (ε=0.1)', color='crimson', linewidth=2)
plt.plot(np.cumsum(rewards_ucb), label='UCB1', color='dodgerblue', linewidth=2)
plt.plot(np.cumsum(rewards_ts), label='Thompson Sampling', color='forestgreen', linewidth=2)
plt.plot([optimal_reward * t for t in range(n_steps)], label='Optimal (Oracle)', color='gray', linestyle='--')
plt.title('Multi-Armed Bandit: Cumulative Rewards Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Simulated Step')
plt.ylabel('Cumulative Reward')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('reports/rl_cumulative_rewards.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot B: Cumulative Regret
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(regret_eg), label='Epsilon-Greedy (ε=0.1)', color='crimson', linewidth=2)
plt.plot(np.cumsum(regret_ucb), label='UCB1', color='dodgerblue', linewidth=2)
plt.plot(np.cumsum(regret_ts), label='Thompson Sampling', color='forestgreen', linewidth=2)
plt.title('Multi-Armed Bandit: Cumulative Regret Comparison (Lower is Better)', fontsize=14, fontweight='bold')
plt.xlabel('Simulated Step')
plt.ylabel('Cumulative Regret')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('reports/rl_cumulative_regret.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot C: Rolling Average Reward (per 100 steps)
window = 100
avg_reward_eg = pd.Series(rewards_eg).rolling(window=window).mean()
avg_reward_ucb = pd.Series(rewards_ucb).rolling(window=window).mean()
avg_reward_ts = pd.Series(rewards_ts).rolling(window=window).mean()

plt.figure(figsize=(10, 6))
plt.plot(avg_reward_eg, label='Epsilon-Greedy', color='crimson', alpha=0.8)
plt.plot(avg_reward_ucb, label='UCB1', color='dodgerblue', alpha=0.8)
plt.plot(avg_reward_ts, label='Thompson Sampling', color='forestgreen', alpha=0.8)
plt.axhline(optimal_reward, color='gray', linestyle='--', label='Max Expected Reward')
plt.title('Multi-Armed Bandit: 100-Step Rolling Average Reward', fontsize=14, fontweight='bold')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('reports/rl_rolling_average_reward.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved MAB simulation figures to reports/")

# %% [markdown]
"""
## 3. Results Summary & Evolution Demonstration
"""

# %%
# Summary Table
summary_df = pd.DataFrame({
    'Strategy': ['Epsilon-Greedy (ε=0.1)', 'UCB1', 'Thompson Sampling'],
    'Total Reward': [np.sum(rewards_eg), np.sum(rewards_ucb), np.sum(rewards_ts)],
    'Final Avg Reward': [np.mean(rewards_eg), np.mean(rewards_ucb), np.mean(rewards_ts)],
    'Total Regret': [np.sum(regret_eg), np.sum(regret_ucb), np.sum(regret_ts)]
})
print("\n📊 REINFORCEMENT LEARNING SIMULATION SUMMARY (2000 steps)")
print("="*75)
print(summary_df.to_string(index=False))

# Show how recommendations evolve over first 500 steps (Thompson Sampling)
print("\n🔮 ONLINE LEARNING RECOMMENDATION EVOLUTION (Thompson Sampling):")
print("="*75)
for cutoff in [10, 100, 500]:
    step_choices = chosen_arms_ts[:cutoff]
    counts = pd.Series(step_choices).value_counts()
    top_chosen_idx = counts.index[0]
    top_chosen_drug = top_drugs[top_chosen_idx]
    print(f"After first {cutoff:<3} steps: Top recommended drug is '{top_chosen_drug}' (selected {counts.iloc[0]} times)")

print("\n✅ Reinforcement learning phase completed successfully!")
