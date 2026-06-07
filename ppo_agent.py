"""
ppo_agent.py — Pure PPO baseline
==================================
Supports CartPole-v1 and CliffWalking-v1 (no world model).

    python ppo_agent.py                          
    python ppo_agent.py --env CliffWalking-v1


"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
import scipy.signal
from scipy.integrate import trapezoid


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

EVAL_N_EPISODES = 30
EVAL_SEED_OFFSET = 1000



ENV_PROFILES = {
    "CartPole-v1": {
        "steps_per_epoch":         4_000,
        "epochs":                  30,
        "max_steps_per_episode":   500,
        "hidden_sizes":            (64, 64),
        "policy_lr":               3e-4,
        "value_lr":                1e-3,
        "train_policy_iterations": 80,
        "train_value_iterations":  80,
        "gamma":                   0.99,
        "lam":                     0.97,
        "clip_ratio":              0.2,
        "target_kl":               0.01,
        "solve_threshold":         500,
        "shaped_reward":           False,
        "discrete_obs":            False,
        "n_states":                None,
    },

    "CliffWalking-v1": {

        "steps_per_epoch":         4_000,
        "epochs":                  80,
        "max_steps_per_episode":   200,

        "hidden_sizes":            (64, 64),

        "policy_lr":               3e-4,
        "value_lr":                1e-3,
        "train_policy_iterations": 80,
        "train_value_iterations":  80,

        "gamma":                   0.99,
        "lam":                     0.99,       
        "clip_ratio":              0.2,
        "target_kl":               0.01,

        "solve_threshold":         -20,          

        "shaped_reward":           False,        

        "discrete_obs":            True,         
        "n_states":                48,          
    },
}

SUPPORTED_ENVS = list(ENV_PROFILES.keys())



def preprocess_obs(obs, cfg: dict) -> np.ndarray:

    if cfg["discrete_obs"]:
        vec = np.zeros(cfg["n_states"], dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec
    return np.asarray(obs, dtype=np.float32)


def get_obs_dim(cfg: dict, env: gym.Env) -> int:

    if cfg["discrete_obs"]:
        return cfg["n_states"]
    return env.observation_space.shape[0]



def shape_reward(obs, next_obs, raw_reward: float,
                 env_name: str, gamma: float) -> float:

    return raw_reward   



def learning_curve_auc(rewards: np.ndarray) -> float:
    if len(rewards) == 0:
        return 0.0
    if len(rewards) == 1:
        return float(rewards[0])
    return float(trapezoid(rewards.astype(np.float64), np.arange(len(rewards))))


def compute_training_metrics(all_rewards: list, mean_rewards: list,
                             n_episodes: int, n_steps: int) -> dict:
    arr = np.array(all_rewards, dtype=np.float32)
    last_n = min(50, len(arr))
    return {
        "train_avg_reward": float(arr.mean()) if len(arr) else 0.0,
        "train_std_reward": float(arr.std()) if len(arr) else 0.0,
        "train_final_mean": float(arr[-last_n:].mean()) if last_n else 0.0,
        "train_final_std": float(arr[-last_n:].std()) if last_n > 1 else 0.0,
        "train_auc": learning_curve_auc(arr),
        "train_mean25_final": float(mean_rewards[-1]) if mean_rewards else 0.0,
        "n_episodes": int(n_episodes),
        "n_steps": int(n_steps),
    }


def print_metrics_summary(train_m: dict, eval_m: dict, log_print=print):
    log_print(f"\n{'='*72}")
    log_print("  TRAINING METRICS")
    log_print(f"{'='*72}")
    log_print(f"  Episodes        : {train_m['n_episodes']}")
    log_print(f"  Steps (total)   : {train_m['n_steps']}")
    log_print(f"  Avg reward      : {train_m['train_avg_reward']:.2f}")
    log_print(f"  Std reward      : {train_m['train_std_reward']:.2f}")
    log_print(f"  AUC (learning)  : {train_m['train_auc']:.1f}")
    log_print(f"  Final mean (50) : {train_m['train_final_mean']:.2f} ± {train_m['train_final_std']:.2f}")

    log_print(f"\n{'='*72}")
    log_print(f"  GREEDY EVAL ({eval_m['n_episodes']} episodes)")
    log_print(f"{'='*72}")
    log_print(f"  Mean reward     : {eval_m['eval_mean']:.2f} ± {eval_m['eval_std']:.2f}")
    log_print(f"  Min / Median / Max : {eval_m['eval_min']:.0f} / "
              f"{eval_m['eval_median']:.0f} / {eval_m['eval_max']:.0f}")
    log_print(f"  Success rate    : {100*eval_m['success_rate']:.1f}% "
              f"({eval_m['n_solved']}/{eval_m['n_episodes']}) "
              f"[≥ {eval_m['solve_threshold']:.0f}]")
    log_print(f"  Mean ep length  : {eval_m['eval_length_mean']:.1f} ± "
              f"{eval_m['eval_length_std']:.1f}")
    log_print(f"  Eval steps      : {eval_m['total_steps']}")



def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:

    def __init__(self, obs_dim: int, size: int, gamma: float, lam: float):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros(size,            dtype=np.int32)
        self.adv_buf  = np.zeros(size,            dtype=np.float32)
        self.rew_buf  = np.zeros(size,            dtype=np.float32)
        self.ret_buf  = np.zeros(size,            dtype=np.float32)
        self.val_buf  = np.zeros(size,            dtype=np.float32)
        self.logp_buf = np.zeros(size,            dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.traj_start = 0, 0

    def store(self, obs, action, reward, value, logp):
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = action
        self.rew_buf[self.ptr]  = reward
        self.val_buf[self.ptr]  = value
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_trajectory(self, last_value: float = 0.0):
        """Call at episode end or epoch boundary."""
        path    = slice(self.traj_start, self.ptr)
        rewards = np.append(self.rew_buf[path], last_value)
        values  = np.append(self.val_buf[path], last_value)
        deltas  = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[path] = discounted_cumulative_sums(deltas, self.gamma * self.lam)
        self.ret_buf[path] = discounted_cumulative_sums(rewards, self.gamma)[:-1]
        self.traj_start = self.ptr

    def get(self):
        assert self.ptr == len(self.obs_buf), \
            "Buffer not full — call finish_trajectory() before get()."
        self.ptr, self.traj_start = 0, 0
        adv_mean = self.adv_buf.mean()
        adv_std  = self.adv_buf.std() + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return (self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf)



def build_mlp(in_dim: int, hidden_sizes: tuple, out_dim: int) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_act: int, hidden: tuple):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, n_act)

    def forward(self, obs):
        return self.net(obs)

    def logprobabilities(self, obs, actions):
        logits    = self(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    @torch.no_grad()
    def sample_action(self, obs):
        logits    = self(obs)
        action    = torch.distributions.Categorical(logits=logits).sample()
        log_probs = torch.log_softmax(logits, dim=-1)
        logp      = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return logits, action, logp


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: tuple):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, 1)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


@torch.no_grad()
def evaluate_ppo_greedy(actor: Actor, env_name: str, cfg: dict,
                        n_episodes: int = EVAL_N_EPISODES,
                        seed_offset: int = EVAL_SEED_OFFSET) -> dict:

    env = gym.make(env_name)
    max_steps = cfg.get("max_steps_per_episode", 500)
    solve_threshold = cfg["solve_threshold"]

    was_training = actor.training
    actor.eval()

    rewards, lengths = [], []
    total_steps = 0

    for ep in range(n_episodes):
        raw_obs, _ = env.reset(seed=SEED + seed_offset + ep)
        obs = preprocess_obs(raw_obs, cfg)
        ep_reward, ep_len = 0.0, 0

        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = actor(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())

            raw_obs, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess_obs(raw_obs, cfg)
            ep_reward += reward
            ep_len += 1
            total_steps += 1
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        lengths.append(ep_len)

    env.close()
    if was_training:
        actor.train()

    rewards_arr = np.array(rewards, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.float32)

    return {
        "eval_rewards": [float(r) for r in rewards],
        "eval_lengths": [int(l) for l in lengths],
        "eval_mean": float(rewards_arr.mean()),
        "eval_std": float(rewards_arr.std()),
        "eval_min": float(rewards_arr.min()),
        "eval_max": float(rewards_arr.max()),
        "eval_median": float(np.median(rewards_arr)),
        "eval_length_mean": float(lengths_arr.mean()),
        "eval_length_std": float(lengths_arr.std()),
        "success_rate": float((rewards_arr >= solve_threshold).mean()),
        "n_solved": int((rewards_arr >= solve_threshold).sum()),
        "n_episodes": n_episodes,
        "total_steps": total_steps,
        "solve_threshold": float(solve_threshold),
    }



def train_policy(actor, policy_optimizer, obs_t, act_t,
                 logp_old_t, adv_t, clip_ratio, target_kl):
    policy_optimizer.zero_grad()
    logp_new = actor.logprobabilities(obs_t, act_t)
    ratio    = torch.exp(logp_new - logp_old_t)
    clip_adv = torch.where(adv_t > 0,
                           (1 + clip_ratio) * adv_t,
                           (1 - clip_ratio) * adv_t)
    policy_loss = -torch.mean(torch.minimum(ratio * adv_t, clip_adv))
    policy_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
    policy_optimizer.step()
    with torch.no_grad():
        kl = torch.mean(logp_old_t - actor.logprobabilities(obs_t, act_t))
    return kl.item()


def train_value_function(critic, value_optimizer, obs_t, ret_t):
    value_optimizer.zero_grad()
    loss = torch.mean((ret_t - critic(obs_t)) ** 2)
    loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    value_optimizer.step()
    return loss.item()



def train_ppo(env_name: str = "CartPole-v1"):
    assert env_name in SUPPORTED_ENVS, \
        f"Unsupported env '{env_name}'. Choose from: {SUPPORTED_ENVS}"

    cfg       = ENV_PROFILES[env_name]
    plots_dir = os.path.join(BASE_DIR, "PPO_plots", env_name)
    os.makedirs(plots_dir, exist_ok=True)

    log_file = open(os.path.join(plots_dir, "training.txt"), "w", encoding="utf-8")

    def log_print(*args, **kwargs):
        """Print to both console and log file."""
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()

    env     = gym.make(env_name)
    obs_dim = get_obs_dim(cfg, env)         
    n_act   = env.action_space.n

    actor  = Actor(obs_dim,  n_act, cfg["hidden_sizes"]).to(DEVICE)
    critic = Critic(obs_dim, cfg["hidden_sizes"]).to(DEVICE)
    policy_optimizer = optim.Adam(actor.parameters(),  lr=cfg["policy_lr"])
    value_optimizer  = optim.Adam(critic.parameters(), lr=cfg["value_lr"])

    buffer = Buffer(obs_dim, cfg["steps_per_epoch"], cfg["gamma"], cfg["lam"])

    log_print(f"\n{'='*60}")
    log_print(f"  PPO — {env_name}  (no world model)")
    log_print(f"  obs_dim={obs_dim} | n_actions={n_act} | device={DEVICE}")
    log_print(f"  hidden={cfg['hidden_sizes']} | "
              f"policy_lr={cfg['policy_lr']} | value_lr={cfg['value_lr']}")
    log_print(f"  steps/epoch={cfg['steps_per_epoch']} | epochs={cfg['epochs']}")
    log_print(f"  γ={cfg['gamma']} | λ={cfg['lam']} | "
              f"clip={cfg['clip_ratio']} | target_kl={cfg['target_kl']}")
    if cfg["discrete_obs"]:
        log_print(f"  obs encoding: one-hot ({cfg['n_states']} states)")
    log_print(f"{'='*60}\n")

    all_rewards, mean_rewards = [], []
    raw_obs, _ = env.reset(seed=SEED)
    observation = preprocess_obs(raw_obs, cfg) 
    ep_return, ep_length = 0.0, 0
    num_episodes_total = 0
    total_steps = 0

    for epoch in range(cfg["epochs"]):
        sum_return   = 0.0
        sum_length   = 0
        num_episodes = 0

        for t in range(cfg["steps_per_epoch"]):
            obs_t = torch.tensor(observation, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                _, action_t, logp_t = actor.sample_action(obs_t)
                value_t             = critic(obs_t)

            action = int(action_t.item())
            logp   = logp_t.item()
            value  = value_t.item()

            raw_obs_new, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Preprocess the new obs immediately — obs stored in buffer is
            # always the float32 vector, never the raw integer.
            observation_new = preprocess_obs(raw_obs_new, cfg)

            stored_reward = shape_reward(
                observation, observation_new, reward, env_name, cfg["gamma"])

            ep_return += reward         
            ep_length += 1
            total_steps += 1

            buffer.store(observation, action, stored_reward, value, logp)
            observation = observation_new

            terminal = done or (t == cfg["steps_per_epoch"] - 1)
            if terminal:
                if done:
                    last_value = 0.0
                else:
                    obs_boot = torch.tensor(observation, dtype=torch.float32,
                                            device=DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        last_value = critic(obs_boot).item()

                buffer.finish_trajectory(last_value)
                sum_return   += ep_return
                sum_length   += ep_length
                num_episodes += 1
                num_episodes_total += 1

                all_rewards.append(ep_return)
                mean_rewards.append(np.mean(all_rewards[-25:]))

                raw_obs, _ = env.reset(seed=SEED + num_episodes_total)
                observation = preprocess_obs(raw_obs, cfg)
                ep_return, ep_length = 0.0, 0


        obs_b, act_b, adv_b, ret_b, logp_b = buffer.get()

        obs_t  = torch.tensor(obs_b,  dtype=torch.float32, device=DEVICE)
        act_t  = torch.tensor(act_b,  dtype=torch.int64,   device=DEVICE)
        adv_t  = torch.tensor(adv_b,  dtype=torch.float32, device=DEVICE)
        ret_t  = torch.tensor(ret_b,  dtype=torch.float32, device=DEVICE)
        logp_t = torch.tensor(logp_b, dtype=torch.float32, device=DEVICE)

        for pi_iter in range(cfg["train_policy_iterations"]):
            kl = train_policy(actor, policy_optimizer, obs_t, act_t,
                              logp_t, adv_t,
                              cfg["clip_ratio"], cfg["target_kl"])
            if kl > 1.5 * cfg["target_kl"]:
                log_print(f"   [KL early stop] epoch={epoch+1} "
                          f"iter={pi_iter+1} kl={kl:.5f}")
                break

        for _ in range(cfg["train_value_iterations"]):
            train_value_function(critic, value_optimizer, obs_t, ret_t)

        mean_ret = sum_return / max(num_episodes, 1)
        mean_len = sum_length / max(num_episodes, 1)
        log_print(f"Epoch {epoch+1:03d}/{cfg['epochs']}  "
                  f"MeanReturn={mean_ret:8.2f}  "
                  f"MeanLen={mean_len:6.1f}  "
                  f"Episodes={num_episodes:4d}  "
                  f"TotalEps={num_episodes_total}")

    env.close()


    train_metrics = compute_training_metrics(
        all_rewards, mean_rewards, num_episodes_total, total_steps
    )
    log_print(f"\n  [Eval] Greedy evaluation ({EVAL_N_EPISODES} episodes)...")
    eval_metrics = evaluate_ppo_greedy(actor, env_name, cfg)
    print_metrics_summary(train_metrics, eval_metrics, log_print=log_print)

    metrics = {
        "env_name": env_name,
        "train": train_metrics,
        "eval": eval_metrics,
        "all_episode_rewards": [float(r) for r in all_rewards],
    }
    metrics_path = os.path.join(plots_dir, "ppo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log_print(f"[Metrics] Saved → {metrics_path}")


    threshold = cfg["solve_threshold"]
    train_avg = train_metrics["train_avg_reward"]
    eval_mean = eval_metrics["eval_mean"]
    eval_std  = eval_metrics["eval_std"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"PPO — {env_name} (no world model)", fontsize=13)

    episodes_x = np.arange(1, len(all_rewards) + 1)

    ax = axes[0, 0]
    ax.plot(all_rewards, alpha=0.30, linewidth=0.8, label="Episode Return")
    ax.plot(mean_rewards, linewidth=2.0, label="Mean-25")
    ax.axhline(train_avg, color='#6340A0', linestyle='-', alpha=0.7, linewidth=1.5,
               label=f"Avg reward ({train_avg:.1f})")
    ax.axhline(eval_mean, color='#1D9E75', linestyle='--', alpha=0.85, linewidth=1.5,
               label=f"Greedy eval ({eval_mean:.1f}±{eval_std:.1f})")
    ax.axhline(threshold, color='r', linestyle=':', alpha=0.5,
               label=f"Solved (≥{threshold:.0f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (raw)")
    ax.set_title("Learning Curve + Avg / Greedy Eval")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    window = 25
    if len(all_rewards) >= window:
        smooth = np.convolve(all_rewards, np.ones(window) / window, mode='valid')
        ax.plot(episodes_x[window - 1:], smooth, linewidth=2.0, color='#1D9E75',
                label=f"Smoothed (w={window})")
    ax.axhline(train_avg, color='#6340A0', linestyle='-', alpha=0.7,
               label=f"Avg ({train_avg:.1f})")
    ax.axhline(eval_mean, color='#0F6E56', linestyle='--', alpha=0.85,
               label=f"Greedy eval ({eval_mean:.1f})")
    ax.axhline(threshold, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"Smoothed (w={window})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.bar(["Train\n(avg)", "Train\n(final 50)", f"Greedy\n({EVAL_N_EPISODES} eps)"],
           [train_metrics["train_avg_reward"],
            train_metrics["train_final_mean"],
            eval_metrics["eval_mean"]],
           yerr=[0, train_metrics["train_final_std"], eval_metrics["eval_std"]],
           capsize=5, color=['#6340A0', '#878787', '#1D9E75'])
    ax.axhline(threshold, color='r', linestyle='--', alpha=0.45, label=f"Solved ({threshold})")
    ax.set_ylabel("Reward")
    ax.set_title("Avg Reward Comparison")
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"Train metrics\n"
        f"  Episodes     : {train_metrics['n_episodes']}\n"
        f"  Steps        : {train_metrics['n_steps']}\n"
        f"  Avg reward   : {train_metrics['train_avg_reward']:.2f}\n"
        f"  Std reward   : {train_metrics['train_std_reward']:.2f}\n"
        f"  AUC          : {train_metrics['train_auc']:.1f}\n"
        f"  Final (50)   : {train_metrics['train_final_mean']:.2f} "
        f"± {train_metrics['train_final_std']:.2f}\n\n"
        f"Greedy eval ({EVAL_N_EPISODES} eps)\n"
        f"  Mean         : {eval_metrics['eval_mean']:.2f} ± {eval_metrics['eval_std']:.2f}\n"
        f"  Min / Med / Max : {eval_metrics['eval_min']:.0f} / "
        f"{eval_metrics['eval_median']:.0f} / {eval_metrics['eval_max']:.0f}\n"
        f"  Success rate : {100*eval_metrics['success_rate']:.1f}% "
        f"({eval_metrics['n_solved']}/{eval_metrics['n_episodes']})\n"
        f"  Ep length    : {eval_metrics['eval_length_mean']:.1f} "
        f"± {eval_metrics['eval_length_std']:.1f}\n"
        f"  Eval steps   : {eval_metrics['total_steps']}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))

    shaped_str = "ON" if cfg["shaped_reward"] else "OFF"
    fig.text(0.5, 0.01,
             f"hidden={cfg['hidden_sizes']}  lr_p={cfg['policy_lr']}  lr_v={cfg['value_lr']}  "
             f"γ={cfg['gamma']}  λ={cfg['lam']}  steps/ep={cfg['steps_per_epoch']}  "
             f"epochs={cfg['epochs']}  shaping={shaped_str}",
             ha='center', fontsize=7.5, color='#555555')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_path = os.path.join(plots_dir, "ppo_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    log_print(f"[Plot] Saved → {fig_path}")


    pth_path = os.path.join(plots_dir, "ppo_agent.pth")
    torch.save({
        "actor":        actor.state_dict(),
        "critic":       critic.state_dict(),
        "env_name":     env_name,
        "obs_dim":      obs_dim,
        "n_actions":    n_act,
        "hidden_sizes": cfg["hidden_sizes"],
        "discrete_obs": cfg["discrete_obs"],
        "n_states":     cfg["n_states"],
        "train_metrics": train_metrics,
        "eval_metrics":  eval_metrics,
        "eval_mean":     eval_metrics["eval_mean"],
        "success_rate":  eval_metrics["success_rate"],
    }, pth_path)
    log_print(f"[Model] Saved → {pth_path}")

    log_file.close()

    return actor, critic, all_rewards, mean_rewards, train_metrics, eval_metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO baseline — multi-env")
    parser.add_argument(
        "--env",
        default="CartPole-v1",
        choices=SUPPORTED_ENVS,
        help=f"Environment to train on. Choices: {SUPPORTED_ENVS}",
    )
    args = parser.parse_args()
    train_ppo(env_name=args.env)
