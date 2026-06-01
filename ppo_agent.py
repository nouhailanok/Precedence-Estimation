# # """
# # ppo_agent.py — Pure PPO baseline for CartPole-v1
# # ==================================================
# # Vanilla PPO with NO world model and NO synthetic transitions.

# # Output
# # ------
# #     PPO_plots/
# #         ppo_curve.png
# #         ppo_agent.pth
# # """

# # import os
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import torch.nn.functional as F
# # from torch.distributions import Categorical
# # import matplotlib.pyplot as plt
# # import gymnasium as gym

# # # ─────────────────────────────────────────────
# # # CONFIG  (keep identical to dyna_ppo_agent.py)
# # # ─────────────────────────────────────────────
# # BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# # ENV_NAME   = "CartPole-v1"
# # PLOTS_DIR  = os.path.join(BASE_DIR, "PPO_plots")

# # N_EPISODES   = 600
# # MAX_STEPS    = 500
# # ROLLOUT_LEN  = 2048     # Standard PPO rollout window for stable GAE estimation
# # PPO_EPOCHS   = 10
# # MINI_BATCH   = 64
# # GAMMA        = 0.99
# # GAE_LAMBDA   = 0.95
# # CLIP_EPS     = 0.2
# # ENT_COEF     = 0.01
# # VF_COEF      = 0.5
# # MAX_GRAD_NORM= 0.5
# # LR           = 3e-4


# # SEED         = 42
# # N_ACTIONS    = 2
# # STATE_DIM    = 4
# # DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # os.makedirs(PLOTS_DIR, exist_ok=True)
# # np.random.seed(SEED)
# # random.seed(SEED)
# # torch.manual_seed(SEED)


# # # ─────────────────────────────────────────────
# # # ACTOR-CRITIC
# # # ─────────────────────────────────────────────
# # class ActorCritic(nn.Module):
# #     def __init__(self, obs_dim=STATE_DIM, n_act=N_ACTIONS, hidden=128): # Bumped hidden to 128
# #         super().__init__()
# #         self.shared = nn.Sequential(
# #             nn.Linear(obs_dim, hidden), nn.Tanh(),
# #             nn.Linear(hidden, hidden),  nn.Tanh(),
# #         )
# #         self.actor  = nn.Linear(hidden, n_act)
# #         self.critic = nn.Linear(hidden, 1)

# #     def forward(self, x):
# #         h = self.shared(x)
# #         return self.actor(h), self.critic(h).squeeze(-1)

# #     def act(self, obs):
# #         logits, value = self(obs)
# #         dist   = Categorical(logits=logits)
# #         action = dist.sample()
# #         return action, dist.log_prob(action), dist.entropy(), value

# #     def evaluate(self, obs, actions):
# #         logits, value = self(obs)
# #         dist      = Categorical(logits=logits)
# #         log_probs = dist.log_prob(actions)
# #         entropy   = dist.entropy()
# #         return log_probs, entropy, value


# # # ─────────────────────────────────────────────
# # # ROLLOUT BUFFER
# # # ─────────────────────────────────────────────
# # class RolloutBuffer:
# #     def __init__(self):
# #         self.obs       = []
# #         self.actions   = []
# #         self.log_probs = []
# #         self.rewards   = []
# #         self.values    = []
# #         self.dones     = []

# #     def push(self, obs, action, log_prob, reward, value, done):
# #         self.obs.append(obs)
# #         self.actions.append(action)
# #         self.log_probs.append(log_prob)
# #         self.rewards.append(reward)
# #         self.values.append(value)
# #         self.dones.append(done)

# #     def clear(self):
# #         self.__init__()

# #     def __len__(self):
# #         return len(self.rewards)

# #     def compute_returns_advantages(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
# #         rewards    = np.array(self.rewards, dtype=np.float32)
# #         values     = np.array(self.values,  dtype=np.float32)
# #         dones      = np.array(self.dones,   dtype=np.float32)
# #         n          = len(rewards)
# #         advantages = np.zeros(n, dtype=np.float32)
# #         last_gae   = 0.0
# #         for t in reversed(range(n)):
# #             next_val  = last_value if t == n - 1 else values[t + 1]
# #             next_done = dones[t]
# #             delta     = rewards[t] + gamma * next_val * (1 - next_done) - values[t]
# #             last_gae  = delta + gamma * lam * (1 - next_done) * last_gae
# #             advantages[t] = last_gae
# #         returns = advantages + values
# #         return returns, advantages


# # # ─────────────────────────────────────────────
# # # PPO UPDATE
# # # ─────────────────────────────────────────────
# # def ppo_update(ac, optimizer, rollout, last_value):
# #     returns, advantages = rollout.compute_returns_advantages(last_value)

# #     obs_t  = torch.tensor(np.stack(rollout.obs),       dtype=torch.float32, device=DEVICE)
# #     act_t  = torch.tensor(np.array(rollout.actions),   dtype=torch.int64,   device=DEVICE)
# #     lp_old = torch.tensor(np.array(rollout.log_probs), dtype=torch.float32, device=DEVICE)
# #     ret_t  = torch.tensor(returns,                     dtype=torch.float32, device=DEVICE)
# #     adv_t  = torch.tensor(advantages,                  dtype=torch.float32, device=DEVICE)
# #     adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

# #     n   = len(rollout)
# #     idx = np.arange(n)

# #     for _ in range(PPO_EPOCHS):
# #         np.random.shuffle(idx)
# #         for start in range(0, n, MINI_BATCH):
# #             mb = idx[start: start + MINI_BATCH]
# #             log_probs, entropy, values = ac.evaluate(obs_t[mb], act_t[mb])

# #             ratio       = torch.exp(log_probs - lp_old[mb])
# #             surr1       = ratio * adv_t[mb]
# #             surr2       = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[mb]
# #             actor_loss  = -torch.min(surr1, surr2).mean()
# #             critic_loss = F.mse_loss(values, ret_t[mb])
# #             entropy_loss= -entropy.mean()

# #             loss = actor_loss + VF_COEF * critic_loss + ENT_COEF * entropy_loss

# #             optimizer.zero_grad()
# #             loss.backward()
# #             nn.utils.clip_grad_norm_(ac.parameters(), MAX_GRAD_NORM)
# #             optimizer.step()


# # # ─────────────────────────────────────────────
# # # TRAINING LOOP
# # # ─────────────────────────────────────────────
# # def train_ppo():
# #     env       = gym.make(ENV_NAME)
# #     ac        = ActorCritic().to(DEVICE)
# #     optimizer = optim.Adam(ac.parameters(), lr=LR, eps=1e-5)
# #     rollout   = RolloutBuffer()

# #     print(f"[PPO] Pure baseline")
# #     print(f"      ROLLOUT_LEN={ROLLOUT_LEN} | PPO_EPOCHS={PPO_EPOCHS} | CLIP_EPS={CLIP_EPS}")

# #     all_rewards, mean_rewards = [], []
# #     ep_reward   = 0.0
# #     ep_count    = 0
# #     steps_total = 0

# #     obs, _ = env.reset(seed=SEED)
    
# #     # Track the active live environment state across rollout blocks
# #     env_is_done = False 

# #     while ep_count < N_EPISODES:
# #         rollout.clear()

# #         # ── Collect ROLLOUT_LEN real steps ────────────────────────────────
# #         for _ in range(ROLLOUT_LEN):
# #             obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
# #             with torch.no_grad():
# #                 action, log_prob, _, value = ac.act(obs_t)
# #             a  = int(action.item())
# #             lp = log_prob.item()
# #             v  = value.item()

# #             next_obs, reward, terminated, truncated, _ = env.step(a)
# #             env_is_done = terminated or truncated

# #             rollout.push(obs, a, lp, float(reward), v, float(env_is_done))

# #             ep_reward   += reward
# #             steps_total += 1
# #             obs          = next_obs

# #             if env_is_done:
# #                 all_rewards.append(ep_reward)
# #                 mean_rewards.append(np.mean(all_rewards[-25:]))
# #                 ep_count  += 1
# #                 ep_reward  = 0.0
# #                 obs, _     = env.reset(seed=SEED + ep_count)
                
# #                 # Keep tracking until rollout buffer fills up completely
# #                 if ep_count >= N_EPISODES:
# #                     break

# #         # ── Bootstrap last value using the correct historical flag ────────
# #         with torch.no_grad():
# #             obs_t     = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
# #             _, last_v = ac(obs_t)
# #             # If the final step hit a terminal wall, bootstrap value is strictly 0.0
# #             last_value= 0.0 if env_is_done else last_v.item()

# #         # ── PPO update ────────────────────────────────────────────────────
# #         ppo_update(ac, optimizer, rollout, last_value)

# #         if ep_count > 0 and ep_count % 10 == 0:
# #             # Avoid list slicing errors if training finishes early
# #             current_mean = mean_rewards[-1] if len(mean_rewards) > 0 else 0.0
# #             print(f"Ep {ep_count:03d}  "
# #                   f"MeanR25={current_mean:.2f}  "
# #                   f"Steps={steps_total}")

# #     env.close()

# #     # ── Plot ──────────────────────────────────────────────────────────────
# #     plt.figure(figsize=(10, 4))
# #     plt.plot(all_rewards,  alpha=0.4, label="Episode Return")
# #     plt.plot(mean_rewards, linewidth=2, label="Mean-25")
# #     plt.grid(alpha=0.3)
# #     plt.xlabel("Episode")
# #     plt.ylabel("Reward")
# #     plt.title("Pure PPO — CartPole-v1 (no world model)")
# #     plt.legend()
# #     plt.tight_layout()
# #     save_fig = os.path.join(PLOTS_DIR, "ppo_curve.png")
# #     plt.savefig(save_fig, dpi=150)
# #     plt.show()
# #     print(f"[Plot] Saved → {save_fig}")

# #     # ── Save weights ──────────────────────────────────────────────────────
# #     save_pth = os.path.join(PLOTS_DIR, "ppo_agent.pth")
# #     torch.save(ac.state_dict(), save_pth)
# #     print(f"[Model] Saved → {save_pth}")

# #     # ── Greedy evaluation ─────────────────────────────────────────────────
# #     eval_env  = gym.make(ENV_NAME)
# #     obs_e, _  = eval_env.reset(seed=SEED + 111)
# #     total_r   = 0.0
# #     done_e    = False
# #     ac.eval()
# #     while not done_e:
# #         with torch.no_grad():
# #             logits, _ = ac(torch.tensor(obs_e, dtype=torch.float32, device=DEVICE).unsqueeze(0))
# #             a_e       = int(torch.argmax(logits).item())
# #         obs_e, r_e, term_e, trunc_e, _ = eval_env.step(a_e)
# #         done_e  = term_e or trunc_e
# #         total_r += r_e
# #     print(f"[Eval] Greedy episode reward = {total_r}")
# #     eval_env.close()


# # # ─────────────────────────────────────────────
# # # ENTRY POINT
# # # ─────────────────────────────────────────────
# # if __name__ == "__main__":
# #     train_ppo()


# """
# ppo_agent.py — Pure PPO baseline for CartPole-v1
#   https://keras.io/examples/rl/ppo_cartpole/

# Key design decisions:
#   - Separate actor and critic networks (not shared backbone)
#   - GAE-λ via scipy.signal.lfilter (discounted cumulative sums)
#   - Separate optimizers for actor and critic (different LRs)
#   - KL early stopping on the policy update
#   - steps_per_epoch rollout structure (not episode-count based)
#   - train_policy_iterations + train_value_iterations inner loops

# Output
#     PPO_plots/
#         ppo_curve.png
#         ppo_agent.pth
# """

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import gymnasium as gym
# import scipy.signal

# BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
# PLOTS_DIR = os.path.join(BASE_DIR, "PPO_plots")
# ENV_NAME  = "CartPole-v1"


# STEPS_PER_EPOCH         = 4000
# EPOCHS                  = 30
# GAMMA                   = 0.99
# CLIP_RATIO              = 0.2
# POLICY_LR               = 3e-4
# VALUE_LR                = 1e-3
# TRAIN_POLICY_ITERATIONS = 80
# TRAIN_VALUE_ITERATIONS  = 80
# LAM                     = 0.97
# TARGET_KL               = 0.01
# HIDDEN_SIZES            = (64, 64)

# SEED   = 42
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs(PLOTS_DIR, exist_ok=True)
# np.random.seed(SEED)
# torch.manual_seed(SEED)


# # ─────────────────────────────────────────────
# # DISCOUNTED CUMULATIVE SUMS
# # ─────────────────────────────────────────────
# def discounted_cumulative_sums(x, discount):
#     """scipy.signal.lfilter trick"""
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# # ─────────────────────────────────────────────
# # BUFFER 
# # ─────────────────────────────────────────────
# class Buffer:
#     """
#     Stores one epoch of trajectories.
#     finish_trajectory() computes GAE-λ advantages + rewards-to-go.
#     get() returns all data and resets the pointer.
#     """
#     def __init__(self, obs_dim, size, gamma=GAMMA, lam=LAM):
#         self.obs_buf    = np.zeros((size, obs_dim), dtype=np.float32)
#         self.act_buf    = np.zeros(size,            dtype=np.int32)
#         self.adv_buf    = np.zeros(size,            dtype=np.float32)
#         self.rew_buf    = np.zeros(size,            dtype=np.float32)
#         self.ret_buf    = np.zeros(size,            dtype=np.float32)
#         self.val_buf    = np.zeros(size,            dtype=np.float32)
#         self.logp_buf   = np.zeros(size,            dtype=np.float32)
#         self.gamma, self.lam = gamma, lam
#         self.ptr, self.traj_start = 0, 0

#     def store(self, obs, action, reward, value, logp):
#         self.obs_buf[self.ptr]  = obs
#         self.act_buf[self.ptr]  = action
#         self.rew_buf[self.ptr]  = reward
#         self.val_buf[self.ptr]  = value
#         self.logp_buf[self.ptr] = logp
#         self.ptr += 1

#     def finish_trajectory(self, last_value=0.0):
#         """
#         Call at episode end or epoch end.
#         Computes GAE-λ advantages and discounted rewards-to-go.
#         """
#         path = slice(self.traj_start, self.ptr)
#         rewards = np.append(self.rew_buf[path], last_value)
#         values  = np.append(self.val_buf[path], last_value)

#         # TD residuals → GAE
#         deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
#         self.adv_buf[path] = discounted_cumulative_sums(deltas, self.gamma * self.lam)

#         # Rewards-to-go (targets for the value function)
#         self.ret_buf[path] = discounted_cumulative_sums(rewards, self.gamma)[:-1]

#         self.traj_start = self.ptr

#     def get(self):
#         """Return all data, normalize advantages, reset pointer."""
#         assert self.ptr == len(self.obs_buf), "Buffer not full — call finish_trajectory first."
#         self.ptr, self.traj_start = 0, 0
#         # Normalize advantages (zero mean, unit std)
#         adv_mean = self.adv_buf.mean()
#         adv_std  = self.adv_buf.std() + 1e-8
#         self.adv_buf = (self.adv_buf - adv_mean) / adv_std
#         return (
#             self.obs_buf,
#             self.act_buf,
#             self.adv_buf,
#             self.ret_buf,
#             self.logp_buf,
#         )


# # ─────────────────────────────────────────────
# # NETWORKS 
# # ─────────────────────────────────────────────
# def build_mlp(in_dim, hidden_sizes, out_dim, output_activation=None):
#     """Build a Tanh MLP """
#     layers = []
#     prev = in_dim
#     for h in hidden_sizes:
#         layers += [nn.Linear(prev, h), nn.Tanh()]
#         prev = h
#     layers.append(nn.Linear(prev, out_dim))
#     if output_activation is not None:
#         layers.append(output_activation)
#     return nn.Sequential(*layers)


# class Actor(nn.Module):
#     """Outputs raw logits"""
#     def __init__(self, obs_dim, n_act, hidden=(64, 64)):
#         super().__init__()
#         self.net = build_mlp(obs_dim, hidden, n_act)

#     def forward(self, obs):
#         return self.net(obs)   # logits

#     def logprobabilities(self, obs, actions):
#         """log π(a|s) for a batch of (obs, action) pairs."""
#         logits    = self(obs)
#         log_probs = torch.log_softmax(logits, dim=-1)
#         # Gather log-prob of the taken action
#         return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

#     @torch.no_grad()
#     def sample_action(self, obs):
#         """Sample action and return (logits, action, log_prob)."""
#         logits = self(obs)
#         action = torch.distributions.Categorical(logits=logits).sample()
#         log_probs = torch.log_softmax(logits, dim=-1)
#         logp = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
#         return logits, action, logp


# class Critic(nn.Module):
#     """Outputs scalar V(s)"""
#     def __init__(self, obs_dim, hidden=(64, 64)):
#         super().__init__()
#         self.net = build_mlp(obs_dim, hidden, 1)

#     def forward(self, obs):
#         return self.net(obs).squeeze(-1)


# # ─────────────────────────────────────────────
# # PPO UPDATE FUNCTIONS
# # ─────────────────────────────────────────────
# def train_policy(actor, policy_optimizer, obs_t, act_t, logp_old_t, adv_t):
#     """
#     One gradient step on the clipped PPO objective.
#     Returns KL divergence for early stopping
#     """
#     policy_optimizer.zero_grad()
#     logp_new = actor.logprobabilities(obs_t, act_t)
#     ratio    = torch.exp(logp_new - logp_old_t)

#     # Clipped surrogate loss
#     clip_adv = torch.where(
#         adv_t > 0,
#         (1 + CLIP_RATIO) * adv_t,
#         (1 - CLIP_RATIO) * adv_t,
#     )
#     policy_loss = -torch.mean(torch.minimum(ratio * adv_t, clip_adv))
#     policy_loss.backward()
#     policy_optimizer.step()

#     # KL estimate for early stopping
#     with torch.no_grad():
#         kl = torch.mean(logp_old_t - actor.logprobabilities(obs_t, act_t))
#     return kl.item()


# def train_value_function(critic, value_optimizer, obs_t, ret_t):
#     """One gradient step on MSE value loss """
#     value_optimizer.zero_grad()
#     value_loss = torch.mean((ret_t - critic(obs_t)) ** 2)
#     value_loss.backward()
#     value_optimizer.step()
#     return value_loss.item()


# # ─────────────────────────────────────────────
# # TRAINING LOOP
# # ─────────────────────────────────────────────
# def train_ppo():
#     env     = gym.make(ENV_NAME)
#     obs_dim = env.observation_space.shape[0]
#     n_act   = env.action_space.n

#     actor  = Actor(obs_dim,  n_act, HIDDEN_SIZES).to(DEVICE)
#     critic = Critic(obs_dim, HIDDEN_SIZES).to(DEVICE)
#     policy_optimizer = optim.Adam(actor.parameters(),  lr=POLICY_LR)
#     value_optimizer  = optim.Adam(critic.parameters(), lr=VALUE_LR)

#     buffer = Buffer(obs_dim, STEPS_PER_EPOCH)

#     print(f"[PPO] Pure baseline — no world model")
#     print(f"      STEPS_PER_EPOCH={STEPS_PER_EPOCH} | EPOCHS={EPOCHS}")
#     print(f"      CLIP_RATIO={CLIP_RATIO} | TARGET_KL={TARGET_KL} | LAM={LAM}")

#     # Per-episode tracking for the learning curve
#     all_rewards, mean_rewards = [], []

#     observation, _ = env.reset(seed=SEED)
#     ep_return, ep_length = 0.0, 0

#     # ── Epoch loop ──────────────────────────────
#     for epoch in range(EPOCHS):
#         sum_return   = 0.0
#         sum_length   = 0
#         num_episodes = 0

#         # ── Step loop ────────────────────────────────────────────────────────
#         for t in range(STEPS_PER_EPOCH):
#             obs_t = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

#             # Sample action from actor
#             with torch.no_grad():
#                 _, action_t, logp_t = actor.sample_action(obs_t)
#                 value_t = critic(obs_t)

#             action = int(action_t.item())
#             logp   = logp_t.item()
#             value  = value_t.item()

#             observation_new, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             ep_return += reward
#             ep_length += 1

#             buffer.store(observation, action, reward, value, logp)
#             observation = observation_new

#             # ── End of trajectory ─────────────────────────────────────────
#             terminal = done or (t == STEPS_PER_EPOCH - 1)
#             if terminal:
#                 if done:
#                     last_value = 0.0
#                 else:
#                     # Bootstrap from critic (episode cut off by epoch boundary)
#                     obs_t_  = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#                     with torch.no_grad():
#                         last_value = critic(obs_t_).item()

#                 buffer.finish_trajectory(last_value)
#                 sum_return   += ep_return
#                 sum_length   += ep_length
#                 num_episodes += 1

#                 # Track per-episode reward (for learning curve)
#                 all_rewards.append(ep_return)
#                 mean_rewards.append(np.mean(all_rewards[-25:]))

#                 observation, _ = env.reset(seed=SEED + num_episodes + epoch * 1000)
#                 ep_return, ep_length = 0.0, 0

#         # ── PPO update ───────────────────────────────────────────────────────
#         obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buffer.get()

#         obs_t  = torch.tensor(obs_buf,  dtype=torch.float32, device=DEVICE)
#         act_t  = torch.tensor(act_buf,  dtype=torch.int64,   device=DEVICE)
#         adv_t  = torch.tensor(adv_buf,  dtype=torch.float32, device=DEVICE)
#         ret_t  = torch.tensor(ret_buf,  dtype=torch.float32, device=DEVICE)
#         logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=DEVICE)

#         # Policy update with KL early stopping
#         for i in range(TRAIN_POLICY_ITERATIONS):
#             kl = train_policy(actor, policy_optimizer, obs_t, act_t, logp_t, adv_t)
#             if kl > 1.5 * TARGET_KL:
#                 print(f"   [KL early stop] epoch={epoch+1} iter={i+1} kl={kl:.5f}")
#                 break

#         # Value function update
#         for _ in range(TRAIN_VALUE_ITERATIONS):
#             train_value_function(critic, value_optimizer, obs_t, ret_t)

#         mean_ret = sum_return / max(num_episodes, 1)
#         mean_len = sum_length / max(num_episodes, 1)
#         print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
#               f"MeanReturn={mean_ret:.2f}  "
#               f"MeanLength={mean_len:.2f}  "
#               f"Episodes={num_episodes}")

#     env.close()

#     # ── Learning curve plot ──────────────────────────────────────────────────
#     plt.figure(figsize=(10, 4))
#     plt.plot(all_rewards,  alpha=0.35, label="Episode Return")
#     plt.plot(mean_rewards, linewidth=2,  label="Mean-25")
#     plt.grid(alpha=0.3)
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.title("Pure PPO — CartPole-v1 (no world model)")
#     plt.legend()
#     plt.tight_layout()
#     fig_path = os.path.join(PLOTS_DIR, "ppo_curve.png")
#     plt.savefig(fig_path, dpi=150)
#     plt.show()
#     print(f"[Plot] Saved → {fig_path}")

#     # ── Save weights ─────────────────────────────────────────────────────────
#     pth_path = os.path.join(PLOTS_DIR, "ppo_agent.pth")
#     torch.save({
#         "actor":  actor.state_dict(),
#         "critic": critic.state_dict(),
#     }, pth_path)
#     print(f"[Model] Saved → {pth_path}")

#     # ── Greedy evaluation ───────────────
#     eval_env  = gym.make(ENV_NAME)
#     obs_e, _  = eval_env.reset(seed=SEED + 111)
#     total_r   = 0.0
#     done_e    = False
#     actor.eval()
#     while not done_e:
#         with torch.no_grad():
#             logits = actor(torch.tensor(obs_e, dtype=torch.float32, device=DEVICE).unsqueeze(0))
#             a_e    = int(torch.argmax(logits).item())
#         obs_e, r_e, term_e, trunc_e, _ = eval_env.step(a_e)
#         done_e  = term_e or trunc_e
#         total_r += r_e
#     print(f"[Eval] Greedy episode reward = {total_r}")
#     eval_env.close()


# # ─────────────────────────────────────────────
# # ENTRY POINT
# # ─────────────────────────────────────────────
# if __name__ == "__main__":
#     train_ppo()

"""
ppo_agent.py — Pure PPO baseline
==================================
Supports CartPole-v1, LunarLander-v3, and MountainCar-v0.
No world model, no synthetic transitions.

Usage
-----
    python ppo_agent.py                          # default: CartPole-v1
    python ppo_agent.py --env LunarLander-v3
    python ppo_agent.py --env MountainCar-v0

Design decisions:
  - Separate Actor and Critic networks (independent optimisers + LRs)
  - GAE-λ via scipy.signal.lfilter (discounted cumulative sums)
  - KL early stopping on the policy update
  - steps_per_epoch rollout structure
  - Per-environment hyperparameter profiles

Why per-env tuning is necessary
---------------------------------
CartPole-v1   : dense +1/step, short episodes (≤500), 4-dim state, 2 actions.
                Small net, 4000 steps/epoch, 30 epochs.

LunarLander-v3: shaped reward (±100 land/crash, leg contacts, velocity penalties),
                episodes up to 1000 steps, 8-dim state, 4 actions.
                Needs larger net, more steps/epoch, more epochs, and a lower
                TARGET_KL (policy is more sensitive — tighter trust region).

MountainCar-v0: reward = −1/step until pos ≥ 0.5, max 200 steps.
                Purely sparse signal → almost no gradient without shaping.
                Potential-based shaping: F(s,s') = γ·Φ(s') − Φ(s)
                where Φ(s) = position + velocity².
                Preserves the optimal policy (Ng et al. 1999).
                Raw return is always plotted (never the shaped value).

Output
------
    PPO_plots/<env>/
        ppo_curve.png
        ppo_agent.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
import scipy.signal

# ─────────────────────────────────────────────
# PATHS & SEED
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# PER-ENVIRONMENT HYPERPARAMETER PROFILES
# ─────────────────────────────────────────────
ENV_PROFILES = {
    "CartPole-v1": {
        # ── rollout ──
        "steps_per_epoch":         4_000,
        "epochs":                  30,
        # ── network ──
        "hidden_sizes":            (64, 64),
        # ── optimisation ──
        "policy_lr":               3e-4,
        "value_lr":                1e-3,
        "train_policy_iterations": 80,
        "train_value_iterations":  80,
        # ── GAE / PPO ──
        "gamma":                   0.99,
        "lam":                     0.97,
        "clip_ratio":              0.2,
        "target_kl":               0.01,
        # ── plotting ──
        "solve_threshold":         475,
        # ── reward shaping ──
        "shaped_reward":           False,
    },

    "LunarLander-v3": {
        # Larger rollout window for the longer, noisier episodes.
        # Lower target_kl — policy is more sensitive in this env;
        # a tighter trust region prevents catastrophic updates.
        # More epochs because the task is harder and the reward landscape
        # has sharp discontinuities (crash vs soft landing).
        # ── rollout ──
        "steps_per_epoch":         8_000,
        "epochs":                  60,
        # ── network ──
        "hidden_sizes":            (128, 128),   # 8-dim state, 4 actions
        # ── optimisation ──
        "policy_lr":               3e-4,
        "value_lr":                1e-3,
        "train_policy_iterations": 80,
        "train_value_iterations":  80,
        # ── GAE / PPO ──
        "gamma":                   0.99,
        "lam":                     0.97,
        "clip_ratio":              0.2,
        "target_kl":               0.005,        # tighter — env is sensitive
        # ── plotting ──
        "solve_threshold":         200,          # gym: ≥200 mean-100
        # ── reward shaping ──
        "shaped_reward":           False,        # reward already dense enough
    },

    "MountainCar-v0": {
        # Short episodes (max 200 steps) but extremely sparse reward.
        # More epochs needed because each epoch contains many failed attempts
        # before shaping guides the policy toward the goal.
        # Larger lam (0.99) to propagate credit further back through the
        # long sequences of −1 steps before a rare success.
        # ── rollout ──
        "steps_per_epoch":         4_000,
        "epochs":                  100,
        # ── network ──
        "hidden_sizes":            (64, 64),
        # ── optimisation ──
        "policy_lr":               3e-4,
        "value_lr":                1e-3,
        "train_policy_iterations": 80,
        "train_value_iterations":  80,
        # ── GAE / PPO ──
        "gamma":                   0.99,
        "lam":                     0.99,         # high lam — long credit assignment
        "clip_ratio":              0.2,
        "target_kl":               0.01,
        # ── plotting ──
        "solve_threshold":         -110,         # higher (less negative) = better
        # ── reward shaping ──
        "shaped_reward":           True,
    },
}

SUPPORTED_ENVS = list(ENV_PROFILES.keys())


# ─────────────────────────────────────────────
# REWARD SHAPING
# ─────────────────────────────────────────────
def shape_reward(obs, next_obs, raw_reward: float,
                 env_name: str, gamma: float) -> float:
    """
    Potential-based reward shaping  F(s, s') = γ·Φ(s') − Φ(s).
    Only active for MountainCar-v0 (raw reward is too sparse for PPO).

    Φ(s) = position + velocity²
      - position  → pushes the car toward the right-side goal
      - velocity² → rewards building kinetic energy (the swinging motion)

    Theory guarantee: adding F preserves the set of optimal policies
    because it is equivalent to potential-based shaping (Ng et al. 1999).
    Raw return is always tracked separately for honest reporting.
    """
    if env_name != "MountainCar-v0":
        return raw_reward
    phi_s  = obs[0]      + obs[1] ** 2
    phi_s2 = next_obs[0] + next_obs[1] ** 2
    return raw_reward + (gamma * phi_s2 - phi_s)


# ─────────────────────────────────────────────
# DISCOUNTED CUMULATIVE SUMS  (scipy lfilter)
# ─────────────────────────────────────────────
def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# ─────────────────────────────────────────────
# BUFFER
# ─────────────────────────────────────────────
class Buffer:
    """
    Fixed-size buffer for one steps_per_epoch rollout.
    finish_trajectory() computes GAE-λ advantages + rewards-to-go.
    get() normalises advantages, resets the pointer, and returns all arrays.
    """
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


# ─────────────────────────────────────────────
# NETWORKS
# ─────────────────────────────────────────────
def build_mlp(in_dim: int, hidden_sizes: tuple, out_dim: int) -> nn.Sequential:
    """Tanh MLP — standard for PPO policy/value networks."""
    layers, prev = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Outputs raw logits → Categorical policy."""
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
    """Outputs scalar V(s)."""
    def __init__(self, obs_dim: int, hidden: tuple):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, 1)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ─────────────────────────────────────────────
# PPO UPDATE FUNCTIONS
# ─────────────────────────────────────────────
def train_policy(actor, policy_optimizer, obs_t, act_t,
                 logp_old_t, adv_t, clip_ratio, target_kl):
    """One gradient step on the clipped surrogate. Returns approx KL."""
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
    """One gradient step on MSE value loss."""
    value_optimizer.zero_grad()
    loss = torch.mean((ret_t - critic(obs_t)) ** 2)
    loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    value_optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_ppo(env_name: str = "CartPole-v1"):
    assert env_name in SUPPORTED_ENVS, \
        f"Unsupported env '{env_name}'. Choose from: {SUPPORTED_ENVS}"

    cfg       = ENV_PROFILES[env_name]
    plots_dir = os.path.join(BASE_DIR, "PPO_plots", env_name)
    os.makedirs(plots_dir, exist_ok=True)

    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_act   = env.action_space.n

    actor  = Actor(obs_dim,  n_act, cfg["hidden_sizes"]).to(DEVICE)
    critic = Critic(obs_dim, cfg["hidden_sizes"]).to(DEVICE)
    policy_optimizer = optim.Adam(actor.parameters(),  lr=cfg["policy_lr"])
    value_optimizer  = optim.Adam(critic.parameters(), lr=cfg["value_lr"])

    buffer = Buffer(obs_dim, cfg["steps_per_epoch"], cfg["gamma"], cfg["lam"])

    print(f"\n{'='*60}")
    print(f"  PPO — {env_name}  (no world model)")
    print(f"  obs_dim={obs_dim} | n_actions={n_act} | device={DEVICE}")
    print(f"  hidden={cfg['hidden_sizes']} | "
          f"policy_lr={cfg['policy_lr']} | value_lr={cfg['value_lr']}")
    print(f"  steps/epoch={cfg['steps_per_epoch']} | epochs={cfg['epochs']}")
    print(f"  γ={cfg['gamma']} | λ={cfg['lam']} | "
          f"clip={cfg['clip_ratio']} | target_kl={cfg['target_kl']}")
    if cfg["shaped_reward"]:
        print(f"  reward shaping: ON (potential-based, γ·Φ(s')−Φ(s))")
    print(f"{'='*60}\n")

    all_rewards, mean_rewards = [], []
    observation, _ = env.reset(seed=SEED)
    ep_return, ep_length = 0.0, 0
    num_episodes_total = 0

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

            observation_new, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Shaped reward stored in buffer; raw reward tracked for plotting
            stored_reward = shape_reward(
                observation, observation_new, reward, env_name, cfg["gamma"])

            ep_return += reward          # always RAW for honest reporting
            ep_length += 1

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

                observation, _ = env.reset(
                    seed=SEED + num_episodes_total)
                ep_return, ep_length = 0.0, 0

        # ── PPO update ────────────────────────────────────────────────
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
                print(f"   [KL early stop] epoch={epoch+1} "
                      f"iter={pi_iter+1} kl={kl:.5f}")
                break

        for _ in range(cfg["train_value_iterations"]):
            train_value_function(critic, value_optimizer, obs_t, ret_t)

        mean_ret = sum_return / max(num_episodes, 1)
        mean_len = sum_length / max(num_episodes, 1)
        print(f"Epoch {epoch+1:03d}/{cfg['epochs']}  "
              f"MeanReturn={mean_ret:8.2f}  "
              f"MeanLen={mean_len:6.1f}  "
              f"Episodes={num_episodes:4d}  "
              f"TotalEps={num_episodes_total}")

    env.close()

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"PPO — {env_name} (no world model)", fontsize=13)

    ax = axes[0]
    ax.plot(all_rewards,  alpha=0.30, linewidth=0.8, label="Episode Return")
    ax.plot(mean_rewards, linewidth=2.0, label="Mean-25")
    threshold = cfg["solve_threshold"]
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.45,
               linewidth=1, label=f"Threshold ({threshold})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (raw)")
    ax.set_title("Learning Curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    window = 25
    if len(all_rewards) >= window:
        smooth = np.convolve(all_rewards, np.ones(window) / window, mode='valid')
        ax.plot(smooth, linewidth=2.0, color='#1D9E75',
                label=f"Smoothed (w={window})")
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.45,
               linewidth=1, label=f"Threshold ({threshold})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Return")
    ax.set_title(f"Smoothed Return (window={window})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    shaped_str = "ON" if cfg["shaped_reward"] else "OFF"
    fig.text(0.5, -0.04,
             f"hidden={cfg['hidden_sizes']}  "
             f"policy_lr={cfg['policy_lr']}  value_lr={cfg['value_lr']}  "
             f"γ={cfg['gamma']}  λ={cfg['lam']}  clip={cfg['clip_ratio']}  "
             f"target_kl={cfg['target_kl']}  "
             f"steps/epoch={cfg['steps_per_epoch']}  epochs={cfg['epochs']}  "
             f"shaping={shaped_str}",
             ha='center', fontsize=7.5, color='#555555')

    plt.tight_layout()
    fig_path = os.path.join(plots_dir, "ppo_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {fig_path}")

    # ── Save weights ──────────────────────────────────────────────────
    pth_path = os.path.join(plots_dir, "ppo_agent.pth")
    torch.save({
        "actor":       actor.state_dict(),
        "critic":      critic.state_dict(),
        "env_name":    env_name,
        "obs_dim":     obs_dim,
        "n_actions":   n_act,
        "hidden_sizes": cfg["hidden_sizes"],
    }, pth_path)
    print(f"[Model] Saved → {pth_path}")

    # ── Greedy evaluation ─────────────────────────────────────────────
    eval_env  = gym.make(env_name)
    obs_e, _  = eval_env.reset(seed=SEED + 111)
    total_r   = 0.0
    done_e    = False
    actor.eval()
    while not done_e:
        with torch.no_grad():
            logits = actor(torch.tensor(obs_e, dtype=torch.float32,
                                        device=DEVICE).unsqueeze(0))
            a_e    = int(torch.argmax(logits).item())
        obs_e, r_e, term_e, trunc_e, _ = eval_env.step(a_e)
        done_e  = term_e or trunc_e
        total_r += r_e
    print(f"[Eval] Greedy episode reward = {total_r}")
    eval_env.close()

    return actor, critic, all_rewards, mean_rewards


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
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