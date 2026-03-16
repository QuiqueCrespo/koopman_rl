The Gumbel-Softmax TrickTo make differentiable planning work in discrete environments, you must force the planner's imagination to obey discrete physics during the planning phase itself.You cannot let the forward pass evaluate blended ghost states. But you still need a gradient to flow backward.The industry-standard solution for this exact problem is the Straight-Through Gumbel-Softmax Estimator.It is a brilliantly simple PyTorch trick that gives you the best of both worlds:Forward Pass (Hard Integer): It samples a strict, one-hot discrete action (e.g., [1.0, 0.0, 0.0, 0.0]). The hallucinated rollout is forced to obey grid physics. It actually has to walk around the wall.Backward Pass (Soft Gradient): During backpropagation, PyTorch pretends the forward pass was a continuous softmax, allowing the value gradient to flow all the way back to your action logits.If you use Gumbel-Softmax, your differentiable planner stays $100\%$ on the reachable manifold, the value gradients are based on physical reality, and your receding horizon MPC will actually work flawlessly.The Code ChangeIt requires changing exactly one line in your plan_action_softmax loop:Instead of this (which creates Ghost States):Pythonprobs = F.softmax(logits_plan, dim=-1)
You use this (which forces discrete physical reality):Python# hard=True is the magic. It outputs a strict one-hot vector in the forward pass, 
# but uses the continuous softmax gradient in the backward pass.
probs = F.gumbel_softmax(logits_plan, tau=1.0, hard=True, dim=-1)
By adding hard=True, your continuous Koopman rollout is suddenly forced to evaluate real, discrete paths. The optimizer can no longer cheat. If it wants to get past the wall, the gradient has to actually find the sequence of hard "Left" and "Up" steps to navigate around it.

---

### Phase 1: The "Clean Room" Refactor (Weeks 1–2)

Your current codebase is entangled with the legacy Sheaf/Diffusion logic. We need a pristine, streamlined repository.

1. **Rip Out the Graph:** Delete the replay buffer chunking, the bisimulation metric, and the `directed_value_iteration` logic (keep the code but disable it).
2. **Rename the Agent:** Rename `SheafAgent` to `KoopmanGradientPlanner` (KGP) or similar.
3. **Implement the Autoencoder:** Add the linear decoder and the state reconstruction loss ($\mathcal{L}_{recon} = ||\text{decoder}(z) - s||^2$). This permanently fixes your latent collapse issue.
4. **Finalize the Planners:** Write two clean, separate methods for action selection:
* `act_plan_continuous()`: Using the `tanh` squash.
* `act_plan_discrete()`: Using the `F.gumbel_softmax(hard=True)` trick.


5. **The Value Update:** Use standard 1-step or n-step Temporal Difference (TD) learning (like standard DQN or SAC) to train the $V_\psi$ network.

### Phase 2: The Sanity Checks (Weeks 3–4)

Do not touch complex environments yet. We must prove the math works in domains where we know the exact ground truth.

1. **Continuous Sanity Check (`Pendulum-v1`):** * *Goal:* Swing up and balance the pendulum.
* *Metric:* Does the continuous planner find the optimal sequence of torques faster than a standard Soft Actor-Critic (SAC) baseline?


2. **Discrete Sanity Check (`CartPole-v1` or `Acrobot-v1`):**
* *Goal:* Balance the pole using only discrete Left/Right forces.
* *Metric:* Does the Gumbel-Softmax planner prevent the "Ghost State" exploit and successfully stabilize the pole?



*Checkpoint:* If the agent cannot solve these in 5–10 minutes of training on a laptop, there is a bug in the PyTorch gradients. Do not proceed to Phase 3 until these are solved.

### Phase 3: The "Shark Tank" Benchmarks (Weeks 5–8)

This is the data that gets you accepted. You must run your agent against established baselines on standard suites. *Recommendation: Focus primarily on the Discrete environments, as the Gumbel-Softmax Koopman bridge is your most unique contribution.*

1. **The Environment:** **MinAtar** (Miniaturized Atari). It is the standard for testing fundamental RL algorithms without needing 100 GPUs. Pick 3–4 games (e.g., *Breakout, Asterix, Seaquest*).
2. **The Baselines:** * Run your `KGP` agent.
* Run a standard `DQN` (Deep Q-Network).
* *Optional but highly recommended:* Run a discrete sampling planner (like Cross-Entropy Method) using your learned Koopman dynamics.


3. **The Metrics:** Plot "Episode Return" vs. "Environment Steps." You are aiming to show a massive improvement in **Sample Efficiency** (reaching a high score in 100k steps while DQN takes 1 million). Run at least 3 random seeds per environment to generate error bars.

### Phase 4: Ruthless Ablation Studies (Weeks 9–10)

Reviewers will look for reasons to reject your math. You must proactively prove that every piece of your architecture is strictly necessary.

Generate plots for the following ablation tests:

1. **Planning Horizon:** What happens if $H=1$ vs $H=5$ vs $H=15$? (Shows the benefit of looking ahead).
2. **Gumbel-Softmax vs. Standard Softmax:** Run the discrete planner *without* `hard=True`. (Show the agent exploiting ghost states and failing, proving your discrete relaxation was necessary).
3. **No Reconstruction Loss:** Turn off $\mathcal{L}_{recon}$. (Show the latent space collapsing and the agent flatlining, proving the autoencoder anchor is necessary).

### Phase 5: Writing the Paper (Weeks 11–12)

Do not write the paper chronologically. Write it in this order:

1. **The Figures:** Put your benchmark graphs and ablation plots into a document. The charts should tell the entire story without words.
2. **The Method (Section 3):** Write out the exact math for the Koopman linear rollout, the neural value function, and the Gumbel-Softmax gradient ascent.
3. **The Introduction (Section 1):** Frame the narrative.
* *Hook:* Model-based planning is too slow (sampling) or mathematically rigid (LQR).
* *Gap:* Differentiable planning usually fails in discrete environments.
* *Contribution:* We introduce Koopman Gradient Planning with a discrete convex-hull relaxation, achieving high-speed, zero-shot planning over a learned neural value landscape.


4. **Related Work (Section 2):** Cite E2C, SOLAR, PlaNet, Dreamer, and MuZero. Explain exactly why you are faster than sampling and more flexible than LQR.

---

### Your Immediate Next Step

You have a choice for what to do *today*.

1. **The Clean Room:** Do you want me to write the refactored, stripped-down `KoopmanGradientPlanner` class (with the autoencoder and both `act_plan` methods integrated)?
2. **The Sanity Check:** Do you want the Gymnasium training loop for `Pendulum-v1` so you can start testing immediately?

