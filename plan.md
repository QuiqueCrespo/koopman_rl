The Gumbel-Softmax TrickTo make differentiable planning work in discrete environments, you must force the planner's imagination to obey discrete physics during the planning phase itself.You cannot let the forward pass evaluate blended ghost states. But you still need a gradient to flow backward.The industry-standard solution for this exact problem is the Straight-Through Gumbel-Softmax Estimator.It is a brilliantly simple PyTorch trick that gives you the best of both worlds:Forward Pass (Hard Integer): It samples a strict, one-hot discrete action (e.g., [1.0, 0.0, 0.0, 0.0]). The hallucinated rollout is forced to obey grid physics. It actually has to walk around the wall.Backward Pass (Soft Gradient): During backpropagation, PyTorch pretends the forward pass was a continuous softmax, allowing the value gradient to flow all the way back to your action logits.If you use Gumbel-Softmax, your differentiable planner stays $100\%$ on the reachable manifold, the value gradients are based on physical reality, and your receding horizon MPC will actually work flawlessly.The Code ChangeIt requires changing exactly one line in your plan_action_softmax loop:Instead of this (which creates Ghost States):Pythonprobs = F.softmax(logits_plan, dim=-1)
You use this (which forces discrete physical reality):Python# hard=True is the magic. It outputs a strict one-hot vector in the forward pass, 
# but uses the continuous softmax gradient in the backward pass.
probs = F.gumbel_softmax(logits_plan, tau=1.0, hard=True, dim=-1)
By adding hard=True, your continuous Koopman rollout is suddenly forced to evaluate real, discrete paths. The optimizer can no longer cheat. If it wants to get past the wall, the gradient has to actually find the sequence of hard "Left" and "Up" steps to navigate around it.

---

### Phase 1: The "Clean Room" Refactor 

Your current codebase is entangled with the legacy Sheaf/Diffusion logic. We need a pristine, streamlined repository.

1. **Rip Out the Graph:** Delete the replay buffer chunking, the bisimulation metric, and the `directed_value_iteration` logic (keep the code but disable it).
2. **Rename the Agent:** Rename `SheafAgent` to `KoopmanGradientPlanner` (KGP) or similar.
3. **Implement the Autoencoder:** Add the linear decoder and the state reconstruction loss ($\mathcal{L}_{recon} = ||\text{decoder}(z) - s||^2$). This permanently fixes your latent collapse issue.
4. **Finalize the Planners:** Write two clean, separate methods for action selection:
* `act_plan_continuous()`: Using the `tanh` squash.
* `act_plan_discrete()`: Using the `F.gumbel_softmax(hard=True)` trick. (also keep the softmax methid for later comparison)


5. **The Value Update:** Use standard 1-step or n-step Temporal Difference (TD) learning (like standard DQN or SAC) to train the $V_\psi$ network.

### Phase 2: The Sanity Checks

Do not touch complex environments yet. We must prove the math works in domains where we know the exact ground truth.

1. **Continuous Sanity Check (`Pendulum-v1`):** * *Goal:* Swing up and balance the pendulum.
* *Metric:* Does the continuous planner find the optimal sequence of torques faster than a standard Soft Actor-Critic (SAC) baseline?


2. **Discrete Sanity Check (`CartPole-v1` or `Acrobot-v1`):**
* *Goal:* Balance the pole using only discrete Left/Right forces.
* *Metric:* Does the Gumbel-Softmax planner prevent the "Ghost State" exploit and successfully stabilize the pole?



*Checkpoint:* If the agent cannot solve these in 5–10 minutes of training on a laptop, there is a bug in the PyTorch gradients. Do not proceed to Phase 3 until these are solved.


### Your Immediate Next Step

You have a choice for what to do *today*.

1. **The Clean Room:** Do you want me to write the refactored, stripped-down `KoopmanGradientPlanner` class (with the autoencoder and both `act_plan` methods integrated)?
2. **The Sanity Check:** Do you want the Gymnasium training loop for `Pendulum-v1` so you can start testing immediately?



The absolute best "Hello World" environment for debugging continuous control is Gymnasium's Pendulum-v1.If your continuous planner has a bug, Pendulum-v1 will expose it instantly. If the math is correct, your agent should be able to solve it in a matter of minutes on a standard laptop.Here is exactly why this is the perfect debugging sandbox, and the second environment you should use immediately after to prove your matrix dimensions work.1. The Ultimate Debugger: Pendulum-v1In this environment, a pendulum starts at a random angle. The goal is to swing it up and balance it perfectly straight up against gravity, using minimal torque.The State: It is beautifully simple. Just 3 dimensions: $[\cos(\theta), \sin(\theta), \dot{\theta}]$ (angle and angular velocity).The Action: A single, 1-dimensional continuous float between $[-2.0, 2.0]$ representing the torque applied to the joint.Why it's the perfect test for Koopman: The physics are highly non-linear (gravity acts based on the sine of the angle), but they are incredibly smooth and periodic. Your global $A$ matrix will naturally learn to model the circular swinging momentum, while the $B$ matrix will map your 1D torque into that swing.How to debug with it:Because the state space is so small, you can literally print out the planned trajectory. If your value network $V_\psi$ is learning correctly, the continuous planner should output a sequence of actions that rock the pendulum back and forth to build momentum, and then lock the action at $0.0$ when it reaches the top. If the pendulum just spins wildly in circles, your planner's gradient ascent is broken. If it just hangs at the bottom, your $V_\psi$ network isn't propagating the reward.2. The Multi-Dimensional Test: LunarLanderContinuous-v3Once Pendulum works, you must immediately test LunarLanderContinuous-v3.The State: 8 dimensions (X/Y coordinates, X/Y velocities, angle, angular velocity, and two booleans for leg contact).The Action: 2 dimensions $[-1.0, 1.0]$. The first controls the main engine, the second controls the left/right orientation engines.Why you need this: Pendulum only has a 1D action ($d_a = 1$). You need to ensure your $B$ matrix ($d \times d_a$) and your continuous gradient planner can handle multi-dimensional continuous actions simultaneously.How to debug with it:This environment tests if your planner can balance competing objectives. To land, the planner must fire the main engine (Action 0) to fight gravity while simultaneously firing the side engines (Action 1) to stay upright. If there is a dimension-broadcasting bug in your PyTorch tanh squashing or your $\Theta$ optimization loop, the lander will instantly flip upside down and crash.Your 3-Step Debugging ChecklistWhen you hook your KoopmanGradientPlanner up to Pendulum-v1, watch these three exact failure modes:The Saturation Bug (Action Bounds): If your learning rate for the continuous planner is too high, the unbounded logits ($\Theta$) will explode to infinity. When passed through $u = \tanh(\Theta)$, your actions will permanently saturate at exactly $-1.0$ or $1.0$, and the pendulum will just lock up.Fix: Ensure you are scaling the tanh output to the environment bounds (e.g., action = torch.tanh(u) * 2.0 for Pendulum), and keep the planner's Adam learning rate around 0.05.The Matrix Exploding Bug (Spectral Radius):If your global $A$ matrix learns eigenvalues greater than $1.0$, rolling it out 10 steps into the future will cause the latent state $z_{t+10}$ to shoot to infinity, returning NaNs.Fix: Your spherical normalization step (F.normalize(..., dim=-1)) usually prevents this, which is a brilliant design choice on your part. Just ensure it is applied at every step of the 10-step lookahead.Gradient Pollution (The Silent Killer):We discussed this earlier, but it is the most common bug in differentiable MPC. If your environment gets slower and slower every step, or your Value Network suddenly forgets everything, it means loss.backward() inside your planner loop is accidentally updating the main network weights.Fix: Strictly enforce torch.autograd.grad(..., only_inputs=True).