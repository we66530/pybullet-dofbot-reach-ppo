# pybullet-dofbot-reach-ppo
PyBullet-based DOFBOT reaching task using PPO reinforcement learning. Includes a Gymnasium-compatible environment, vectorized training, CSV logging, and delta-angle position control designed for transfer to real servo-based DOFBOT arms.
## 🎮 Demos & Usage

This repository provides multiple scripts for visualization, interactive control, training, and inference of a PPO-based DOFBOT reaching task in PyBullet.

---

### 1️⃣ PyBullet GUI Visualization

Launch the basic PyBullet GUI to visualize the DOFBOT in the scene.

```bash
python show_dofbot_gui.py
```

### 2️⃣ Interactive Slider Control

Run the DOFBOT with real-time joint control using GUI sliders, useful for kinematic inspection and debugging.

```bash
python show_dofbot_with_slider.py
```

### 3️⃣ PPO Training
Train the DOFBOT reaching policy using Proximal Policy Optimization (PPO).

```bash
python train_dofbot_reach_ppo.py
```

Default training steps: 500,000

Training length and hyperparameters can be freely modified in the script

✅ Trained models will be saved automatically for later evaluation.

### 4️⃣ Policy Inference

```bash
python test_dofbot_reach_policy.py
```

