# Sensor Simulation — Presentation Notes

> Technical and strategic presenter notes for the **Sensor Simulation with Agentic Workflow** deck.
> Audience: engineering leadership, LHA stakeholders (XCATH, Remedy), I4H program management.
> Tone: technically precise, strategically framed, demo-ready.

---

## SLIDE 1 — Title

**Sensor Simulation: X-Ray–Guided Robotic Catheter Interventional System**
Isaac for Healthcare

**Say:**

"This presentation covers three things. First, the current state of our sensor simulation stack — what's built, what's running, what the numbers are. Second, the gap analysis against XCATH's production requirements — what we've closed and what remains. Third, the agentic workflow architecture that turns a multi-week manual experiment cycle into a multi-hour automated loop. I'll go deep on the physics where it matters and keep the strategic framing tight."

---

## SLIDE 2 — Sensor Overview Matrix

**Say:**

"We have five sensor modalities on the roadmap. Two are implemented today — X-ray fluoroscopy and ultrasound B-mode. Both use Slang as the GPU rendering backend. X-ray fluoroscopy is the furthest along and the primary focus for I4H v0.6: we have a fully differentiable volume renderer via DiffDRR, and as of this sprint, we have fused volume-plus-catheter Beer-Lambert compositing running on GPU at 25 frames per second."

"Ultrasound is implemented with volumetric ray marching, BVH-accelerated triangle intersection, and full 6-DOF differentiable rendering through PyTorch autograd. The remaining three — endoscopy, CBCT, and force/torque — are planned and I'll touch on those at the end."

"The critical gap for training readiness is multi-env support. Both sensors are single-env today. Sprint 2 targets 512+ parallel environments for X-ray, which is the minimum for PPO-class RL training at reasonable wall-clock time."

---

## SLIDE 3 — Catheter Physics Solvers

**Say:**

"The catheter is modelled as a Cosserat rod — a one-dimensional continuum with both translational and rotational degrees of freedom at each node. Three solver backends exist, all exposing the same interface: a tensor of 3D centerline positions in metres."

"The production solver is the one that matters for training. It's a fully batched XPBD solver with mesh BVH collisions, proximal kinematic control for push-and-rotate actions, and a direct solve on the block-tridiagonal JMJT system. It runs at 1,300 Hz single-env on an A6000 — well above the 1,000 Hz target."

"The self-contained XPBD solver is our research workhorse. It implements the same Block-Thomas O(n) direct solve entirely in Warp kernels with zero external dependency on the Newton library. This is the solver we pair with the Slang renderer in the unified sim loop because it has no Isaac Sim overhead."

"The Newton bridge wraps the external Newton library's XPBD rod solver. It's single-env only and used primarily for validation against the production solver — ensuring our Warp reimplementation matches Newton's reference output."

"All three solvers output positions in SI metres. The compositing pipeline multiplies by 1000 to convert to millimetres, which is the coordinate convention used by clinical C-arm geometry — SID, SOD, and pixel spacing are all specified in mm."

---

## SLIDE 4 — Fluoroscopy Rendering: Three Compositing Paths

**Say:**

"Three compositing paths exist, ordered by fidelity and performance."

"The headline result is the Slang GPU path. This is a single fused ray march through the patient's CT attenuation volume and the catheter cylinder geometry simultaneously. One shader dispatch, one Beer-Lambert integral: I equals I-naught times exponential of negative integral of total mu along the ray. Both the volume mu and the catheter mu accumulate in the same exponent. This runs at 25 FPS on an A6000 at 512-by-512 resolution. Zero CPU compositing. The catheter segments are uploaded as a structured GPU buffer — endpoint pairs, radius, and attenuation coefficient per segment."

"The CPU Beer-Lambert path is the fallback with full detector physics. It computes the exact cylinder chord thickness per pixel — that's the 2-root-of-r-squared-minus-d-squared formula from ray-cylinder intersection — accumulates an attenuation map, applies the Beer-Lambert exponential, then adds veiling glare, detector PSF blur, and Poisson quantum noise. It's physically rigorous but runs at 2–5 FPS because it's pure NumPy. The fix is straightforward: port the per-pixel loop to a Warp GPU kernel, which would push it to 30+ FPS."

"The Isaac Lab 3D path is for Omniverse visualization. It renders the catheter as capsule markers over a textured DRR quad in the viewport. It's real-time but not physically correct — the catheter is an opaque 3D mesh, not a transmission shadow."

"The key technical insight is that Beer-Lambert compositing is multiplicative, not additive. The catheter darkens the background rather than painting over it. This is the correct physics: a radiopaque device in an X-ray beam attenuates photons — it doesn't occlude them. Self-crossings where the catheter loops back over itself produce correctly increased attenuation because the mu values add in the exponent before exponentiation."

---

## SLIDE 5 — Beer-Lambert Compositing: The Physics Chain

**Say:**

"Let me walk through the five-stage detector physics chain implemented in the CPU path. This is the pipeline that produces images indistinguishable from real fluoroscopy."

"Stage one: attenuation map construction. For each of the N-minus-1 catheter segments, we compute the perpendicular distance from every pixel to the segment centreline, then evaluate the cylinder chord function — two times the square root of r-squared minus d-squared — where r is the cone-beam-magnified projected radius. This chord is normalized by the diameter and weighted by the segment's linear attenuation coefficient mu. The result is accumulated additively into a single-channel attenuation map. Additive accumulation in the exponent is critical — it correctly models stacked attenuators without double-counting."

"Stage two: Beer-Lambert transmission. I-final equals I-DRR times exp of negative attenuation-map. This is the core physics. Where mu is zero, transmission is 1 — background passes through unchanged. Where mu is 5.0 — a platinum marker band — transmission is 0.7%, essentially opaque."

"Stage three: veiling glare. The intensity blocked by the catheter — I-DRR times one-minus-transmission — is blurred with a large Gaussian kernel, sigma 18 pixels, and 3% of that is added back. This models photon scatter in the patient tissue and detector housing. The large sigma approximates the long-range scatter halo you see on real flat-panel detectors."

"Stage four: detector PSF. A small Gaussian blur at sigma 0.7 pixels models the finite spatial resolution of the CsI scintillator. This softens the edges slightly — matching real detector optics."

"Stage five: Poisson quantum noise at 2000 photons per pixel. The composited intensity is scaled to photon counts, Poisson-sampled, and scaled back. This matches the shot noise characteristics of low-dose pulsed fluoroscopy at 1–3 millirad per frame. This noise is critical for domain randomization — a policy trained on clean images will fail on real noisy fluoroscopy."

---

## SLIDE 6 — Per-Segment Attenuation Profile

**Say:**

"The catheter is not uniformly radiopaque. A real device has distinct material zones, and each zone has a different effective attenuation coefficient. We model this with a five-zone piecewise profile."

"At the proximal end: tungsten marker bands at mu 3.0. These are the bright landmarks the interventionalist uses for orientation. Then the braided nitinol shaft at mu 0.8 — moderately opaque, matching a typical 6-French microcatheter. A transition zone from 60% to 85% of the rod length where the braid density tapers from mu 0.8 down to 0.2. Then the soft polymer tip — PEBAX — at mu 0.15, nearly transparent. And finally the distal platinum coil marker at mu 5.0 — the densest, darkest feature on the image."

"This profile is not cosmetic. It drives correct training signal. A policy that navigates by tip position needs to see the tip marker clearly against anatomy. A policy that tracks the shaft needs accurate shaft visibility. Getting the mu profile wrong means the policy learns from unrealistic visual features and fails sim-to-real."

---

## SLIDE 7 — C-arm Projection Model

**Say:**

"The fluoroscopy projection uses a standard pinhole camera model parameterised by interventional C-arm conventions."

"Intrinsics: focal length in pixels equals SID divided by pixel spacing — that's source-to-image distance over detector pitch. At SID 1000 mm and 0.81 mm per pixel, focal length is 1235 pixels. Principal point at detector centre."

"Extrinsics: rotation is composed as R-x of cranio-caudal angle times R-y of LAO/RAO angle. The X-ray source sits at (0, 0, negative SOD) in the camera frame — 600 mm from the iso-centre. The iso-centre is at the world origin."

"The critical detail is cone-beam magnification. Each segment's projected radius scales as physical radius times focal length divided by camera-frame depth: r-px equals r-physical times f over z-cam. Segments closer to the X-ray source appear larger on the detector. This exactly matches real cone-beam geometry and matters for correct catheter shadow width — a 1 mm catheter at 200 mm depth projects differently than the same catheter at 400 mm depth."

---

## SLIDE 8 — Slang GPU Unified Sim Loop

**Say:**

"This slide shows the complete data flow of the unified sim loop — physics to pixels in one tight cycle."

"The XPBD rod solver steps the catheter forward in time. Positions come out as an N-by-3 tensor in metres. We scale to millimetres and offset by the volume centre to place the catheter inside the CT coordinate frame."

"Those positions become a structured GPU buffer: each segment is an 8-float record — two endpoints, radius, and mu. This buffer is uploaded to the Slang renderer as a StructuredBuffer."

"The Slang shader dispatches one thread per detector pixel. Each thread casts a ray from the X-ray source through its pixel, then marches along the ray in fixed 0.5 mm steps. At each step, it samples the CT mu volume via trilinear interpolation, and separately evaluates catheter attenuation by looping over all segments and checking ray-cylinder proximity. Both contributions accumulate into a single running integral. At the end of the march: I equals I-naught times exp of negative integral. One dispatch, one readback, one frame."

"The performance is 25 FPS on an RTX A6000. The bottleneck is the per-step catheter segment loop — it's O(steps times segments). For 30 segments and 2048 ray steps, that's 60,000 intersection tests per pixel. We can improve this with a spatial acceleration structure — BVH or uniform grid over segments — which would push to 60+ FPS."

---

## SLIDE 9 — Performance Baseline

**Say:**

"Physics: 1,300 Hz single-env, exceeding the 1,000 Hz target. 512-env is Sprint 2."

"Slang GPU compositing: 40 milliseconds per frame — 25 FPS. This is for a full 512-by-512 fused DRR-plus-catheter render with no CPU compositing. Target was under 5 ms; we're 8x off, but the path to 60+ FPS is clear — spatial acceleration for the catheter segment loop and kernel occupancy tuning."

"CPU Beer-Lambert: 200–500 ms per frame. This is the full physics chain including scatter, PSF, and Poisson noise. Acceptable for offline dataset generation but not real-time training. Porting to a Warp GPU kernel is the fix."

"State-based RL: PPO training runs today at 512 parallel environments with the production solver. No pixel observations — state-only. Image-based RL requires either the Slang GPU path at multi-env scale or the Warp GPU compositing kernel."

---

## SLIDE 10 — XCATH Gap Analysis

**Say:**

"This is the competitive gap analysis against XCATH's production simulation stack. XCATH has five validated rendering modes on top of CTA volumes — DRR, Vessel Boost, DSA, Bolus Label, and Bolus Centerline. They need paired synthetic training data — CTA plus fluoroscopy plus ground-truth catheter pose — that does not exist in any clinical dataset."

"Six gaps are now closed. Beer-Lambert catheter compositing — closed, with both a CPU path and a fused GPU path that XCATH doesn't have. Per-segment attenuation profile — closed. Detector physics: Poisson noise, veiling glare, detector PSF — all closed on the CPU path. And we've added a capability XCATH lacks entirely: fused GPU DRR-plus-catheter rendering in a single shader dispatch."

"The remaining gaps cluster around DSA-specific features. Vessel Boost — amplifying vessel voxel mu by a factor of 8 — is a one-day implementation. The 4-step DSA pipeline — contrast DRR, mask DRR with jitter, log subtraction, gamma correction — is approximately three days. Bolus tracking with time-varying contrast dynamics is the largest remaining item at roughly one week."

"Strategically: the gaps we've closed are the hard ones — the physics engine, the rendering kernel, the differentiable pipeline. The remaining gaps are image processing and configuration — they're engineering work, not research risk."

---

## SLIDE 11 — End-to-End Pipeline: Four Stages

**Say:**

"The agentic workflow decomposes the catheter intervention problem into four stages. Each stage produces artifacts that feed the next."

"Stage 1: Patient Digital Twin. Raw CTA volume goes in. Out comes an HU-to-mu mapped 3D attenuation volume, a binary vessel mask, a VMTK centerline graph, and a Dijkstra arrival map encoding contrast travel time per voxel."

"Stage 2: Physics Simulation plus Compositing. The XPBD rod solver steps the catheter. Two compositing modes are available. The fused GPU path renders DRR-plus-catheter in a single ray march at 25 FPS. The CPU path provides full detector physics — scatter, PSF, Poisson. A third mode — max-attenuation volume compositing via atomic-max on GPU — is planned for true volumetric instrument injection where the catheter mu replaces the volume mu at occupied voxels."

"Stage 3: Sensor Simulation. On the Slang GPU path, stages 2 and 3 collapse into a single dispatch — there is no separate rendering step. On the CPU path, realism effects are applied after compositing: veiling glare at 3% scatter fraction, detector PSF at 0.7 px sigma, and Poisson noise at 2000 photons per pixel."

"Stage 4: Policy Training. Today this is state-based PPO at 512 parallel environments. The target is image-based RL using fluoroscopy pixel observations, which requires multi-env rendering — Sprint 2. The future pipeline is teleop demos into imitation learning via GR00T-H behavioral cloning, then RL fine-tuning with PPO or SAC, then simulation-in-the-loop evaluation."

---

## SLIDE 12 — OpenClaw Skills Architecture

**Say:**

"Each pipeline stage is wrapped as an OpenClaw skill — a portable, self-describing unit that the agent discovers, configures, and chains."

"Seven skills cover the full loop. Skill 1: patient-digital-twin — CTA to mu volume, vessel mask, centerline, and arrival map. The agent decides segmentation thresholds, injection root, and hemisphere selection. Skill 2: catheter-physics-sim — configures the rod solver, SDF collision, and selects the compositing mode. Skill 3: sensor-sim-xray — selects the rendering mode, C-arm preset, and realism parameters. Skill 4: dataset-creation — packages paired multimodal data as HDF5 with proper train/val/test splits. Skill 5: reward-function — configures the multi-component RL reward. Skill 6: policy-training — runs IL or RL with configurable algorithm, schedule, and stopping criteria. Skill 7: evaluation — runs SIL episodes and reports success rate, navigation time, contact forces, dose, and FID."

"The key design principle is that skills are composable and the agent reasons between them. After evaluation, the agent analyzes failure modes — for example, 'policy fails at ICA-MCA bifurcation' — and autonomously proposes configuration changes for the next cycle: adjust curriculum, increase demos at failure points, tweak domain randomization."

---

## SLIDE 13 — Rendering Modes and Realism Features

**Say:**

"The sensor-sim-xray skill exposes six rendering modes, three of which are implemented today."

"Standard DRR: pure volume ray-march, no catheter. Implemented and differentiable. DRR-with-catheter: fused volume plus catheter in a single ray march — our headline capability. CPU Beer-Lambert: pre-rendered DRR PNGs composited with the full detector physics chain."

"Three modes are planned: Vessel Boost — multiply vessel voxel mu by amplification factor A, typically 8. DSA — the clinical gold standard for catheter navigation, requiring a four-step pipeline of contrast DRR, mask DRR with geometric jitter, log subtraction, and gamma-corrected post-processing. And temporal DSA with bolus dynamics — per-frame mu updates driven by a gamma-variate contrast arrival model."

"On realism features: the CPU path has the full chain — Beer-Lambert, per-segment mu, cone-beam magnification, veiling glare, detector PSF, and Poisson noise. The Slang GPU path currently has Beer-Lambert, per-segment mu, and implicit cone-beam magnification from the 3D geometry. Scatter, PSF, and Poisson are not yet in the shader — they'd need to be added as post-processing passes or integrated into the ray march."

"Still missing across both paths: gamma correction for clinical display transfer function, physics-based scatter beyond the 2D veiling approximation, beam hardening correction for polyenergetic spectra, and misregistration jitter for DSA mask subtraction."

---

## SLIDE 14 — Dataset Creation Skill

**Say:**

"Skill 4 — dataset-creation — is the bridge between simulation and learning. It takes raw simulation output and packages it into the paired, labeled, split-ready datasets that imitation learning and RL pipelines consume. Without this skill, every researcher manually wrangles files, invents ad-hoc formats, and loses reproducibility. With it, the agent produces a training-ready artifact in one call."

"The core output is a paired multimodal dataset. Every record contains: a fluoroscopy frame — the 512-by-512 Beer-Lambert composite — alongside the catheter's 3D centerline pose at that timestep, the ground-truth C-arm extrinsics — rotation and translation — contact force vectors from the collision solver, and a microsecond-resolution timestamp linking the frame to the physics clock. This is the data contract: any downstream model — PoseNet, segmentation, RL value function — can index into the same HDF5 and pull exactly the modalities it needs."

"The agent makes four decisions when invoking this skill."

"First: dataset size. How many episodes, how many frames per episode. A typical run is 200 episodes at 50 frames each — 10,000 paired records. The agent scales this based on the target model's data appetite and available GPU hours."

"Second: domain randomization ranges. The agent specifies randomization bounds for each physics and rendering parameter — catheter stiffness sampled uniformly from 1e7 to 5e8 Pa, C-arm LAO/RAO angle from negative 30 to positive 30 degrees, Poisson photon count from 1,000 to 5,000 to span low-dose to cine acquisition, patient anatomy offset within the field of view. Every parameter drawn per-episode is stored alongside the record as metadata, so the training pipeline can condition on or marginalize over any axis of variation."

"Third: train/val/test split. Default is 80/10/10 by episode — not by frame — so no temporal leakage between splits. The agent can adjust ratios for few-shot regimes or large-scale pretraining."

"Fourth: storage format. HDF5 is the default — hierarchical, random-access, compression-friendly. WebDataset is the alternative for streaming large-scale training on distributed clusters where sequential tar-based reads outperform random-access IO. The agent selects based on the downstream training infrastructure."

"The dataset schema is designed for XCATH's paired data requirement: CTA plus fluoroscopy plus ground-truth pose. This is exactly the data that does not exist in any clinical dataset — the whole point of the simulation stack is to synthesize it at scale with perfect ground truth. Every frame has exact catheter tip coordinates in 3D, exact C-arm geometry, and exact contact forces — labels that are physically impossible to obtain from clinical fluoroscopy."

"One strategic point: dataset-creation is also the skill that enables offline evaluation of rendering realism. By storing frames alongside their generation parameters, we can compute FID against a reference distribution of real clinical fluoroscopy images. The evaluation skill consumes these datasets to quantify the sim-to-real gap and feed that signal back to the agent's iterative refinement loop."

---

## SLIDE 15 — Reward Function Design

**Say:**

"The RL reward has seven components, weighted to balance navigation efficiency against safety constraints."

"Target proximity is the dominant term at weight 1.0 — negative L2 distance from tip to goal. Progress reward at 0.2 gives credit for reducing distance over time, avoiding sparse-reward problems. Success bonus at 10.0 provides a strong terminal signal."

"Safety terms: wall contact penalty at 0.5 penalizes forces exceeding a threshold — this is how we prevent the policy from pushing through vessel walls. Tip force penalty at 0.3 specifically penalizes distal tip forces to avoid perforation — the most dangerous failure mode."

"Efficiency terms: procedure time penalty at 0.1 encourages faster navigation. Fluoroscopy dose penalty at 0.05 minimizes imaging frames — matching the ALARA principle in real interventional radiology."

"Today's implementation uses only distance-to-target with push/rotate actions. The full reward table is the target specification for the complete training pipeline."

---

## SLIDE 16 — Policy Training Skill

**Say:**

"Skill 6 — policy-training — is where simulation converts into autonomy. It takes the paired datasets from Skill 4 and the reward specification from Skill 5, and produces a trained catheter navigation policy."

"The training pipeline has three stages, executed sequentially. First: imitation learning. Teleop demonstrations — collected by a human operator or a scripted planner driving the catheter through the vessel tree — are fed into GR00T-H behavioral cloning. The model learns a base policy that can navigate the coarse trajectory. IL alone gets you to roughly 50–60% success rate — it captures the general strategy but lacks the precision for bifurcation decisions and tight vessel segments."

"Second: RL fine-tuning. The IL-initialized policy is refined with on-policy RL — PPO or SAC — running in the multi-env catheter simulation. This is where the reward function matters. The policy learns to minimize tip force at vessel walls, reduce fluoroscopy dose, and maximize progress toward the target. PPO is the default because it's stable with the high-dimensional action space of proximal push-and-rotate control. SAC is the alternative when sample efficiency matters more than stability — fewer environment steps but more sensitive to hyperparameters."

"Third: SIL checkpoint evaluation. Periodically during RL training, the agent snapshots the policy and runs simulation-in-the-loop evaluation — 100 episodes with the full physics and rendering stack. This produces the metrics that feed back into the agent's reasoning: success rate, mean navigation time, max contact force. The agent uses these to decide whether to continue training, adjust learning rate, or terminate early."

"The agent makes five decisions when configuring this skill. IL epochs — typically 50 to 200, depending on demo quality and quantity. RL algorithm — PPO for stability, SAC for efficiency. Learning rate schedule — linear decay is the default; cosine annealing for longer runs. Checkpoint frequency — every 100 epochs balances storage cost against evaluation granularity. Early stopping criteria — the agent terminates if success rate plateaus for 500 consecutive epochs or if max contact force exceeds a safety threshold."

"Current implementation: PPO via RSL-RL at 512 parallel environments with state observations only. The IL pipeline and GR00T-H integration are not yet implemented — that's Phase 2 work. Today, training starts from a random policy and relies entirely on RL exploration, which requires more environment steps but avoids the teleop data collection bottleneck."

"Strategically, this skill is where compute cost concentrates. A full PPO run at 512 envs for 2000 epochs takes approximately 4–6 hours on an A6000. The agent's ability to reason about hyperparameters — adjusting learning rate after 500 epochs of plateau, switching from PPO to SAC if sample efficiency is poor — directly reduces the number of wasted training runs. In the manual pipeline, an engineer guesses hyperparameters, waits 6 hours, checks results, and re-launches. The agent monitors continuously and adapts mid-run."

---

## SLIDE 17 — Evaluation Skill

**Say:**

"Skill 7 — evaluation — closes the loop. It takes a trained policy checkpoint and produces quantitative metrics that the agent uses to decide: ship it, iterate, or change strategy."

"The evaluation protocol is simulation-in-the-loop — SIL. The policy runs in the full physics simulation with the full rendering stack, not a simplified proxy. 100 evaluation episodes, each with a randomized start configuration and target position. The agent collects six metrics."

"Success rate: the percentage of episodes where the catheter tip reaches within epsilon of the target. This is the headline number. Anything below 80% for a well-defined anatomy means the policy needs more training or the reward function needs adjustment."

"Mean navigation time: average seconds from episode start to target reached. Clinical reference is 30–90 seconds for a straightforward ICA-to-MCA navigation. Faster is better, but not at the expense of safety."

"Max wall contact force: the peak force exerted on any vessel wall during any episode. This is the safety-critical metric. Real vessel perforation risk begins around 0.1–0.3 N depending on vessel caliber and wall condition. If max contact force exceeds the threshold, the policy is unsafe regardless of success rate."

"Fluoroscopy dose: total number of imaging frames consumed across all episodes. In clinical practice, every frame delivers radiation to the patient — the ALARA principle demands minimizing dose. A policy that succeeds but requires 500 frames per episode is clinically unacceptable compared to one that succeeds in 50 frames."

"FID — Frechet Inception Distance: a quantitative measure of the sim-to-real gap. The agent computes FID between the synthetic fluoroscopy frames generated during evaluation and a reference distribution of real clinical fluoroscopy images. FID below 50 indicates visually plausible imagery; above 100 signals a domain gap that will degrade transfer. This metric feeds directly back into the sensor-sim-xray skill — if FID is high, the agent adjusts rendering parameters: increase noise, add scatter, modify gamma."

"Registration accuracy: the trained model is tested on real DSA images — not simulated — to measure the actual sim-to-real transfer gap. This is the ultimate validation metric, but it requires a held-out set of clinical images with ground-truth annotations."

"The agent's behavior after evaluation is what distinguishes this from a metrics dashboard. The agent reasons about failure patterns. It clusters failed episodes by anatomy region — bifurcation failures, tortuous segment failures, distal branch failures. It correlates failures with physics parameters — did high-stiffness episodes fail more? Did certain C-arm angles produce worse visibility? Then it proposes specific remediation: 'Add 20 teleop demos at the ICA-MCA bifurcation. Increase domain randomization range for C-arm angle from plus-minus 15 to plus-minus 30. Switch to curriculum learning with bifurcation-first staging.'"

"This iterative refinement loop — evaluate, diagnose, prescribe, re-run — is the core value proposition of the agentic workflow. A human engineer does this manually over days. The agent does it in minutes between training runs, and retains the full history of what was tried and why."

---

## SLIDE 18 — Agent Interaction Model

**Say:**

"Let me walk through a concrete agent session to make this tangible."

"The developer says: 'Train a catheter navigation policy for cerebral ICA to MCA. Patient CTA number 42, Philips Azurion C-arm, DSA mode, 50 teleop demos.'"

"The agent runs patient-digital-twin: loads the CTA, maps HU to mu, extracts a VMTK centerline with 5,243 nodes, computes a Dijkstra arrival map from the ICA root, and selects left-hemisphere-only injection."

"Then catheter-physics-sim: configures Newton with stiffness 2.5 N/m and friction 0.3, builds the vessel SDF, applies Beer-Lambert compositing with the per-segment mu profile — tungsten markers at 3.0, nitinol shaft at 0.8, platinum tip at 5.0 — and collects 50 teleop demos averaging 45 seconds each."

"Then sensor-sim-xray: selects the Philips Azurion 7 C-arm preset — SDD 1240, SID 780, 2480-by-1920 detector — renders via the Slang fused path at 25 FPS with DSA mode and domain randomization enabled. Detector physics: Poisson noise at 2000 photons per pixel, veiling glare at sigma 18, PSF at sigma 0.7. Output: 10,000 paired frames across 200 episodes."

"Then dataset-creation packages as HDF5 with 80/10/10 splits — 2.3 gigabytes."

"Now policy-training — and this is where I want to slow down, because this is the most compute-intensive skill and the one where the agent's reasoning has the highest leverage."

"The agent configures a two-phase training run. Phase one: imitation learning from the 50 teleop demos via GR00T-H behavioral cloning, 100 epochs. The agent chose 100 epochs — not 50, not 200 — because it has learned from prior runs in its persistent memory that 50 demos at 45 seconds each, at 30 Hz control frequency, yield approximately 67,500 state-action pairs. At that dataset size, the loss curve typically plateaus around epoch 80–100. Going beyond 100 epochs risks overfitting to the demonstrator's idiosyncrasies — the exact trajectories rather than the navigation strategy."

"Phase one produces a warm-start policy. It knows the general approach: advance the catheter, follow the vessel, steer at branches. But it hasn't learned precision. Success rate at this stage is typically 50–60% — the policy reaches the target in easy anatomies but fails at tight bifurcations and tortuous segments."

"Phase two: RL fine-tuning with PPO for 2000 epochs. The agent selected PPO over SAC because the 512-environment setup provides enough parallel rollouts for on-policy learning — PPO's higher sample consumption is offset by the massive batch size. The agent sets an initial learning rate of 3e-4 with linear decay to 1e-5 over the 2000 epochs. Clipping epsilon starts at 0.2 and anneals to 0.1 to tighten the trust region as the policy improves."

"During training, the agent monitors three signals every 100 epochs. First: mean episode reward — is it still improving? Second: success rate on a held-out evaluation set — 20 episodes, separate from training. Third: max contact force — is the policy learning to be gentle, or is it brute-forcing through vessel walls? If success rate plateaus for 400 consecutive epochs, the agent halves the learning rate. If max contact force trends upward despite improving success rate, the agent increases the wall-contact penalty weight from 0.5 to 0.8 and resumes."

"At epoch 1600, the agent observes that the held-out success rate has peaked at 73% and the reward curve has flattened. It saves this as the best checkpoint and lets the remaining 400 epochs run as confirmation — success rate doesn't improve further, confirming epoch 1600 as optimal. Total training wall-clock time: approximately 5 hours on the A6000."

"The agent reports: 'Training complete. Best checkpoint: epoch 1600. IL phase converged at epoch 87. RL fine-tuning improved success rate from 58% (post-IL) to 73% (epoch 1600). Training terminated at epoch 2000 — no further improvement in final 400 epochs. Max contact force: 0.08 N, within safety threshold of 0.1 N.'"

"Then evaluation runs 100 SIL episodes against the epoch-1600 checkpoint. Full physics, full rendering, randomized anatomy placement and target position. Results: 73% success rate, 38-second mean navigation time, 0.08 N max contact force, average 85 fluoroscopy frames per episode."

"Here's where the agent shows its value. It doesn't just report numbers — it reasons about them. It clusters the 27 failed episodes by anatomy region and identifies that 19 of 27 failures — 70% — occur at the ICA-MCA bifurcation. The policy overshoots the branch point and enters the ACA instead of the MCA. The remaining 8 failures are scattered across tortuous M1 segments."

"The agent's diagnosis: the IL demos don't contain enough bifurcation-specific examples. The 50 teleop demos traverse the bifurcation successfully each time, but the RL exploration phase encounters the bifurcation from angles the demonstrator never showed. The policy lacks a robust bifurcation strategy."

"The agent's prescription: add a curriculum stage. Re-run policy-training with a modified config — first 500 epochs restrict the start position to 10 mm proximal to the bifurcation, forcing the policy to practice the critical decision point. Then expand to full-length navigation for the remaining 1500 epochs. Also: collect 20 additional teleop demos specifically targeting the bifurcation approach from varied angles."

"The agent re-runs the pipeline with these changes. Second cycle result: success rate 84%. Bifurcation failures drop from 19 to 6. The agent reports: 'Remaining failures are distal MCA — suggest collecting 20 more teleop demos targeting distal branches for the next cycle.'"

"That entire loop — from CTA to trained policy to failure analysis to curriculum redesign to re-training — took hours, not weeks. And the agent retains the full context: it knows what was tried, why it was tried, what worked, and what to try next."

---

## SLIDE 19 — Missing Features: Blockers vs Closed

**Say:**

"Let me be direct about what's done and what's blocking the agent."

"Five critical capabilities were implemented this sprint and are no longer blockers. Beer-Lambert catheter compositing on both CPU and GPU. Per-segment attenuation profile matching real device construction. Detector physics chain — Poisson, scatter, PSF. C-arm projection model with cone-beam magnification. And state-based RL training with 512 parallel environments."

"Fourteen features remain. The highest-impact blockers are these four:"

"One: vessel mask input. Without it, the agent cannot configure DSA, vessel boost, or selective injection. This is a data pipeline issue, not a rendering issue."

"Two: the DSA pipeline. DSA is the primary imaging mode for catheter navigation in clinical practice. Training on DRR alone will produce policies that fail on real DSA imagery. This is a three-day implementation once vessel mask is available."

"Three: multi-env fluoroscopy. Current rendering is single-env. PPO-class training at 512 environments requires either batched Slang dispatch or 512 parallel CPU compositors. This is the Sprint 2 target and the longest pole."

"Four: image-based RL observations. The current environment is state-only. For image-guided navigation — which is the clinical reality, where the interventionalist sees nothing but the fluoroscopy screen — we need fluoroscopy pixel observations in the environment's observation dict. This depends on multi-env rendering."

"The remaining ten are realism and polish: gamma correction, physics-based scatter, beam hardening, misregistration jitter, C-arm vendor presets, bolus tracking, per-frame mu updates, max-attenuation volume compositing, and FID-based realism metrics. Important for sim-to-real transfer, but not blocking the first training run."

---

## SLIDE 20 — Implementation Roadmap

**Say:**

"Three phases, dependency-ordered."

"Phase 1: Simulation Fidelity. Weeks 1 through 3. The core compositing — Beer-Lambert, detector physics, attenuation profile, Slang fused renderer — is done. That was roughly 7 days of the planned effort. Remaining items: vessel mask plus vessel boost plus DSA pipeline at 3 days. Gamma correction, scatter convolution, and jitter at 1 day. C-arm vendor presets at half a day. Bolus tracking stages 1 and 2 at roughly 1.5 weeks. Selective injection and realism metrics at 3 days. Total remaining Phase 1 effort: approximately 2.5 weeks."

"Phase 2: Skill Packaging. Weeks 3 through 5. Wrap each stage as an OpenClaw skill with a definition file, entry points, config templates, and example inputs/outputs. Seven skills, approximately 2 weeks."

"Phase 3: Agent Integration. Weeks 5 through 7. Skill discovery and dependency resolution. Natural language to YAML config mapping. The iterative refinement loop — agent runs evaluation, reasons about metrics, proposes changes, re-runs. And integration into Slack or IDE for developer interaction."

"The critical-path insight: Phase 1 core compositing is complete. The remaining Phase 1 items are engineering work with no research risk. The long pole is multi-env fluoroscopy in Sprint 2, which Phase 2 and Phase 3 can partially overlap with."

---

## SLIDE 21 — What This Enables for XCATH

**Say:**

"XCATH's current workflow is entirely manual. An engineer runs VMTK in a conda environment to segment vessels and compute arrival maps. Manually configures C-arm geometry and DSA parameters. Manually renders 150 frames and inspects them visually. Manually exports paired data and manages file splits. Manually trains PoseNet and evaluates on a held-out set. Each experiment cycle takes 2 to 3 weeks."

"With the agentic workflow, each of those manual steps maps to a skill the agent executes autonomously. The developer describes intent in natural language — 'use Philips Azurion, clinical DSA' — and the agent translates that to config, runs the pipeline, evaluates the output, and iterates."

"Experiment cycle time compresses from weeks to hours. And the agent runs continuously — it can execute overnight, analyze in the morning, and have updated results ready before standup."

"The critical path to enabling this: Phase 1 core compositing is done. Remaining rendering features are 2.5 weeks of engineering. Skill packaging is 2 weeks. Agent integration is 2 weeks. Total: 7 weeks from today to a working agent-driven pipeline."

---

## SLIDE 22 — PRD Requirements Mapping

**Say:**

"Mapping to the PRD requirements."

"RQ-02-1, reference workflow for catheter navigation: the 4-stage pipeline — patient twin, physics sim, sensor sim, policy training — is implemented. Core pipeline functional."

"RQ-02-2, simulation and asset stack: partially complete. Beer-Lambert compositing and catheter physics are done. DSA, vessel boost, bolus dynamics, beam hardening, and C-arm vendor presets are the remaining items."

"RQ-02-3, OpenClaw-enabled agentic co-development: skill packaging has not started. The seven skills are designed, the interfaces are defined, but the actual skill definition files and entry points need to be written. This is Phase 2."

"RQ-06, unified sensor simulation API: the renderer exists and works. The skill wrapper that exposes it through the OpenClaw interface is pending."

"For the LHAs — XCATH and Remedy — the skills need to be configured with XCATH's validated parameters: k equals 20, gamma 0.8, reference velocity 150 mm/s, and specific Philips and GE C-arm presets. This requires the preset registry, which is a Phase 1 remaining item."

---

## SLIDE 23 — Future Sensors

**Say:**

"Five additional sensors complete the XCath simulation environment."

"Force/torque is the highest priority and lowest effort. Contact force data already exists in the XPBD collision solver — it just needs to be exposed as an observation. This unblocks the wall-contact and tip-force penalties in the reward function."

"Endoscopy and RGB are medium priority. Isaac Lab already has a camera sensor — the work is creating realistic vessel interior USD assets for intravascular views."

"IVUS — intravascular ultrasound — adapts the existing Slang ultrasound renderer to a higher-resolution, shorter-range regime with detailed vessel wall models."

"CBCT — cone-beam CT — requires batched DRR rendering from multiple angles plus FDK reconstruction. Moderate effort, depends on the Slang ray-caster we're already building."

"Pressure and flow are low priority, high effort. They require a hemodynamic solver — 1D fluid equations on the vessel centerline graph — which is a new capability."

"The architecture is designed to be extensible. Each sensor follows the same pattern: a Slang or Warp rendering kernel, a Python wrapper exposing a unified API, an OpenClaw skill definition, and integration into the observation dict."

---

## CLOSING — Key Takeaways for the Room

**Say (if asked to summarize):**

"Three takeaways."

"One: the hard rendering problem is solved. We have physically correct Beer-Lambert catheter compositing running on GPU in a single fused ray march at 25 FPS. This is a capability XCATH does not have. The remaining rendering gaps — DSA, vessel boost, bolus — are engineering, not research."

"Two: the RL pipeline is functional. State-based PPO training runs today at 512 parallel environments. The path to image-based RL requires multi-env fluoroscopy rendering, which is Sprint 2."

"Three: the agentic workflow compresses experiment cycles from weeks to hours. Seven skills, four pipeline stages, an iterative refinement loop. The critical path is 7 weeks: 2.5 weeks of remaining rendering features, 2 weeks of skill packaging, 2 weeks of agent integration."

"The ask: prioritize the vessel mask pipeline and multi-env rendering. Those are the two items that unblock everything downstream — DSA, image-based RL, and the full agentic loop."
