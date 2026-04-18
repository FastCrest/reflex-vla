# Inference Infrastructure Options for Reflex VLA (2026)

**Date:** 2026-04-16
**Author:** Research pass (agent-assisted enumeration)
**Status:** Draft — comprehensive landscape, not shortlist

## Context

Reflex is a VLA deployment CLI running benchmarks + user-facing inference on remote GPUs. We currently use Modal. The painful packaging problem:

- **LIBERO** (2023 research code, Lifelong-Robot-Learning/LIBERO) needs `mujoco-py`, older `gym`, Python 3.8-ish, and specific CUDA/cuDNN.
- **lerobot** (2026, HuggingFace) needs modern `transformers>=4.57`, `torch>=2.4`, Python 3.10+, `mujoco>=3.x` (the `dm_control` version, not mujoco-py).
- They **refuse to coexist** in one Python env. The HuggingFace port `huggingface/lerobot-libero` papers over *some* of this but issues #2114 and #2697 confirm ongoing version-mismatch pain. The robosuite version that LIBERO pins fights the mujoco version lerobot wants; transformer versions diverge; numpy pins fight.

**The operational question:** which infra makes "run LIBERO in one env, pass observations to lerobot in another env, get actions back" *cheap to build* and *cheap to run*?

This doc enumerates every serverless GPU / cloud-GPU / edge-GPU option a VLA company might use in 2026, evaluates package-isolation patterns, and ranks by fit-for-reflex.

---

# PART 1: Serverless GPU (pay-per-request, ephemeral)

## Modal

### What it is
Python-native serverless GPU. You define an `app.function()`, decorate with GPU type, and Modal spins up containers on demand. Strongest DX in the category; the team built a custom container runtime (not Docker) but supports `Image.from_dockerfile()`, `Image.debian_slim().pip_install()`, `Image.micromamba()`, etc. Multiple images per app — each function can pin a different image/Python version.

### Pricing
- H100: $3.95/hr base
- A100 80GB: $2.10/hr base
- T4: $0.59/hr base
- **Multipliers:** 1.25x region multiplier (US/EU/APAC) → up to 3.75x for non-preemptible workloads in some regions. Real effective rate on preemptible A100: ~$2.50/hr; on-demand: ~$4.80/hr in the worst case.
- Free tier: $30/month compute credits
- Per-second billing, no commitment

### Docker / env flexibility
- `Image.from_dockerfile(path)` — full Dockerfile support
- `Image.debian_slim().pip_install(...)` — declarative, layer-cached
- `Image.micromamba()` — conda-based for tricky ML deps
- **Critical for reflex:** different `@app.function()` calls can use different images. Function A loads LIBERO image, Function B loads lerobot image. They call each other via `.remote()` or `.spawn()`.
- Shared state via `modal.Dict`, `modal.Queue`, `modal.Volume` (persistent, POSIX-ish), `modal.NetworkFileSystem` (deprecated, prefer Volume).
- Tunnels + WebSockets supported for low-latency streaming between external client and a Modal container.

### Jetson / edge story
- **No native Jetson support.** Modal is A100/H100/T4/L4/L40S/B200 in cloud datacenters (Ashburn, Paris, etc.). You can `modal run` benchmarks but cannot deploy the resulting engine to a customer's Jetson through Modal.
- Workaround: Modal can host the `reflex check` validator and email/Slack the TRT engine to the customer, who runs `reflex serve` locally on Jetson.

### Cold-start
- Warm: tens of ms
- Cold (with large image): 5-30s depending on image size and `enable_memory_snapshot` usage
- Memory snapshot feature: ~500ms cold starts for models that fit the snapshot. Excellent for ~1B param VLAs but SmolVLA + lerobot runtime may exceed snapshot budget.

### Fit for reflex
- **Daily dev:** 10/10. Python-first, `modal run` feels like local. Already integrated.
- **Benchmarking:** 9/10. `modal.Volume` for model weights, per-function images for LIBERO vs lerobot, `.map()` for parallel rollouts.
- **Production edge:** 2/10. No Jetson.

### Tier: S for dev/benchmarking, D for edge
Keep Modal as the daily driver. The per-function-image model is the clearest win for LIBERO vs lerobot separation (see "Most-likely-wins" section below).

---

## Replicate

### What it is
Model-as-API hosting. You push a `cog.yaml` + Python predictor, Replicate wraps it in a REST endpoint. Originally built for Stable Diffusion; strong public model library; image generation focus.

### Pricing
- CPU: $0.000115/sec = $0.414/hr
- T4: $0.000225/sec = $0.81/hr
- A100 80GB: $0.001400/sec = $5.04/hr (significantly more expensive than Modal/RunPod)
- Per-second billing

### Docker / env flexibility
- Cog is a Docker wrapper — you can `build` a custom image, but Replicate bakes the output into their inference layer. Not raw Docker.
- Limited to one Python env per model; two-env isolation requires deploying two models and chaining REST calls between them (high overhead).

### Jetson / edge story
None. Cloud only.

### Cold-start
- Popular pre-cached models: instant
- Custom model deployments: **60+ seconds** cold start (well-documented weakness)
- Mitigated by keeping models warm with min-instances (costs $$)

### Fit for reflex
- **Daily dev:** 4/10. Cog is opinionated; doesn't match reflex's multi-env needs.
- **Production edge:** 1/10.

### Tier: C
Good if you want to publish a SmolVLA demo as a public Replicate model. Bad for internal dev or dep conflicts.

---

## RunPod Serverless

### What it is
Two products under one brand: "Pods" (rent a container for N hours) and "Serverless" (event-driven, scale-to-zero). Serverless workers are Docker containers you build and push; RunPod runs them behind a queue.

### Pricing
- Serverless A100 80GB: ~$0.0004/sec = $1.44/hr active (significantly cheaper than Modal)
- Serverless H100: $0.00116/sec = $4.18/hr
- Serverless T4: $0.00023/sec = $0.83/hr
- No cold-start charge (only active execution)
- "FlashBoot" caches Docker layers — 48% of cold starts <200ms; large containers 6-12s

### Docker / env flexibility
- **Bring your own Dockerfile.** Fullest Docker support of any serverless GPU provider. Multi-stage builds work. You push to Docker Hub / ECR / their registry, point RunPod at it.
- Single container = single env, but you can run two containers (LIBERO worker + lerobot worker) as separate "endpoints" and coordinate via their queue.
- No Modal-equivalent Dict/Queue primitives — you're on your own for shared state (use Redis / S3).

### Jetson / edge story
- No native Jetson rental in Serverless.
- Pods product has community-hosted machines; a few Jetson Orins appear intermittently but not reliably.

### Cold-start
- FlashBoot: 48% <200ms
- Large containers: 6-12s
- "Active workers" cost $$ but stay warm

### Fit for reflex
- **Daily dev:** 7/10. Costs ~40% less than Modal for same GPU; Docker-first fits the "two containers" pattern well.
- **Benchmarking:** 8/10. Cheap, and the Docker-first model lets you pin LIBERO's old CUDA.
- **Production edge:** 3/10. No Jetson.

### Tier: A for benchmarking, B for dev
Strong Modal alternative if you want more Docker control and ~40% cost savings.

---

## Baseten

### What it is
Truss-based model deployment. Production-leaning: emphasis on SLAs, autoscaling, enterprise features. Strong vLLM / SGLang / TRT-LLM integration (fastest LLM serving in the category).

### Pricing
- T4: $0.63/hr
- A10G: $1.21/hr
- A100 80GB: $4.00/hr
- H100: $6.50/hr
- B200: $9.98/hr
- Per-minute billing, no idle cost

### Docker / env flexibility
- Truss is a light Python wrapper; `truss init`, `truss deploy`. Can bring custom Docker via `base_image` + `docker_server` argument.
- Supports vLLM/SGLang/TRT-LLM/Triton as first-class deploys.
- Two-env isolation is awkward — you'd deploy two Truss models and chain REST calls.

### Jetson / edge story
None. Enterprise edge is on their roadmap per marketing but no product.

### Cold-start
- "Blazing fast" per marketing; empirically 3-8s for small models, longer for LLMs.
- Enterprise plan gets warm pools.

### Fit for reflex
- **Daily dev:** 5/10. Overkill for a solo founder; excellent once you need 99.9% SLAs.
- **Production edge:** 1/10. No Jetson.

### Tier: B for production LLM serving, D for reflex's current scale
Revisit when reflex has real customer traffic.

---

## Together AI Inference

### What it is
Primarily LLM inference provider (serverless pay-per-token) and dedicated endpoints. Custom model hosting available via Dedicated Endpoints.

### Pricing
- Serverless (per-token, not per-GPU): Llama 4 Maverick $0.27/$0.85 per 1M tokens input/output
- Dedicated endpoints: H100 $5.50/hr, up to 43% off their older rates (announced 2026)

### Docker / env flexibility
- Dedicated endpoints accept HuggingFace models. **No custom Docker.** This is a token-in, token-out service for LLMs.
- Not applicable for VLA inference with action heads.

### Jetson / edge story
None.

### Cold-start
N/A (always-warm).

### Fit for reflex
- **Daily dev:** 1/10. Can't deploy SmolVLA as-is.
- **Production edge:** 1/10.

### Tier: F
Not the right shape for Reflex. Good for LLM-only adjacent products.

---

## Cerebrium

### What it is
Python-first serverless GPU. Similar pitch to Modal — define a Python function, deploy. 12+ GPU types, infrastructure-as-code, pay-per-second.

### Pricing
- Not publicly posted (contact sales) — competitive with Modal per industry consensus.
- Claim "up to 50% cheaper than cold-start-heavy alternatives."

### Docker / env flexibility
- Supports custom Dockerfiles via entry point override
- Supports any Python framework (transformers, torch, TF, sklearn, custom)

### Jetson / edge story
None.

### Cold-start
- 2-5s cold start (claim)
- No idle billing

### Fit for reflex
- **Daily dev:** 6/10. Similar DX to Modal but less battle-tested, smaller community.
- **Production edge:** 1/10.

### Tier: B
Reasonable Modal alternative if pricing shakes out cheaper, but the ecosystem is thinner.

---

## fal.ai

### What it is
Image/video generation–focused serverless GPU. Optimized for diffusion models. Fastest cold start on SDXL-class workloads in the category.

### Pricing
- Contact for H100; competitive per-second rates
- Free credits for new accounts

### Docker / env flexibility
- Custom models supported via their "Serverless" product
- Diffusion-optimized runtime may not help VLAs (action heads aren't their use case)
- Custom Docker images: **60+ second cold starts** (per independent benchmarks)

### Jetson / edge story
None.

### Cold-start
- SDXL-class: ~2-5s (their specialty)
- Generic custom models: 60+s

### Fit for reflex
- **Daily dev:** 3/10. Wrong audience.
- **Production edge:** 1/10.

### Tier: D
Wrong specialty for reflex. Good if you add a video-gen feature to the product.

---

## Beam Cloud

### What it is
Serverless GPU with custom container runtime (`beta9`, open-source) that lazy-loads image layers. Pitches sub-1-second cold starts as the headline feature.

### Pricing
- Free tier: 10 hours GPU
- CPU: $0.190/core, RAM $0.020/GB, storage free
- GPU rates not publicly posted but claimed competitive
- No charge for cold starts / spin-up

### Docker / env flexibility
- Python-native SDK (like Modal)
- Custom container runtime — not raw Docker, but reads Dockerfile-style specs
- Lazy layer loading means cold starts on large images (e.g., 10GB lerobot + deps) stay fast

### Jetson / edge story
None.

### Cold-start
- Cold: 2-3s claimed, sub-1s for some workloads
- Warm: 50ms

### Fit for reflex
- **Daily dev:** 7/10. Legitimately interesting cold-start story. Python SDK mirrors Modal.
- **Production edge:** 1/10.

### Tier: B
Worth a weekend bake-off vs Modal if cold starts become the bottleneck on `reflex bench`.

---

## Mystic.ai

### What it is
"Deploy ML in your own AWS/GCP/Azure account, or our shared cluster." Bring-your-own-cloud orchestration + hosted option. Pipeline Core platform.

### Pricing
- Not publicly listed; contact sales
- BYO-cloud means you pay cloud provider GPU rates + Mystic management fee

### Docker / env flexibility
- Versioning + environment management
- Cross-cloud auto-scaling
- Custom containers supported

### Jetson / edge story
None.

### Cold-start
Not publicly benchmarked.

### Fit for reflex
- **Daily dev:** 4/10. Heavyweight for a solo founder.
- **Production edge:** 2/10. BYO-cloud could extend to a Jetson via Tailscale-wrapped nodes, but nothing native.

### Tier: C
Good at enterprise scale; overkill for reflex today.

---

## Lightning AI (Lightning Studios + deployments)

### What it is
AI Studios = persistent GPU workspaces (notebook-like, feels like your laptop). Lightning deploy = serverless deployment of Studios as endpoints.

### Pricing
- Free: $0/mo, 15 credits, 1 Studio (4-hr restarts), 100GB storage
- Pro: $50/mo (annual) or $600/mo (monthly), 40 credits, 24/7 Studio, multi-GPU, 2TB
- Teams: $140/user/mo (annual), H100/H200 GPU access, unlimited storage
- GPU rates: T4 $0.68/hr, L4 $0.70/hr, A10G $1.80/hr (with credits included in plans)

### Docker / env flexibility
- Studios are persistent VMs — you have full root, can install conda/uv/whatever
- Deployment layer converts Studios to serverless endpoints
- Custom Dockerfile support in deployment layer

### Jetson / edge story
None.

### Cold-start
- Studio "restart": 30-60s
- Deployment cold start: comparable to Modal

### Fit for reflex
- **Daily dev:** 7/10. The "persistent workspace" model is nice for iterative debugging; the Studio-to-endpoint promotion is smooth.
- **Production edge:** 1/10.

### Tier: B
Good fit if you want a "notebook + deploy" workflow; less ideal for CLI-first like reflex.

---

# PART 2: GPU Rental / Reserved (Explicit Docker)

## RunPod On-Demand Pods

### What it is
Rent a GPU container by the hour. Full SSH access, full root, any Docker image. "Secure Cloud" = tier-3 DC, "Community Cloud" = hosted on home labs (cheaper, variable reliability).

### Pricing
- RTX 4090: $0.34-0.69/hr (community) — $0.79/hr secure
- A100 80GB: $1.89/hr (community) — $2.17/hr secure
- H100 SXM: $3.49/hr secure
- Per-minute billing on pods

### Docker / env flexibility
- Bring your own Docker image. Explicit `docker run` under the hood.
- Multi-stage builds, multiple venvs, any Python version — completely flexible.
- Persistent volumes available.

### Jetson / edge story
- Some community pods advertise Jetson Orin 64GB but supply is unreliable (bursty).

### Cold-start
- Container start: 30s-2min (pulling image, allocating GPU)

### Fit for reflex
- **Daily dev:** 8/10. Cheapest cleaner than serverless for benchmark sweeps.
- **Benchmarking:** 9/10. Perfect for "spin up pod, run 50 LIBERO tasks, teardown."
- **Production edge:** 3/10. Unreliable Jetson supply.

### Tier: A for benchmarking
The "two containers, one pod" pattern (LIBERO container + lerobot container, talking via localhost) is trivially easy here. See part 3.

---

## Lambda Labs

### What it is
Pure GPU rental, 1-hour minimum. "ML-first" cloud — pre-installed CUDA, PyTorch, Jupyter. Reserved instances 15-30% off.

### Pricing
- H100: $2.89/hr (cheapest on-demand H100 in the market)
- A100 80GB: $1.29/hr
- A10G: $0.75/hr
- RTX 6000 Ada: $1.10/hr
- 1-hour minimum, hourly billing after

### Docker / env flexibility
- Standard Linux VM with NVIDIA toolkit
- Full Docker support (bring your own image)
- Multi-stage builds, any venv setup

### Jetson / edge story
None.

### Cold-start
- Instance launch: <60s

### Fit for reflex
- **Daily dev:** 7/10. Cheapest H100 in the market if you need raw training; less ideal for ephemeral inference.
- **Benchmarking:** 9/10. Cheapest H100 on demand; reserved instances make training affordable.
- **Production edge:** 1/10.

### Tier: A for training, B for inference benchmarks
If reflex ever needs to train / distill, this is the cheapest H100 option.

---

## Vast.ai

### What it is
P2P GPU marketplace. Hosts are individuals renting out home-lab GPUs; Vast is the broker. Prices set by supply/demand across 40+ "data centers" (read: garages + colos).

### Pricing
- RTX 4090: $0.29-0.59/hr (5-10x cheaper than AWS)
- A100 80GB: $2.50+/hr
- H100 80GB: $3.20+/hr
- On-demand / interruptible / reserved

### Docker / env flexibility
- All instances are Docker. Select any public/private image.
- Full root, full flexibility.

### Jetson / edge story
- Mostly gaming GPUs and data center GPUs. No reliable Jetson supply.

### Cold-start
- Depends on host; 1-5 minutes typical.

### Fit for reflex
- **Daily dev:** 5/10. Cheap but reliability varies; not ideal for CI.
- **Benchmarking:** 7/10. Fantastic for "burn $5 on a 4090 for 8 hours of rollouts."
- **Production edge:** 1/10.

### Tier: B for one-off experiments
Use when you don't care about SLA. Never put customer traffic on Vast.

---

## CoreWeave

### What it is
Kubernetes-native GPU cloud. Enterprise-tier. Purpose-built DCs. Managed Kubernetes Service (CKS) with H100/H200/B200 availability.

### Pricing
- H100: $4.76+/hr on-demand
- A100: $2.21+/hr
- 8x H100 HGX: $49.24/hr
- Up to 60% off with reserved commitments
- No egress charges

### Docker / env flexibility
- Full Kubernetes — any container, any image
- Helm charts, NVIDIA GPU operator, K8s-native

### Jetson / edge story
None (data center only).

### Cold-start
- K8s pod scheduling: 30s-2min depending on image pull

### Fit for reflex
- **Daily dev:** 3/10. K8s overhead is heavy for a solo founder.
- **Benchmarking:** 7/10 if you already have K8s expertise.
- **Production edge:** 2/10.

### Tier: C for solo; A at Series B scale
Revisit once reflex has a customer ops team.

---

## Crusoe

### What it is
GPU cloud with focus on low-carbon (flare-gas-powered DCs) + AI-native infra. Strong H100/H200/B200/MI300X availability.

### Pricing
- H100: $2.74/hr
- H200: $3.14/hr
- A100 80GB: $1.76/hr
- A100 40GB: $1.42/hr
- B200: $5.87/hr
- MI300X: $3.45/hr
- Reserved + spot + on-demand

### Docker / env flexibility
- Standard cloud VM
- Full Docker support

### Jetson / edge story
None.

### Cold-start
- VM launch: ~2 min

### Fit for reflex
- **Daily dev:** 6/10. Good H100/H200 pricing if you want to train from scratch.
- **Production edge:** 1/10.

### Tier: B for training
Competitive with Lambda; better for multi-node training.

---

## Paperspace (DigitalOcean)

### What it is
Acquired by DigitalOcean. Gradient = notebook/ML platform; Core VMs = raw GPU rental.

### Pricing
- A100 on-demand: $3.18/hr (expensive)
- H100: $5.95/hr
- Reserved A100: $1.15/hr (36-month commit)
- Growth plan $39/mo to access many GPU tiers

### Docker / env flexibility
- Core VMs: full root, full Docker, persistent storage
- Gradient: notebook UX, restricted

### Jetson / edge story
None.

### Cold-start
- VM launch: 1-2 min

### Fit for reflex
- **Daily dev:** 5/10. Expensive vs Lambda/Runpod for same GPU.
- **Production edge:** 1/10.

### Tier: C
Price-performance has been outpaced by Lambda/Runpod. Skip.

---

## Nebius

### What it is
Former Yandex Cloud spinoff. NVIDIA-partnered (2026 $2B investment). Purpose-built AI cloud, H200/GB200/B300/B200/H100.

### Pricing
- H100: $2.00/hr area (negotiated; per-GPU on 8-GPU presets around $0.90/hr)
- NIM microservice 8-GPU: $0.90/GPU/hr = $7.20/hr total
- Up to 35% off with commitments

### Docker / env flexibility
- Full VM, full Docker
- Kubernetes available

### Jetson / edge story
None.

### Cold-start
- Standard cloud VM timing

### Fit for reflex
- **Daily dev:** 6/10. Competitive H100 pricing, credible supply.
- **Production edge:** 1/10.

### Tier: B
Worth an eye if you need a big H100 cluster for training π0-sized models from scratch.

---

## Hyperbolic

### What it is
"Open-access" AI cloud. Marketplace + serverless inference + GPU rental. Positions against RunPod on price.

### Pricing
- RTX 4090: $0.50/GPU/hr
- A100: $1.80/GPU/hr
- H100 PCIe: $3.00/hr
- H100 SXM: $3.20/hr
- Serverless inference: $0.40-$4.00 per 1M tokens (Llama 70B / 405B tier)

### Docker / env flexibility
- Full Docker on rental pods
- OpenAI-compatible API on serverless side

### Jetson / edge story
None.

### Cold-start
- Container boot: 1-2 min

### Fit for reflex
- **Daily dev:** 6/10. Cheaper than RunPod marginally; smaller ecosystem.
- **Production edge:** 1/10.

### Tier: B
Alt-RunPod; not worth switching unless price-sensitive.

---

## SkyPilot (meta-orchestrator, not a provider)

### What it is
OSS CLI that runs ML jobs across 20+ clouds (AWS / GCP / Azure / Lambda / RunPod / Vast / Nebius / CoreWeave / on-prem / K8s / Slurm). Writes one YAML, picks cheapest GPU that's available.

### Pricing
- Free / OSS — pays cloud providers' underlying prices
- Discovers cheapest spot H100 across all providers ($16-98/hr range)

### Docker / env flexibility
- Pyxis/enroot containerization
- Works with any Docker image
- Abstracts cloud differences

### Jetson / edge story
None (so far).

### Cold-start
- Adds 10-30s overhead on top of underlying provider

### Fit for reflex
- **Daily dev:** 6/10. Useful if you're juggling many providers; overhead for solo.
- **Benchmarking:** 8/10. Fantastic for "run LIBERO bench on whoever's cheapest right now."

### Tier: A for cost-chasing
If reflex grows and you care about $/benchmark-run, SkyPilot is the unlock.

---

# PART 3: Edge / Jetson Rental

## CloudJetson.com

### What it is
Hourly cloud access to real NVIDIA Jetson Orin (Nano / NX / AGX) and Jetson Thor hardware. Specifically built for robotics, AV, edge-AI teams who need on-target validation without shipping a kit.

### Pricing
- Not publicly listed in web search; contact for quote
- Industry rumor: $0.50-1.50/hr for Orin Nano class; $2-4/hr for AGX Orin 64GB; Thor tier premium

### Docker / env flexibility
- Real hardware — full root, JetPack installed
- Custom Docker images fine
- JetPack SDK baked in (JetPack 6.x / JetPack 7 rolling out per NVIDIA 2026 announcements)

### Jetson / edge story
- **This IS the Jetson story.** Only provider that actually rents the target silicon.
- Perfect for CI: build TRT engine on Jetson, validate latency, tear down.

### Cold-start
- Board provisioning: 2-5 min (real flash / reset cycle)

### Fit for reflex
- **Daily dev:** 3/10 (overkill day-to-day; you can validate on Modal first).
- **Production edge:** 10/10. **This is the missing piece for Reflex's "check --on-hardware" command.**

### Tier: S for production edge validation
Integrate into `reflex check --target=jetson-agx` to actually measure real TRT latency / memory pre-ship.

---

## Jetson AI Lab (cloud component)

### What it is
NVIDIA's curated "Jetson AI Lab" — primarily a docs+tutorials+benchmarks site (jetson-ai-lab.com, github.com/NVIDIA-AI-IOT/jetson-ai-lab). Includes a small cloud access layer for tutorials but isn't a production rental service.

### Pricing
- Tutorial tier: free with NVIDIA developer account
- No commercial rental offering

### Docker / env flexibility
- NVIDIA's own tutorial notebooks + Docker images for Orin generation
- Not a BYO environment

### Jetson / edge story
- Reference for Jetson perf benchmarks, quantization recipes (Cosmos Reason2 W4A16 on Orin Nano example), VLA/VLM deployment guides on-device.

### Cold-start
N/A

### Fit for reflex
- **Daily dev:** 5/10 as a reference (not rental)
- **Production edge:** 3/10 (can't rent, only learn)

### Tier: B as a reference, N/A as infra
Bookmark the benchmarks; use CloudJetson for actual Jetson compute.

---

## Luxonis Oak-D / DepthAI HubAI

### What it is
Luxonis OAK cameras have on-device AI accelerators. DepthAI HubAI lets you manage, convert, deploy models to OAK cameras over the cloud. OAK 4 = self-contained perception + cloud management.

### Pricing
- Cameras bought outright ($200-$600 per unit); HubAI cloud management subscription (not public)

### Docker / env flexibility
- Convert models to DepthAI format (MyriadX blob / Myriad RVC4)
- Custom pipelines in Python
- **Not a Jetson substitute** — different accelerator (Movidius VPU family, 4 TOPS for OAK-D)

### Jetson / edge story
- Complementary to Jetson; handles vision/depth, not full-VLA policy.

### Cold-start
- Device boot: seconds

### Fit for reflex
- **Daily dev:** 2/10.
- **Production edge:** 4/10 for the vision front-end of a multi-chip robot (Luxonis camera + Jetson policy compute), but not for VLAs as a whole.

### Tier: C — adjacent, not core
Interesting for a "Luxonis perception → Reflex policy on Jetson" pipeline story but not replacing anything today.

---

## Raspberry Pi cloud / bare-metal Pi rental

### What it is
Various providers (Mythic Beasts, PiCloud, hobby colocation) rent Raspberry Pi 5 / CM5 boards hourly.

### Pricing
- Pi 5 rental: $0.01-$0.05/hr

### Docker / env flexibility
- Full Docker on 64-bit ARM
- No GPU; CPU inference only

### Jetson / edge story
- Not a VLA target. 8GB RAM / no dedicated NN accelerator = can't run SmolVLA at useful speed.
- Relevant only if reflex adds a "Pi + Hailo" product line.

### Cold-start
- Board provisioning: 1-2 min

### Fit for reflex
- **Daily dev:** 1/10.
- **Production edge:** 2/10.

### Tier: D
Ignore for VLA. Revisit if you add a Hailo-8L / Coral Edge TPU SKU.

---

## Self-hosted home lab via Tailscale + LM Link (+ Headscale)

### What it is
You buy a Jetson Orin or a tower with 4090 / 4090 Ti, put it on your home network, Tailscale it, expose the inference endpoint to Reflex customers as a Tailscale magic-DNS hostname.

### Pricing
- $500-$4000 one-time hardware
- ~$0.20/kWh electricity (cheap in long-run for heavy usage)
- Tailscale: free for up to 100 devices

### Docker / env flexibility
- It's your box. Full flexibility.
- **Crucially**: you can run LIBERO in one Docker + lerobot in another Docker, share a UNIX socket or localhost port.

### Jetson / edge story
- If you bought a Jetson, yes. You are the Jetson cloud.
- LM Link (Tailscale + LM Studio, launched Feb 2026) explicitly shows the "point customer's app at remote GPU as if local" pattern — applicable to VLA endpoints.

### Cold-start
- Whatever your box does.

### Fit for reflex
- **Daily dev:** 7/10 for you personally — cheapest long-run if you have the hardware.
- **Production edge:** 4/10 as a dev story; N/A as a customer product (customers won't tailscale into your basement).

### Tier: B as a dev-env hack
Perfect for "test on Jetson before shipping to a customer." Not a product for end users.

---

# PART 4: Package Isolation Patterns (single node, multiple envs)

## `uv` virtual envs

### What
`uv venv lerobot-env` + `uv venv libero-env`. Activate one, run. Swap. Modern (2024+) fast pip replacement with lockfile support and concurrent-group conflict declaration.

### How LIBERO/lerobot helps
- Two venvs side by side; separate `uv.lock` per venv
- Parent process spawns child subprocesses with different `PATH` prepended
- Cross-venv communication: subprocess.Popen + JSON over stdin/stdout, or gRPC, or ZeroMQ
- Declared conflicts via `[tool.uv].conflicts` in pyproject.toml — useful for groups

### Cost
- Minimal. `uv sync` is <10s per venv.
- 2 × disk for site-packages (lerobot alone is ~4GB with torch + CUDA libs)

### Fit
- **Daily dev:** 10/10. Zero infra, all in-repo.
- **Production:** 5/10. Works but fragile; every engineer must recreate.

### Tier: S for solo dev right now
This is probably the cheapest fix for the actual LIBERO-vs-lerobot problem.

---

## conda env + activation scripts

### What
`conda create -n libero python=3.8 ...` + `conda create -n lerobot python=3.10 ...`. Activate one env per subprocess.

### How it helps
Same as uv but slower. Conda is better at non-Python system libs (CUDA, MuJoCo) which matters for LIBERO.

### Cost
- Conda is slow (20-60s per env create)
- Disk: 2 × env

### Fit
- **Daily dev:** 7/10. Proven for research-code.
- **Production:** 5/10.

### Tier: A
If uv chokes on LIBERO's mujoco-py native deps, conda is the fallback.

---

## Docker multi-stage builds

### What
One Dockerfile, multiple `FROM` stages.
```dockerfile
FROM python:3.8 AS libero_stage
RUN pip install libero==1.0.0 mujoco-py==2.1 ...

FROM python:3.10 AS lerobot_stage
RUN pip install lerobot==0.4.0 transformers==4.57 ...

FROM ubuntu:22.04 AS runtime
COPY --from=libero_stage /opt/libero-venv /opt/libero-venv
COPY --from=lerobot_stage /opt/lerobot-venv /opt/lerobot-venv
ENTRYPOINT ["/opt/entrypoint.sh"]
```
The entrypoint picks which venv to activate based on a command-line flag.

### How LIBERO/lerobot helps
- Two fully isolated venvs in one image
- Each spawns as a subprocess with its own `PATH` + `PYTHONPATH`
- Communication: stdin/stdout pipes, UNIX sockets, localhost ports
- Layer-cached builds = fast iteration

### Cost
- Large image (~12GB for lerobot + LIBERO + both CUDA versions if needed)
- Build time: 5-15 min first time, ~1-3 min cached

### Fit
- **Daily dev:** 7/10.
- **Production:** 9/10. **This is the clean answer for a single-node deploy.**

### Tier: S for production reflex-cli shipped to customers
Ship one "reflex-runtime" image that contains both envs internally.

---

## Distroless + entrypoint routing

### What
Google's distroless base images — minimal, no shell. Combine with a tiny Go/Rust router binary that spawns the correct Python env.

### How it helps
- Tiny attack surface
- Forces you to be explicit about env selection
- Great for production

### Cost
- More upfront work; harder to debug
- ~10-20% smaller images

### Fit
- **Daily dev:** 3/10. Pain for iteration.
- **Production:** 9/10.

### Tier: A for late-stage prod hardening
Not a day-1 move; revisit when reflex has enterprise customers.

---

## Nix packaging

### What
`flake.nix` declaring two fully-reproducible Python envs. Nix store allows multiple versions of everything to coexist (content-addressed). `poetry2nix` bridges from `pyproject.toml`.

### How LIBERO/lerobot helps
- Nix is the only tool that actually SOLVES the "two conflicting sets of system libs on one host" problem structurally.
- Nix store: `/nix/store/abc-libero-mujoco/...` and `/nix/store/def-lerobot-mujoco/...` coexist, no LD_LIBRARY_PATH manipulation needed.
- `lib.meta.lowPrio`/`highPrio` for version picking
- `pythonRelaxDepsHook` for forcibly relaxing version pins when researchers over-pin

### Cost
- **Steep** learning curve (maybe 1-2 weeks for a first real flake)
- Nix expressions for CUDA are still tricky; requires `cachix` for reasonable build times
- Your CI must run Nix

### Fit
- **Daily dev:** 3/10 for most people; 9/10 if you already live in Nix
- **Production:** 9/10 — truly reproducible, across Mac/Linux/NixOS/Docker

### Tier: A at team of 3+; D for solo
Solo founder should not adopt Nix for this. Save for post-seed team.

---

## Dev container spec

### What
`.devcontainer/devcontainer.json` — VS Code / Cursor standard for defining a dev environment. `additionalVersions` on the Python feature allows multiple Python installs. Multiple `.devcontainer` dirs support multiple configs.

### How it helps
- Sugars over Docker
- Cursor / VS Code / JetBrains / OpenClaw all understand the spec
- Mono-repo: `.devcontainer/libero/` and `.devcontainer/lerobot/`

### Cost
- Low. Mostly JSON glue.

### Fit
- **Daily dev:** 7/10 (great for new contributors).
- **Production:** 2/10 (not a runtime; dev-only).

### Tier: B
Use for onboarding, not runtime.

---

# PART 5: Multi-Container Orchestration

## Modal chained functions + WebSocket / tunnels

### What
Two `@app.function(image=libero_image)` + `@app.function(image=lerobot_image)`. Either:
- Call one from the other via `.remote()` (adds ~50-200ms per call)
- Open a `modal.tunnel` between them (microseconds on shared hardware, low-single-digit ms cross-region)

WebSocket support on Modal (launched ~2024, matured in 2026) allows long-lived bidirectional streams. Modal Dict / Queue for shared state.

### Cost
- Two container-seconds running concurrently
- Modal's scheduler starts the second container on demand; cold-start each leg

### Fit
- **Daily dev:** 10/10 (same platform as today)
- **Production:** 8/10 (customer can't deploy Modal to Jetson, but for hosted inference it's the cleanest)

### Tier: S for reflex's "run LIBERO + lerobot in separate envs on Modal"
**This is likely the winning pattern today.** See "Most-likely-wins" section.

---

## Docker Compose + Compose Bridge

### What
`docker-compose.yml` with two services:
```yaml
services:
  libero_env:
    image: reflex/libero:latest
    command: python -u /app/libero_server.py
  lerobot_env:
    image: reflex/lerobot:latest
    command: python -u /app/lerobot_server.py
    depends_on: [libero_env]
```
They share a Docker network; `libero_env` talks to `lerobot_env:8000` on a local bridge. Compose Bridge (new-ish 2024 feature) converts Compose to Kubernetes if you scale up.

### Fit
- **Daily dev:** 9/10. Familiar, fast. Runs on Mac/Linux/WSL.
- **Production:** 6/10 (compose-in-prod is usually replaced by K8s).

### Tier: A
Great for local iteration and CI. Compose Bridge makes scaling painless later.

---

## Kubernetes + Helm

### What
Two Pods (or two containers in one Pod with sidecar). Helm chart per service. Service discovery via DNS.

### Fit
- **Daily dev:** 3/10 for solo.
- **Production:** 10/10 at scale.

### Tier: A at Series A scale; D solo

---

## Ray Serve

### What
Ray cluster with per-deployment runtime envs. Each `@serve.deployment` can specify `runtime_env={"pip": [...], "image_uri": "..."}`. Ray 2.54 supports different container images per deployment (with Podman under the hood).

### Constraints
- Runtime-env + container-image cannot be combined cleanly (quirk).
- Ray and Python versions must match host-and-container to the patch.

### Fit
- **Daily dev:** 4/10 (Ray setup is heavy).
- **Production:** 7/10 if you already have Ray for training.

### Tier: B
Worth it if Reflex adopts Ray for distill training.

---

## Modal Volumes + shared state

### What
`modal.Volume.from_name("reflex-shared", create_if_missing=True)` mounted into both LIBERO and lerobot functions at e.g. `/shared`. Write observations + actions as parquet / npz / pickle. Polling or file-watch.

### Fit
- **Daily dev:** 6/10 (polling is ugly but it works)
- **Production:** 5/10 (latency too high for real-time control)

### Tier: B
Good for offline benchmarking pipelines; bad for interactive control.

---

## Seldon Core v2

### What
K8s-native MLOps. MLServer runtime; deploys models as K8s CRDs. Multi-model serving, parallel inference.

### Fit
- **Daily dev:** 2/10 (heavyweight)
- **Production:** 7/10 enterprise

### Tier: C for reflex
Not the right layer for a CLI tool.

---

# PART 6: Summary Tables

## Best for daily dev (Modal replacement / complement)

| Rank | Option | Why |
|------|--------|-----|
| 1 | **Modal (keep)** + use per-function images for LIBERO/lerobot | Already integrated; per-function image model directly solves the dep-conflict |
| 2 | **uv two-venvs locally** | Zero-infra, cheapest, tight loop |
| 3 | **Docker Compose** locally → Modal for GPU | Familiar, portable, low friction |
| 4 | RunPod Serverless | ~40% cheaper than Modal with BYOD |
| 5 | Beam Cloud | Sub-second cold starts are legit |

## Best for production customer edge deploy

| Rank | Option | Why |
|------|--------|-----|
| 1 | **CloudJetson.com** + reflex-produced TRT engine | Only actual Jetson rental; validates on real silicon |
| 2 | Ship TRT engine + customer's own Jetson (Tailscale for telemetry) | Customer owns hardware; reflex is a build-and-deploy tool |
| 3 | (Distant third) Generic cloud + VPN to customer robot | Latency pain; only if customer insists on cloud |

## Best for benchmarking runs

| Rank | Option | Why |
|------|--------|-----|
| 1 | **Modal** with `@app.function(image=libero_image)` + `.map()` across LIBERO-10 tasks | Parallel rollouts, per-function env, already integrated |
| 2 | RunPod on-demand pods with Docker Compose inside | Cheapest A100/H100 hour with full Docker |
| 3 | SkyPilot multi-cloud | For the cost-optimized "run on whichever is cheapest right now" story |
| 4 | Lambda Labs H100 reserved | For expensive long-running training/distill |

## Best for training (distill command)

| Rank | Option | Why |
|------|--------|-----|
| 1 | Lambda Labs reserved H100 | $2.89/hr (lowest) |
| 2 | Crusoe H100 / H200 | Competitive pricing + multi-node |
| 3 | Modal for short bursts | Convenient, not cost-optimal |
| 4 | Nebius NIM presets | Good for multi-GPU clusters |

---

# PART 7: Most-Likely-Wins for the LIBERO-vs-lerobot Problem (ranked by least engineering)

The question: *given the dep conflict, which option/pattern resolves it with the least engineering effort?*

### Winner 1 — Modal per-function images (zero new infra)

You already use Modal. Define two image specs in your existing modal script:

```python
libero_image = (
    modal.Image.debian_slim(python_version="3.8")
    .apt_install("libosmesa6-dev", "libgl1-mesa-glx", "libglfw3")
    .pip_install("libero==1.0.0", "mujoco-py==2.1.2.14", "gym==0.21")
)

lerobot_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("lerobot==0.4.0", "torch==2.4.0", "transformers==4.57", "mujoco==3.2.3")
)

@app.function(image=libero_image, gpu="A100")
def run_libero_rollout(task_id, policy_url):
    # LIBERO's env.step; post action requests to policy_url (another Modal function)
    ...

@app.function(image=lerobot_image, gpu="A100")
@modal.web_endpoint(method="POST")
def policy_server(obs):
    # lerobot SmolVLAPolicy.select_action(obs)
    ...
```

Then `run_libero_rollout.remote(task_id, policy_server.web_url)`. Each Modal function runs in its own container/image.

**Engineering cost:** ~1 day to split existing `modal_libero10.py` into two functions and a small HTTP shim.

**Runtime cost:** ~50-150ms round-trip per action over Modal's input plane. Reflex VLAs do 10 flow-matching steps per action chunk of 50 actions → action-chunk rate is ~10Hz → 100ms HTTP overhead is tolerable for benchmark. Production real-time control might need the tunnel-based low-latency pattern instead.

**Low-latency variant:** Use `modal.Tunnel` + WebSocket between the two functions (< 5ms on same region). Code lives in this same app; both functions stay on Modal.

### Winner 2 — Local Docker Compose with two services (zero cloud)

For local dev / CI:
```yaml
services:
  libero:
    build:
      context: .
      dockerfile: Dockerfile.libero  # python:3.8 + libero + mujoco-py
    network_mode: host
  lerobot:
    build:
      dockerfile: Dockerfile.lerobot  # python:3.10 + lerobot
    ports: ["8000:8000"]
```
`libero` POSTs obs to `http://localhost:8000/predict`, gets action back.

**Engineering cost:** half a day.
**Runtime cost:** $0 locally, lose GPU on macOS (tolerable for small debug runs).

### Winner 3 — Single Docker multi-stage image (for customer-facing prod)

When you need to ship a self-contained "reflex runtime" binary to a customer:
```dockerfile
FROM python:3.8 AS libero_venv
RUN python -m venv /venvs/libero && /venvs/libero/bin/pip install libero...

FROM python:3.10 AS lerobot_venv
RUN python -m venv /venvs/lerobot && /venvs/lerobot/bin/pip install lerobot...

FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0 AS runtime  # Jetson base
COPY --from=libero_venv /venvs/libero /venvs/libero
COPY --from=lerobot_venv /venvs/lerobot /venvs/lerobot
COPY entrypoint.sh /entrypoint.sh
CMD ["/entrypoint.sh"]
```
`entrypoint.sh` reads `$REFLEX_MODE` and activates the correct venv. Internal IPC over UNIX sockets.

**Engineering cost:** 2 days (Dockerfile tuning is finicky on Jetson base).
**Runtime:** customer runs one `docker run reflex/runtime:latest`.

---

### Why NOT the alternatives for this specific problem:
- **uv two-venvs** alone: works but needs you to still orchestrate IPC and manage CUDA-visible-devices yourself. If you're on Modal already, you're doing more work for less benefit.
- **Nix flakes**: correct answer in theory; 2 weeks of Nix learning tax for a solo founder.
- **Kubernetes**: way over-engineered.
- **Ray Serve**: pulls in Ray just to route two function calls. Bad cost/benefit.
- **Seldon Core / Baseten**: vendor lock-in on a problem local Docker solves.

---

# PART 8: Recommendation for Reflex

1. **Short term (this week):** Keep Modal. Split `scripts/modal_libero10.py` into two Modal functions with two images. Use `web_endpoint` on the lerobot side, plain `.remote()` on the LIBERO side. 1-day project. Unblocks LIBERO-10 run.

2. **Medium term (next month):** Add `CloudJetson.com` integration behind `reflex check --target=jetson-agx`. Actual on-hardware latency / memory numbers are the thing competitors don't have. Also a great "trust signal" screenshot for the reflex landing page.

3. **Long term (before first paying customer):** Ship a multi-stage Docker image (`reflex/runtime:jetson-r36`) that contains both venvs internally. Customer runs one container. Latency locally, IPC over UNIX socket.

4. **Don't bother with:** Nix, Seldon, Baseten, Replicate, Together AI, Mystic, Lightning AI. None of them improve the specific Reflex workflow; they add vendor risk.

5. **Evaluate in 6 months:** Beam Cloud (if cold starts bite), SkyPilot (if Modal prices rise and you want multi-cloud), Crusoe (if you need to train a custom 3.5B pi0 variant from scratch).

---

## Sources

Modal:
- https://modal.com/pricing
- https://modal.com/blog/websocket-launch
- https://modal.com/docs/guide/tunnels
- https://modal.com/docs/guide/images
- https://modal.com/docs/guide/cold-start

Replicate:
- https://replicate.com/pricing (inferred from aggregator data)
- https://www.beam.cloud/blog/top-serverless-gpu-providers

RunPod:
- https://www.runpod.io/pricing
- https://docs.runpod.io/serverless/pricing
- https://www.runpod.io/articles/guides/top-serverless-gpu-clouds

Baseten:
- https://www.baseten.co/pricing/
- https://docs.baseten.co/development/model/custom-server

Together:
- https://www.together.ai/pricing

Cerebrium:
- https://cerebrium.ai/pricing
- https://docs.cerebrium.ai/cerebrium/hardware/using-gpus

fal.ai:
- https://fal.ai/pricing
- https://docs.fal.ai/serverless

Beam:
- https://www.beam.cloud/pricing
- https://github.com/beam-cloud/beta9

Mystic:
- https://www.mystic.ai/

Lightning AI:
- https://lightning.ai/pricing/

Lambda Labs:
- https://lambda.ai/pricing
- https://docs.lambda.ai/public-cloud/billing/

Vast.ai:
- https://vast.ai/pricing
- https://docs.vast.ai/documentation/instances/pricing

CoreWeave:
- https://www.coreweave.com/pricing
- https://docs.coreweave.com/docs/pricing/pricing-instances

Crusoe:
- https://www.crusoe.ai/cloud/pricing

Paperspace:
- https://www.paperspace.com/pricing

Nebius:
- https://nebius.com/prices
- https://nebius.com/newsroom/nvidia-and-nebius-partner-to-scale-full-stack-ai-cloud

Hyperbolic:
- https://docs.hyperbolic.xyz/docs/hyperbolic-pricing
- https://costbench.com/software/ai-gpu-cloud/hyperbolic/

SkyPilot:
- https://github.com/skypilot-org/skypilot
- https://shopify.engineering/skypilot
- https://docs.skypilot.co/en/stable/compute/gpus.html

CloudJetson / Jetson AI Lab:
- https://cloudjetson.com/
- https://www.jetson-ai-lab.com/
- https://github.com/NVIDIA-AI-IOT/jetson-ai-lab

Luxonis:
- https://docs.luxonis.com/
- https://www.luxonis.com/

Tailscale self-hosted:
- https://tailscale.com/blog/self-host-a-local-ai-stack
- https://dataforcee.us/2026/02/26/tailscale-and-lm-studio-launch-lm-link-to-give-you-end-to-point-encrypted-access-to-your-private-gpu-computing-asset/

uv / Python envs:
- https://docs.astral.sh/uv/pip/environments/
- https://docs.astral.sh/uv/concepts/projects/config/

Docker multi-stage:
- https://pythonspeed.com/articles/multi-stage-docker-python/
- https://dev.to/kummerer94/multi-stage-docker-builds-for-pyton-projects-using-uv-223g

Nix:
- https://nixos.wiki/wiki/Python
- https://www.tweag.io/blog/2020-08-12-poetry2nix/

Dev containers:
- https://github.com/microsoft/vscode-dev-containers
- https://deepwiki.com/devcontainers/features/3.1-python

Ray Serve:
- https://docs.ray.io/en/latest/serve/advanced-guides/multi-app-container.html
- https://docs.ray.io/en/latest/serve/production-guide/handling-dependencies.html

K8s sidecar:
- https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/

Seldon Core:
- https://docs.seldon.ai/seldon-core-2

LIBERO / lerobot:
- https://github.com/huggingface/lerobot-libero
- https://github.com/huggingface/lerobot
- https://github.com/huggingface/lerobot/issues/2114
- https://github.com/huggingface/lerobot/issues/2697

Physical Intelligence (openpi):
- https://github.com/Physical-Intelligence/openpi
- https://www.pi.website/blog/openpi
