# Customer first-run transcript (2026-04-19)

**Method.** Pretended to be a first-time user who has only read `README.md`. Followed the quickstart commands verbatim on a fresh NVIDIA TRT container + Modal A10G. Did NOT peek at `src/`, `tests/`, `reflex_context/`, or any docs beyond README. Recorded every output, did not troubleshoot on the fly.

**Reproducer:** `modal run scripts/modal_customer_dogfood.py`.

**Bottom line:** the end-to-end customer flow works — `curl POST /act` returns a 50×32 action chunk with every field the README promises, in ~0.2s on `onnx_trt_fp16`. But four surprises would dent customer trust.

---

## Surprise #1 (SEVERE) — the README quickstart routes customers to the abandoned decomposed export path

The README says:
```bash
reflex export lerobot/smolvla_base --target desktop --output ./smol
```

What actually gets produced:
```
-rw-r--r-- 1 root root   1161789 Apr 19 04:17 vision_encoder.onnx
-rw-r--r-- 1 root root 392953856 Apr 19 04:17 vision_encoder.onnx.data
-rw-r--r-- 1 root root   1145891 Apr 19 04:16 expert_stack.onnx
-rw-r--r-- 1 root root 405340160 Apr 19 04:16 expert_stack.onnx.data
-rw-r--r-- 1 root root 202541732 Apr 19 04:17 expert_stack.trt
-rw-r--r-- 1 root root   2229534 Apr 19 04:18 decoder_prefill.onnx
-rw-r--r-- 1 root root 594411520 Apr 19 04:18 decoder_prefill.onnx.data
-rw-r--r-- 1 root root   1964 Apr 19 04:18 text_embedder.onnx
-rw-r--r-- 1 root root 189235200 Apr 19 04:18 text_embedder.onnx.data
-rw-r--r-- 1 root root      3968 Apr 19 04:18 state_proj_bias.npy
-rw-r--r-- 1 root root    123008 Apr 19 04:18 state_proj_weight.npy
```

This is the **5-file decomposed export** — the path we abandoned because it had 12 known correctness bugs (LIBERO-10 = 0% task success on this path). The `cos=+1.000000` machine-precision story in the README belongs to the **monolithic** path, which requires an explicit `--monolithic` flag.

**The customer running the README quickstart verbatim gets the wrong path.** They think they got the verified cos=1.0 export; they actually got the abandoned decomposed one. If they compare actions against PyTorch reference they'll see real drift, despite the VERIFICATION.md showing PASS (the VERIFICATION.md only verifies one stage — the expert stack — not the full pipeline).

**Fix options:**
1. Change the CLI default to `--monolithic` (switch the flag polarity: `--decomposed` becomes the opt-in)
2. Add `--monolithic` to the README quickstart command verbatim
3. Delete the decomposed exporter entirely (simplest, most honest)

Option 1 or 3 is the right call. Option 2 is a workaround that leaves the sharp edge.

---

## Surprise #2 (HIGH) — VERIFICATION.md says "Model type: unknown" for a supported model

```
- **Model:** `lerobot/smolvla_base`
- **Model type:** unknown
```

SmolVLA is one of four explicitly supported models. The VERIFICATION.md receipt — our "ship a receipt, not just a binary" pitch — reports the model type as unknown. A customer reading this on day one assumes the export silently fell back to a generic path and lost some model-specific handling.

Root cause is probably in the decomposed exporter's config-writer — the monolithic path likely writes `model_type` correctly. But this is the customer experience today for the README-path customer.

---

## Surprise #3 (MEDIUM) — scary server log warnings on first boot

The customer who reads `reflex serve ./smol` stdout sees:
```
WARNING: ModelImporter.cpp:420: Make sure input position_ids has Int64 binding.
WARNING: ModelImporter.cpp:420: Make sure input prefix_offset has Int64 binding.
WARNING: Detected layernorm nodes in FP16.
WARNING: Running layernorm after self-attention with FP16 Reduce or Pow may cause overflow.
WARNING: Forcing Reduce or Pow Layers in FP32 precision
```

Also:
```
INFO: AutoProcessor not available from HuggingFaceTB/SmolVLM2-500M-Video-Instruct:
      Could not import module 'SmolVLMProcessor'. Are this object's requirements
      defined correctly? -- will use manual image preprocessing
```

These are harmless (expected FP16 warnings + a known processor fallback) but the customer doesn't know that. Reflex either needs to:
- Suppress known-benign warnings
- Translate the one benign fallback into a single INFO line ("Note: using manual image preprocessing — expected")
- Add them to `reflex doctor`'s known-good list so `doctor` can explicitly mark them safe

Right now the customer sees 4+ warnings on a successful boot. That's a trust dent.

---

## Surprise #4 (LOW) — README curl example has no image

The README's `curl` example:
```bash
curl -X POST http://localhost:8000/act -H 'content-type: application/json' \
  -d '{"instruction":"pick up the red cup","state":[0.1,0.2,0.3,0.4,0.5,0.6]}'
```

No image. The server accepts this and returns actions — but a VLA without an image is obviously degenerate. What does the server actually do? (Probably uses a zero or random image — customer has no idea.) The README teaches the customer a bad pattern.

Also: `state` is 6 floats, but the response is 50×32 actions. Customer has no guidance on:
- What if my robot has 7 DOF? 14 DOF?
- What's the scale/range of `state` values?
- Is 6 the correct length for SmolVLA?
- Why is the output 32-dim when I sent 6-dim state?

---

## What worked well

- **Install from git URL:** worked. ~1–2 min on the TRT container image. No cuDNN errors (TRT container has it).
- **`reflex --help` / `reflex doctor` / `reflex models` / `reflex targets`:** all work. `reflex doctor` output is reassuring.
- **Export time:** 197s for SmolVLA (~3 min on A10G). Acceptable for a first run.
- **VERIFICATION.md:** exists, contains SHA256 of every file + opset + metadata. Good receipt pattern (minus the "Model type: unknown" bug).
- **Server warmup:** 69s — within the README's promised 30–90s window.
- **`/act` response:** worked on first try with the README curl. All fields present (`actions`, `num_actions`, `latency_ms`, `denoising_steps`, `inference_mode`). Response time 200ms.
- **inference_mode reported:** `onnx_trt_fp16` — customer knows they're on GPU.

The ergonomic loop (install → export → serve → /act) is smooth. The positioning problems are in where `reflex export` routes + what the README teaches.

---

## Transcript (condensed)

```
$ python --version          → Python 3.12.x    (0.1s)
$ pip --version             → pip 24.x         (0.7s)
$ which reflex              → /usr/local/bin/reflex  (0.2s)
$ reflex --version          → 0.1.0            (4.6s) ← first import, slow
$ reflex --help             → full command list  (2.9s)
$ reflex doctor             → passes           (6.9s)
$ reflex models             → 5 models listed   (2.6s)
$ reflex targets            → 5 targets listed  (2.4s)
$ reflex export lerobot/smolvla_base --target desktop --output /tmp/smol
                            → PASS max_diff=3.34e-06  (197.2s)
                              [⚠ routed to DECOMPOSED path, not monolithic]
$ ls -la /tmp/smol          → 16 files, 1.7GB  (0.1s)
                              [⚠ 5-file decomposed layout, not single model.onnx]
$ test -f VERIFICATION.md   → YES              (0.1s)
$ head VERIFICATION.md      → OK, receipt has SHA256 + metadata
                              [⚠ "Model type: unknown"]
$ reflex serve ./smol ...   → ready after 69s
                              [⚠ 4+ TRT/FP16 warnings in log]
$ curl POST /act            → 50×32 action chunk, onnx_trt_fp16  (0.2s)
                              [✓ all README-promised fields present]
```

---

## Recommended fixes ordered by customer impact

1. **Switch `reflex export` default to monolithic.** The decomposed path shouldn't be the default the README documents; it shouldn't be reachable without an explicit opt-in flag. (Severe: this is the path the whole launch story depends on.)
2. **Fix model_type detection in the decomposed path's `VERIFICATION.md` writer** (OR delete the decomposed path entirely, which is cleaner).
3. **Add `image` parameter to the README curl example.** Show what a real request looks like — document the `image` field's expected format (base64 JPEG or URL), expected `state` shape per model, expected action output shape.
4. **Suppress or translate benign server startup warnings.** The customer should see a clean boot log, not 4 TRT warnings.
5. **Clarify `reflex doctor` output to include a "known-safe warnings" section.** When TRT/FP16 warnings appear, `doctor` should be able to say "these are expected."

Items 1 and 3 are the biggest. The first fix aligns shipped behavior with the verified parity story; the third teaches customers what a real `/act` call looks like.

---

## What a first-time customer would say

After this run, a candid customer message to us would probably read:

> "It worked — I got actions back in a couple of hundred ms. Quick for a first install. But your README says cos=1.0 verified parity, and my VERIFICATION.md says 'Model type: unknown' and I got five ONNX files instead of one. Did I do something wrong? And I noticed the curl doesn't send an image — is that intentional? What's the right way to send an image?"

Two of those questions (model_type + 5 files vs 1) come directly from the decomposed-vs-monolithic default bug.

---

## Related

- `scripts/modal_customer_dogfood.py` — the reproducer
- `README.md` — the customer's starting point
- `src/reflex/cli.py` — where the `reflex export` default is set
- `06_experiments/monolithic_parity_table.md` — the verified path (monolithic)
- `02_bugs_fixed/smolvla_pipeline_bugs.md` — the 12 known bugs in the decomposed path the customer just got
