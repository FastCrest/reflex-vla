# launch/

Drafts for public posts. **Nothing here is published yet — all need user approval before going live.**

| File | Where it goes | Status |
|---|---|---|
| `lerobot_3146_draft.md` | Comment on huggingface/lerobot#3146 | Draft v1 |
| `show_hn_draft.md` | news.ycombinator.com Show HN | Draft v1 |
| `reddit_robotics_draft.md` | reddit.com/r/robotics | Draft v1 |

## Sequencing

When ready to launch:

1. **Post LeRobot #3146 first** (most strategic — that's where the actual VLA users live)
2. **48-72h later, post Show HN** (orthogonal audience, broader tech)
3. **Same day or next day, post r/robotics** (third audience)

Don't post all three the same day — reduces signal in each, and means you can't respond to comments in any of them.

## Pre-launch checklist

- [ ] All current Phase I + II + III work pushed to github.com/rylinjames/reflex-vla
- [ ] README.md current with TRT FP16 numbers
- [ ] Roadmap repo current with the same numbers (single source of truth)
- [ ] At least one Jetson benchmark run (or honest "untested on Jetson" disclaimer)
- [ ] GitHub Issues open and you commit to <24h response
- [ ] Discord invite or Slack-channel link in the README
- [ ] `pip install ... @ git+https://...` install path tested on a fresh box
