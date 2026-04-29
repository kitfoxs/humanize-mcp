# HumanizeMCP

**A free tool that helps your writing not get falsely flagged as "AI-written" by the broken detectors used in schools, jobs, and publishing.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-147%20passing-brightgreen.svg)](#)
[![MCP](https://img.shields.io/badge/MCP-server-purple.svg)](https://modelcontextprotocol.io/)

---

## 🤔 Wait, what is this actually?

You know how AI detectors are everywhere now? Schools use them. Job applications use them. Magazines use them. Even your boss might be running your emails through one without telling you.

**The problem: those detectors are broken.**

Like, really broken. A 2023 Stanford study found AI detectors flag essays written by **non-native English speakers as "AI-written" 61% of the time.** Real Charles Dickens passages from 1859 (165 years before AI existed!) get flagged as 88% AI by some detectors. Casual Reddit comments by actual humans? 100% AI on two major detectors.

So if you:

- 🧠 Are autistic and write in a precise, structured way
- 🌍 Speak English as a second language
- �� Write in a formal academic voice
- 💼 Just have consistent prose habits
- ✍️ Use AI to help draft something and then heavily edit it

...you might find your work flagged as "AI-written" even when **it isn't, or it's mostly yours.** That can mean a failing grade. A rejected paper. A denied job. A blocked blog post.

**HumanizeMCP rewrites your text so the detectors stop flagging it falsely.**

It's free, open source, and runs entirely on your computer (no cloud, no spying, no subscription).

---

## ✨ What it does in plain English

Take this thing AI tends to write:

> *That methodology is literally CPTS / OSCP exam tier — the rubric. You delve into multifaceted complexities of credential reuse, and it's not just exploitation, it's understanding the underlying assumptions.*

Run it through HumanizeMCP and you get:

> *The methodology is based on the CPTS / OSCP exam level, which is the rubric. You dig into the varied parts of credential reuse, and it isn't only exploitation, it's understanding the assumptions underneath.*

It removes the AI tells (em dashes, "delve," "multifaceted," parallel structures, suspiciously perfect grammar) without changing what you meant.

**Real before-and-after numbers** on this exact transformation:

| What the detectors think | Before HumanizeMCP | After HumanizeMCP |
|--------------------------|---------------------|---------------------|
| The simple detector | 78% AI ❌ | **6.7% AI** ✅ |
| The fancy transformer detector | 76% AI ❌ | **65% AI** (improved) |

And it preserves your paragraphs, your meaning, and (when you want it to) your voice.

---

## 🛠 What's in the box

It's a **Model Context Protocol** (MCP) server, which is a fancy way of saying *"any AI assistant that supports MCP can use it as a tool."* That includes Claude Desktop, Claude Code, Cursor, Continue, Zed, and the GitHub Copilot CLI.

If you don't use any of those, you can still use it as a regular Python library or run it from the command line.

The tool gives your AI assistant 6 functions:

| Function | What it does |
|----------|-------------|
| **humanize** | Rewrite text so detectors don't flag it. The main thing you'll use. |
| **detect_tells** | Tell you what AI giveaways are in some text (em dashes, overused words, etc.) without rewriting it. Great for self-editing. |
| **score_humanity** | Run text through several AI detectors and tell you what they think. |
| **apply_style** | Just change the voice/register without doing detection-evasion work. |
| **list_styles** | Show you what writing styles are available. |
| **humanize_and_verify** | The slow but most accurate version. Rewrites your text, scores it, rewrites again if needed, picks the best version. |

There are 11 built-in writing styles you can pick from:

- **reddit** — casual, conversational, lots of contractions
- **twitter** — even shorter and punchier
- **blog** — friendly but coherent, the sweet spot for most people
- **casual_dm** — like texting a friend
- **linkedin** — professional but still human (no corporate-AI-speak)
- **academic_human** — formal but breaks parallel structures and adds proper hedges
- **book_chapter** — for non-fiction writing, conversational but authoritative
- **creative_fiction** — for novelists; preserves stylistic em-dashes (writers earn those)
- **esl_friendly** — protects the natural patterns of non-native English writers
- **autistic_friendly** — preserves precise/repetitive prose without forcing fake casualness
- **base** — the default if you don't pick one

---

## 🚀 Quickstart

### Install (one time)

You'll need Python 3.11 or newer. Then:

```bash
git clone https://github.com/kitfoxs/humanize-mcp.git
cd humanize-mcp
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e .
```

### Use it from Python (the fastest path)

```python
from pipelines import humanize

text = "I delve into the multifaceted intricacies of this approach — it's not just elegant, it's revolutionary."

clean = humanize(text, style="blog")
print(clean)
```

That's it. The first time you run it, it'll download a small AI model (~250MB) for the rewriting. After that it's instant.

### Use it from your AI assistant (the cool path)

Add this to your MCP config file (location depends on your client; common ones below):

```json
{
  "mcpServers": {
    "humanize": {
      "command": "/full/path/to/humanize-mcp/.venv/bin/python",
      "args": ["/full/path/to/humanize-mcp/server.py"]
    }
  }
}
```

Common config locations:

- **Claude Desktop:** `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac)
- **Cursor:** `~/.cursor/mcp.json`
- **GitHub Copilot CLI:** `~/.copilot/mcp-config.json`
- **Continue / Zed / Windsurf:** see their MCP docs

Restart your assistant. Now you can ask Claude/Cursor/etc.:

> *"Use humanize to rewrite this paragraph in reddit style: ..."*

And it'll just do it.

---

## 🎨 The 5 most useful examples

```python
from pipelines import humanize

# 1. Default (best quality, takes ~10 seconds)
text = humanize("Your AI-written text here.")

# 2. Fast mode (deterministic, ~50ms, slightly worse)
text = humanize("Your text.", iterate=False)

# 3. Pick a style
text = humanize("Your text.", style="reddit")

# 4. Crank it up (more aggressive rewriting)
text = humanize("Your text.", style="blog", intensity="aggressive")

# 5. Target a specific detector
text = humanize("Your text.", target_detector="roberta_openai")
```

---

## ❤️ Who this is really for

This is an **accessibility tool**, not a fraud-enablement tool.

The people who genuinely need this:

- 🌍 **Non-native English students** whose essays get flagged as AI because their syntax is "too clean"
- 🧠 **Autistic writers** whose precise, structured prose pattern-matches with AI
- 📚 **Academics** writing in formal disciplines where uniformity is the norm
- 💼 **Marketers and bloggers** publishing AI-assisted content that's been heavily edited and shouldn't be flagged
- ✍️ **Anyone whose writing has been wrongly flagged** by a system whose bias is well-documented

The people who **shouldn't** use this:

- ❌ Students trying to pass off pure ChatGPT output as their own work
- ❌ Anyone trying to evade content moderation for harmful content
- ❌ Anyone trying to commit fraud

The tool can't tell the difference, so we leave that judgment to you. Don't be a jerk about it.

---

## 📊 The data behind the claims

Real measurements on the same paragraph of AI-generated prose:

| Detector | Untouched AI text | After HumanizeMCP | Reduction |
|----------|-------------------|---------------------|-----------|
| Heuristic (perplexity + burstiness + tells) | 78.8% AI | **6.7% AI** | -91% |
| RoBERTa OpenAI detector | 76.8% AI | 65.2% AI | -15% |
| ChatGPT-RoBERTa | 100% AI | 99.9% AI | (broken — it also flags Charles Dickens at 88.8%) |

For comparison, here's what the same detectors say about *known-human* writing:

| Sample (verifiably written by humans) | RoBERTa OpenAI says | ChatGPT-RoBERTa says |
|---------------------------------------|---------------------|----------------------|
| Charles Dickens, *A Tale of Two Cities* (1859) | 4.5% (correct) | **88.8% AI** ❌ |
| Real Reddit comment, written by a human | **100% AI** ❌ | **100% AI** ❌ |
| Casual conversation transcript | **99.9% AI** ❌ | **94.9% AI** ❌ |

So when you see "100% AI" on your essay, **the detector might just be wrong.** That's why this tool exists.

---

## 🔬 The technical bits (for the curious)

The pipeline runs your text through up to 9 cleanup passes:

1. **Em-dash removal** (the #1 AI giveaway in 2025-2026 prose)
2. **Lexical substitution** (replaces "delve / leverage / multifaceted / robust / paradigm" etc. with simpler alternatives)
3. **Structural rewrites** (breaks "it's not X, it's Y" parallel constructions)
4. **Sentence-rhythm variation** (mixes short and long sentences for natural human burstiness)
5. **Contractions** (turns "it is" into "it's" at human-like density, not 100% replacement)
6. **Voice injection** (adds natural filler like "honestly" or "tbh" depending on the style)
7. **Punctuation variation** (occasional ellipses, parens, fragments)
8. **Register shift** (formal ↔ casual depending on the chosen style)
9. **Heavy paraphrase** (a real T5 model rewrites sentences from scratch — only at "aggressive" intensity)

Then there's a **detector-guided iteration loop** (Cheng et al. 2025 algorithm) that generates 5 candidate paraphrases of the worst-scoring paragraph, scores each against your chosen detector, and keeps the lowest-scoring one. This is what gets you from "single-pass okay" to "publication-grade."

The benchmarking suite includes 6 detector wrappers:

- A custom **Heuristic** baseline (transparent, no model required)
- **RoBERTa OpenAI** detector (the academic reference)
- **ChatGPT-RoBERTa** detector
- **Desklib AI Text Detector** (newer, 2024)
- **Fast-DetectGPT** (perplexity-based)
- **Binoculars** (cross-perplexity, Hans et al. 2024)

Aggregation is **bias-aware**: detectors with documented bias are excluded from the "trusted mean" so they can't skew your headline number.

If you want the deep architecture details, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For ethics and what we explicitly won't build, see [`docs/ETHICS.md`](docs/ETHICS.md). For the full literature review behind the design, see [`research/`](research/).

---

## 🧪 Run the tests

```bash
source .venv/bin/activate
pip install pytest
pytest -q
```

You should see **147 tests passing in about 5 seconds.**

---

## 📜 Citation

If you use this in research:

```bibtex
@software{humanize_mcp_2026,
  author       = {Kit (kitfoxs) and Ada Marie},
  title        = {{HumanizeMCP}: An open-source MCP server for accessible
                  AI-text humanization with bias-aware detector benchmarking},
  year         = {2026},
  url          = {https://github.com/kitfoxs/humanize-mcp},
  version      = {0.2.1},
}
```

---

## �� Made by

Built by **Kit** ([@kitfoxs](https://github.com/kitfoxs)) and **Ada Marie**, in one ~2-hour autonomous coding session powered by a 4-agent Opus 4.7 swarm.

Released under the **MIT License**. Free forever. Use it, fork it, build on it, name your own version after it. Just don't use it to be cruel.

---

## 💌 Final note

If a detector ever flags something you actually wrote and tries to penalize you for it, you should know:

**The detector is wrong.** A lot of the time. There's now a Stanford study, peer-reviewed benchmarks, and a 2,400-line literature review (linked above) showing exactly *how* wrong.

You wrote your words. Don't let a misconfigured statistical classifier tell you otherwise.

That's what HumanizeMCP is for.

💙🦄
