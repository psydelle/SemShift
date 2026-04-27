---
name: SemShift study background
description: Theoretical motivation, method, and research questions for the SemShift project
type: project
---

Relational categories (verbs) are more mutable than entity categories (nouns). Under semantic strain (incompatible verb-noun pairings), verbs should shift more than nouns.

**Original motivation**: collocations and abstract nouns are slow to process. Could semantic neighbourhood density (SND) be the locus of processing cost? Brysbaert concreteness ratings failed to predict RTs, and are theoretically problematic (Pollock, 2017). SND was adopted as a proxy correlated with concreteness.

**SND finding**: SND for nouns similarly distributed across collocation (kill time) vs productive (kill rabbit) conditions. iRT ~ Condition + SND showed no significant differences.

**Current approach**: Delta metrics as a proxy for SND for verbs.
- X[word] = average embedding across all contexts for that word (generic representation)
- Y[word] = embedding in a specific collocation
- Delta = X - Y (deviation from typical usage)
- |Delta| = magnitude of semantic shift

**Research questions**:
1. Mutability: Is |Delta_verb| > |Delta_noun|? (verbs shift more than nouns)
2. RT prediction: Does |Delta| predict processing costs? (larger shift = slower RT)
3. Directionality: Do Deltas point in systematic directions? (future: steer generative models toward novel collocations — not for EMNLP)

**Target venue**: EMNLP

**Why:** The more semantically incompatible the noun is w.r.t. the verb, the greater the change in verb meaning.

**How to apply**: Delta computation must control for pool size (larger pool = more stable average = regression to mean artefact). Verb-queried set has better verb pool sizes; noun-queried has better noun pool sizes. Consider computing verb Deltas from verb-queried set and noun Deltas from noun-queried set.
