# NEURAX — Hacker News Launch Guide (Show HN)

Complete playbook for publishing NEURAX on Hacker News. Follow exactly.

---

## 1. LES RÈGLES D'OR (officielles)

- **Show HN = quelque chose qu'on peut essayer immédiatement** — pas de landing page, pas de signup. NEURAX est parfait : open source + demo live + `cargo install`
- **Ne demandez JAMAIS de upvotes** (ni à des amis) — la détection anti-ring de HN peut shadowban votre URL ou bannir votre domaine
- **Pas de repost** — ne supprimez pas pour reposter. HN a un "pool" où les modérateurs ré-exposent parfois les bons posts
- **Le titre doit commencer par "Show HN:"** — sinon pas de queue dédiée ni de protection

---

## 2. LE TITRE — 90% du succès

**Format :** `Show HN: [ce que c'est] – [un différenciateur]`
**Règles :** < 80 caractères · zéro adjectif · zéro superlatif · zéro emoji · décrire, pas vendre

### Option retenue (76 chars, vérifié) :
```
Show HN: NEURAX – analytical compiler that predicts ML training costs in <50ms
```

### Alternatives si besoin :
```
Show HN: NEURAX – compute FLOPs/VRAM/cost of an architecture JSON in <50ms
Show HN: NEURAX – analytical compiler for neural net cost estimation (Rust+MLIR)
```

### ❌ À NE PAS mettre :
- "revolutionary", "game-changing", "best", "10x faster"
- "NEURAX – the future of AI" ❌
- Version numbers ("NEURAX 0.1.0") ❌

---

## 3. LE PREMIER COMMENTAIRE — votre pitch (poster dans les 60 secondes)

C'est le texte le plus lu avant même le clic. Écrivez-le AVANT de soumettre.

```
I built NEURAX because I kept facing the same problem: before renting
8x H100s, I needed to know if an architecture would fit in VRAM and what
training would actually cost — but frameworks like PyTorch only give you
those numbers after you train.

NEURAX is an analytical compiler: it runs a 10-pass IR
(Architecture → Graph → Tensor → Operator → Compute → Memory →
Parallelism → Hardware → Cost → Report) over a model JSON and computes
55+ metrics — FLOPs, VRAM, latency, USD, kWh, CO2 — deterministically,
in under 50ms, on a CPU, with zero GPU.

The interesting technical choice: instead of executing the model, it
derives everything from closed-form analytical formulas per
architecture family (Transformer, MoE, CNN, SSM, RNN, Diffusion...).
There's also a custom MLIR backend — 10 dialects on LLVM 18 — for
lowering.

Honest limitations: the benchmark table currently shows predictions,
not measured runs — I'm looking for validation partners to compare
against real training. Cost metrics assume hardware specs in the
config.

Everything is open source (9 crates on crates.io):
cargo install neurax-cli
https://github.com/rustnew/NEURAX

I'd love feedback on the analytical formulas and the IR design — and
if you have real training numbers for GPT/LLaMA/Mixtral, I'd be very
interested in comparing.
```

---

## 4. LE TIMING — fenêtre exacte

| Créneau | Pourquoi |
|---------|----------|
| **Mardi–Jeudi, 8h–10h ET** | Pic d'activité : côte Est au réveil, côte Ouest au bureau, Europe en après-midi |
| Dimanche 10h ET (repli) | Moins de concurrence, audience exploratoire |
| ❌ Lundi matin / vendredi après-midi / week-end | Mort : personne ne commence un thread technique |

**La décision front page se joue dans les 60-90 premières minutes** : 8-12 points dans la première heure = graduation probable.

> ⚠️ Important : postez à un moment où vous êtes disponible 3-4h. Ne postez pas à 9h ET si c'est 2h du matin chez vous et que vous allez dormir.

---

## 5. STRATÉGIE DE COMMENTAIRES (les 2 premières heures)

| Heure | Action |
|-------|--------|
| T+0 | Soumettre + premier commentaire immédiat (60 sec max) |
| T+0 → T+2h | **Répondre à CHAQUE commentaire en <15 min**, 2-4 phrases, sans défensive |
| T+2h → T+6h | Surveiller, répondre aux arrivants tardifs toutes les 2h |
| T+6h → T+24h | Réponses occasionnelles si le thread survit |

**Réponses pré-préparées aux objections prévisibles :**

**"Pourquoi pas XLA/IREE/ONNX?"**
> Différentes étapes du cycle de vie. XLA/IREE exécutent le modèle (runtime). NEURAX travaille en design-time : avant d'acheter le GPU. Vous ne pouvez pas exécuter un modèle que vous n'avez pas encore construit.

**"Tes prédictions sont-elles exactes?"**
> Le tableau de benchmark montre 11 modèles analysés en <10ms chacun. L'exactitude des prédictions vs les runs réels est ce que je valide maintenant — je cherche des partenaires de validation. Les formules analytiques sont déterministes : même entrée → même sortie.

**"C'est juste une calculatrice de FLOPs?"**
> La partie visible est la calculatrice, mais le cœur est le pipeline IR 10 passes + les dialectes MLIR custom + le lowering LLVM 18. Chaque passe transforme la représentation et ajoute des métriques (parallélisme, mémoire par précision bf16/fp8, ridge point, etc.).

**"Pourquoi Rust?"**
> Déterminisme, performance (analyses <10ms même pour 175B params), zéro GC pour un pipeline d'analyse, et un écosystème fort pour les compilateurs (melior/MLIR).

**"Qui a besoin de ça?"**
> Toute équipe qui décide d'un budget GPU avant de former : estimation de coût 8x H100, vérification VRAM, planification énergie/CO2, conformité (EU AI Act, CSRD).

**"Pourquoi tu as mis MLIR là-dedans?"**
> NEURAX a un backend MLIR avec 10 dialectes custom qui reflètent le pipeline IR — le lowering vers LLVM IR est le chemin vers l'exécution CPU/GPU. C'est la partie "compilateur" du projet.

---

## 6. PRÉPARATION AVANT LE JOUR J (3-7 jours)

- [ ] **Compte HN avec karma** : si le compte est neuf (<2-3 semaines, <50 karma), il est downranké. Commentez utilement pendant 2-3 semaines si besoin
- [ ] **Démo live prête** : rustnew.github.io/NEURAX doit marcher parfaitement (les gens vont l'essayer immédiatement)
- [ ] **README optimisé** : lien demo + cargo install en haut, benchmark table visible
- [ ] **GIF démo** : canvas + analyse en action (si possible)
- [ ] **10-20 supporters** prévenus : "je lance demain 9h ET, si ça vous intéresse, un commentaire utile dans le thread m'aiderait" — JAMAIS "upvotez"
- [ ] **Préparer la réponse au "hug of death"** : GitHub Pages + repo doivent tenir 5-30K visiteurs
- [ ] **Imprimer le premier commentaire** (section 3) et les réponses (section 5)

---

## 7. APRÈS LA PUBLICATION

**Si front page (5K-30K visites, 500-2K stars pour un OSS) :**
- Répondez pendant 18-24h — un thread bien géré survit 18-24h vs 4-6h sans le fondateur
- Un commentaire bien répondu = plus de visibilité (l'algorithme pondère la vélocité des commentaires)
- Les investisseurs lisent le thread avant un call — c'est un actif de crédibilité

**Si ça ne décolle pas (5 upvotes, disparaît) :**
- ❌ Ne supprimez pas, ne repostez pas (flag automatique)
- Lisez les 3-5 commentaires reçus — ils valent de l'or
- Tentez un autre jour avec un angle différent (ex: post technique sur les formules analytiques, pas Show HN)
- Retentez dans 6+ mois avec une version significativement améliorée

---

## 8. CHECKLIST FINALE (matin du jour J)

```
[ ] 8h30 ET : tout est prêt (démo, README, commentaires imprimés)
[ ] 8h-10h ET : soumettre avec le bon titre "Show HN: ..."
[ ] +60 sec : premier commentaire posté
[ ] 0-2h : réponses <15 min à tout
[ ] 2-6h : surveillance
[ ] JAMAIS : demander des upvotes, argumenter, supprimer/reposter
```