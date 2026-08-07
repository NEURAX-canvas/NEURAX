# 🎓 arXiv Endorsement Outreach — NEURAX

**Objectif :** Obtenir un parrainage (endorsement) d'un auteur arXiv établi pour soumettre le technical report NEURAX en `cs.LG`.

**Contexte :** Depuis le 21 janvier 2026, arXiv exige pour une première soumission sans affiliation académique un **endorsement personnel** d'un auteur ayant déjà publié dans la catégorie visée. arXiv ne peut pas délivrer lui-même d'endorsement.

**Preuves à joindre / mentionner pour crédibiliser la demande :**
- ✅ 8 PR acceptées dans des awesome lists (awesome-machine-learning, awesome-rust-llm, best-of-ml-rust, etc.)
- ✅ Technical report complet : `paper/main.pdf` (8 pages, compilation propre)
- ✅ Dépôt Zenodo (DOI `10.5281/...` une fois publié)
- ✅ Repo GitHub public : https://github.com/rustnew/NEURAX
- ✅ Documentation complète : https://rustnew.github.io/NEURAX/

---

## 📌 Conseils généraux

1. **Privilégiez le canal LinkedIn** — les chercheurs y répondent plus qu'ailleurs, et vous pouvez cibler précisément.
2. **Cherchez des chercheurs en systèmes ML / compilation** (IREE, XLA, MLIR, TVM, ONNX Runtime) — ils comprendront immédiatement l'intérêt de NEURAX.
3. **Personnalisez toujours** la première phrase (mentionnez leur projet/paper).
4. **Envoyez 10-15 demandes** — le taux de réponse typique est de 1 sur 10-20.
5. **Attendez 5-7 jours**, puis relancez (template inclus).
6. **L'endorsement est gratuit** pour eux : 2 minutes dans leur compte arXiv, et c'est une contribution à la communauté.
7. **Soyez honnête** : dites clairement que vous êtes indépendant sans affiliation.

---

## 1. Email (format long, à envoyer sur l'email pro trouvé via leur page labo)

**Objet :** Request for arXiv endorsement (cs.LG) — open-source analytical compiler

```
Dear [Name],

I'm reaching out because I built NEURAX, an open-source analytical compiler for
neural network architectures (https://github.com/rustnew/NEURAX), and I would
like to deposit its technical report on arXiv. As an independent researcher
without an institutional affiliation, arXiv requires me to be personally
endorsed by an established author in the cs.LG category.

Why I think you might be willing to endorse me:
- NEURAX predicts training cost, peak memory, and inference behavior of
  architectures (Transformer, CNN, MoE, Diffusion, SSM, GNN, ...) in under 50ms
  with zero GPU, using an 11-phase IR pipeline and MLIR/LLVM 18 backend.
- The validation reports 97-99.5% accuracy across five families vs real
  training runs (see BENCHMARKS.md in the repo).
- The paper (8 pages) is ready: DOI available on Zenodo, PDF at
  https://github.com/rustnew/NEURAX/tree/main/paper
- 8 pull requests I authored have been accepted into awesome lists in the ML/Rust
  ecosystem (awesome-machine-learning, best-of-ml-rust, awesome-rust-llm, ...).

The endorsement itself takes ~2 minutes from your arXiv account and simply
vouches that I am a member of the scientific community submitting quality work.
I commit to only submitting rigorous, well-documented research.

Would you be open to endorsing me? Thank you for considering — and regardless
of your answer, I'd welcome any feedback on the project.

Best regards,
Martial Fossouo
Independent Researcher — martialwato50@gmail.com
https://github.com/rustnew/NEURAX
```

---

## 2. LinkedIn (version courte — message InMail < 1800 caractères)

**Première ligne personnalisée obligatoire :** référencez leur travail
("J'ai vu votre travail sur X — impressionnant, notamment Y").

```
Bonjour [Prénom],

Développeur indépendant, je viens de terminer un technical report sur NEURAX,
un compilateur analytique open source qui prédit le coût d'entraînement, la
mémoire GPU et le comportement d'une architecture (Transformer, MoE, Diffusion,
SSM...) en moins de 50 ms, sans GPU. Implémenté en Rust avec backend MLIR/LLVM 18.
Le papier (8 pages) est prêt et archivé avec DOI.

Pour le déposer sur arXiv (cs.LG), il me faut un endorsement d'un auteur établi
dans cette catégorie, et je n'ai pas d'affiliation académique.

La demande prend 2 minutes de votre côté : vous me parrainez depuis votre compte
arXiv. Je m'engage à ne soumettre que des travaux de qualité.

Le repo : https://github.com/rustnew/NEURAX — validation à 97-99.5% vs entraînements réels.

Seriez-vous disposé à m'endorser ? Merci d'avance, et au plaisir d'échanger sur
le projet.

Martial
```

---

## 3. X / Twitter (version très courte — à poster publiquement ou en DM)

```
I built an open-source analytical compiler for neural nets (Rust + MLIR/LLVM 18):
predicts training cost/memory/behavior in <50ms, no GPU, 97-99.5% accuracy vs
real runs. Paper is ready w/ DOI. As an independent researcher I need an arXiv
cs.LG endorsement to submit it.

If you're an established arXiv author in ML systems, could you spare 2 min to
endorse me? https://github.com/rustnew/NEURAX

#AcademicTwitter #OpenSource #ML #LLM
```

**Hashtags recommandés :** `#AcademicTwitter`, `#OpenSource`, `#MachineLearning`, `#LLM`, `#MLCompilers`, `#RustLang`
**Meilleur moment :** mardi-jeudi, 9h-11h heure US.

---

## 4. Discord / Slack communautés

Copier la version courte du message (section 3) dans :
- **Hugging Face Discord** (`#community` ou `#ml-systems`)
- **MLOps.community Slack**
- **Rust AI / RustML Discord** (si existant — chercher "Rust Machine Learning Discord")
- **Serveur LLM engineering** (ex. "Latent Space")

---

## 5. Template de relance (après 5-7 jours sans réponse)

**Objet :** Re: Request for arXiv endorsement — NEURAX

```
Dear [Name],

Just following up on my previous message about endorsing me for arXiv (cs.LG)
for my paper on NEURAX, the open-source analytical compiler for neural
networks. I know requests like this are common, so I'll keep it brief:

The paper is archived with a DOI (Zenodo) and the project has been accepted in
8 awesome lists. The endorsement is a 2-minute action from your arXiv account.

If you're not comfortable endorsing, no problem at all — I appreciate the time
you've already given. If you have a moment, feedback on the project itself would
still be very welcome.

Best regards,
Martial
```

---

## 6. Liste de cibles (chercheurs / communautés à contacter)

> À personnaliser et compléter. Privilégier ceux qui travaillent sur : MLIR, IREE, XLA, TVM, compilateurs ML, modèles économiques du ML, hardware-aware ML.

| # | Cible | Canal | Statut | Date | Relance |
|---|-------|-------|--------|------|---------|
| 1 | Auteurs IREE / MLIR (Google) | Email labo | À faire | | |
| 2 | Auteurs ONNX Runtime (Microsoft) | Email labo | À faire | | |
| 3 | Chercheurs TVM / Apache (OctoML) | LinkedIn | À faire | | |
| 4 | Auteurs de papiers "scaling laws" récents (Meta AI) | Email | À faire | | |
| 5 | Auteurs Mamba / SSM (universités US) | Email | À faire | | |
| 6 | Auteurs FlashAttention / PagedAttention | Email | À faire | | |
| 7 | Communauté Hugging Face (maintainers) | Discord | À faire | | |
| 8 | Chercheurs Rust + ML (communauté rust-ml) | GitHub/Discord | À faire | | |
| 9 | Auteurs de papiers LLM efficiency (quantization, serving) | LinkedIn | À faire | | |
| 10 | Auteurs de surveys récents en cs.LG (2024-2026) | Email | À faire | | |

---

## 7. Rappel : que se passe-t-il après l'endorsement ?

1. L'endorseur reçoit une notification arXiv et clique "Endorse".
2. Vous soumettez le papier en `cs.LG` (le formulaire ne demande plus l'endorsement).
3. arXiv recompile `main.tex` → vérification "PDF preview" → soumission finale.
4. Réponse du modérateur de catégorie sous 24-48h.
5. L'endorsement vaut pour **toutes vos soumissions futures** dans cette catégorie.
