# Zenodo Deposit — Step-by-Step Checklist

> Objectif : obtenir un **DOI** pour le technical report NEURAX en quelques minutes, sans endorsement, avec une visibilité Google Scholar. Le fichier `paper/main.pdf` est prêt (8 pages, compilé sans warning).

---

## Étape 0 — Prérequis (2 min)

- [ ] Créer un compte Zenodo : https://zenodo.org → **"Sign up"** (login via GitHub ou email)
- [ ] (Optionnel mais fortement recommandé) Créer un ORCID : https://orcid.org — **2 min**, gratuit, donne de la crédibilité et est requis plus tard pour arXiv
- [ ] Avoir `paper/main.pdf` sous la main (déjà prêt dans le repo)

---

## Étape 1 — Nouveau dépôt

1. Aller sur https://zenodo.org → **"Upload"** (bouton en haut à droite)
2. Glisser-déposer `paper/main.pdf` dans la zone de dépôt
3. Le fichier doit s'afficher dans la liste (nom `main.pdf` — ou renommer en `neurax-paper.pdf` pour plus de clarté, Zenodo permet de renommer)

---

## Étape 2 — Métadonnées (copier-coller)

### Type de dépôt
| Champ | Valeur |
|-------|--------|
| Upload type | **Publication** |
| Publication type | **Preprint** |

### Informations de base
| Champ | Valeur |
|-------|--------|
| **Title** | `NEURAX: An Analytical Compiler for Neural Network Architectures` |
| **Authors** | `Fossouo, Martial` — affiliation : `Independent Researcher` (cliquer "Add" puis compléter) |
| **Description** | Copier l'abstract ci-dessous |

**Description** (copier-coller dans le champ, Zenodo accepte le Markdown) :

```
Training modern deep learning models is expensive and unpredictable. Practitioners routinely commit GPU resources to architectures whose memory footprint, training cost, and runtime behavior are unknown until training begins, leading to wasted compute, out-of-memory failures, and slow iteration cycles.

We present **NEURAX**, an analytical compiler for neural network architectures that predicts training cost, peak memory usage, latency, and inference behavior *before* training, in under 50 milliseconds and with zero GPU requirement. NEURAX operates at design time through an eleven-phase intermediate representation (IR) pipeline composed of ten analytical dialects. The system supports ten architecture families (Transformer, CNN, MoE, Diffusion, GNN, RNN, SSM, GAN, Hybrid, and Multimodal) with 680+ configurable blocks and 88 reference templates.

Analytical predictions are validated against real training runs with a reported average accuracy of 99%+ across validated families. NEURAX is implemented in Rust with an MLIR/LLVM 18 backend and is released under the MIT license.

The latest version and source code are available at https://github.com/rustnew/NEURAX
```

### Licence et accès
| Champ | Valeur |
|-------|--------|
| **Access right** | **Open Access** |
| **License** | **Creative Commons Attribution 4.0 International (CC BY 4.0)** — recommandé pour un preprint scientifique ; alternativement `MIT` si tu préfères |

### Champ additionnel : keywords
```
neural networks; compiler; MLIR; LLVM; Rust; deep learning; cost prediction; memory estimation; LLM training
```

### Champ : version
| Champ | Valeur |
|-------|--------|
| **Version** | `1.0.0` |

### Champ : language
| Champ | Valeur |
|-------|--------|
| **Language** | `English` |

---

## Étape 3 — Vérification et publication

- [ ] Cliquer **"Save"** → le brouillon est créé
- [ ] Cliquer **"Preview"** pour vérifier l'aperçu
- [ ] Cliquer **"Publish"** (définitif — le DOI est créé immédiatement)

**Résultat** : tu obtiens un DOI au format `10.5281/zenodo.XXXXXXX` + une page publique avec badge DOI.

---

## Étape 4 — Après publication (actions de visibilité)

- [ ] Copier le **badge DOI** (bouton "Badges" sur la page) et l'ajouter au README de NEURAX
- [ ] Ajouter le DOI dans `docs/LINUX_FOUNDATION_APPLICATION.md` et `GO_TO_MARKET_STRATEGY.md` comme preuve de publication
- [ ] Le preprint est indexé par Google Scholar automatiquement (quelques semaines)
- [ ] **Utiliser le lien DOI + la page publique comme preuve de sérieux dans les demandes d'endorsement arXiv** (Voie 2)

---

## Note : `.zenodo.json` (intégration GitHub, optionnel mais utile)

Le fichier `.zenodo.json` créé à la racine du repo permet à Zenodo de créer un DOI **à chaque release GitHub** (fonctionnalité "GitHub integration"). Ce n'est pas nécessaire pour le dépôt manuel du papier, mais :

1. Va sur https://zenodo.org/account/settings/github/
2. Connecte ton compte GitHub
3. Active l'intégration pour le repo `rustnew/NEURAX`

À chaque `git tag v1.x.x` + release GitHub, Zenodo archivera le code source et générera un DOI de code (complémentaire au DOI du papier).

---

## Résumé des identifiants qui seront générés

| Objet | Identifiant | Usage |
|-------|-------------|-------|
| Papier | DOI Zenodo `10.5281/...` | Citation, preuve de publication |
| Code source | DOI par release GitHub | Archivage du code |
| ORCID | ID auteur | Profil auteur, requis pour arXiv |
