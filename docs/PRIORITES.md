# 🎯 Priorités Projet Neurax Agent

## Légende
| Icône | Signification |
|-------|---------------|
| 🔴 **BLOCKANT** | Bloque le pipeline complet |
| 🟡 **IMPORTANT** | Fonctionnalité critique manquante |
| 🔵 **AMÉLIORATION** | Optimisation / qualité de vie |
| ⚪ **NICE TO HAVE** | Bonus une fois le reste solide |

---

## PRIORITÉ 1 — Pipeline intégral Frontend → Backend 🔴

### 1.1 — Tester le flux complet AIChatDrawer → Agent → Canvas → Rust

**Constat :** Le flux a été validé en isolé (agent→validation, agent→Rust séparément) mais JAMAIS en bout-en-bout via l'interface utilisateur.

**À faire :**
- Lancer l'UI (neurax-ui) avec `pnpm dev`
- Lancer l'agent FastAPI (port 8099)
- Taper un prompt → vérifier que les SSE events arrivent au canvas
- Vérifier que `compileToNeuraxIR()` produit un topologie valide
- Vérifier que `POST /analyze` (Rust, port 9098) accepte et analyse la topologie
- Vérifier que les résultats d'analyse s'affichent dans l'UI

**Critère de succès :** Une architecture générée depuis l'UI est analysée par le backend Rust sans erreur et les métriques s'affichent.

### 1.2 — Ajouter le type `multimodal` au parseur Rust

- **Fichier :** `neurax-parser/src/model_config.rs`, fonction `LayerType::from_str()`
- Le parseur rejette `"multimodal"` comme model_type → l'analyse Rust échoue
- Solution : mapper `"multimodal"` → `Self::Custom` ou un nouveau variant dédié
- **Test :** `curl POST /analyze` avec model_type="multimodal" doit retourner 200

---

## PRIORITÉ 2 — Catalogue & Validation 🟡

### 2.1 — Vérifier la cohérence des blocs dans TOUS les catalogues

**Constat :** L'agent GPT-2 a produit des blocs `mha`, `ffn`, `embedding`, `positional_encoding` qui EXISTENT dans le catalogue `transformer` mais certaines validations ont échoué (probablement un problème de chargement du catalogue dans `catalogue_store.py`).

**À faire :**
- Vérifier que `get_catalogue_for_family()` retourne tous les blocs de chaque famille
- Tester l'agent avec `get_catalogue_for_family('transformer')` pour GPT-2 → valider l'architecture
- Vérifier que `select_family` utilise `get_catalogue_for_family()` et non un sous-ensemble

### 2.2 — Catalogue multimodal : vérifier que les blocs `concat`/`merge` sont fan-in capable

**Constat :** `concat` et `merge` doivent avoir `maxInputs: -1` (fan-in illimité) pour la fusion de branches.

**À faire :**
- Vérifier `maxInputs` pour `concat`, `merge`, `add`, `residual` dans `multimodal`
- Tester avec 3+ branches parallèles pour confirmer

### 2.3 — Validateur `topology_validator.py` : Support de multiples inputs

**Constat :** `_has_io_path()` cherche un chemin depuis UN `input` vers UN `output`. Pour multimodal, il y a 2 inputs (`input_vision`, `input_text`).

**À faire :**
- Vérifier que les 2 inputs trouvent bien un chemin vers l'output
- Ajouter un test unitaire avec 2 inputs parallèles + concat + output

---

## PRIORITÉ 3 — Robustesse & Auto-Correction 🟡

### 3.1 — Améliorer le mécanisme de rétroaction

**Constat :** L'agent a auto-corrigé (output manquant → 2e tentative réussie) mais le mécanisme est basique.

**À faire :**
- Ajouter un message d'erreur détaillé dans le feedback au LLM
- Limiter à 3 tentatives max avec escalade vers l'utilisateur si échec
- Logger la trace complète des tentatives

### 3.2 — Gestion des erreurs API

**Constat :** Si `OPENAI_API_KEY` expire ou si le LLM est indisponible, l'agent plante sans message clair.

**À faire :**
- `except` sur `openai.APIConnectionError` / `openai.RateLimitError` / `openai.AuthenticationError`
- Messages utilisateur explicites (en français ?)
- Timeout configurable par phase (select_family, plan, exec)

### 3.3 — Healthcheck & Redémarrage des services

**Constat :** Le service Rust (neurax-service) tombe sans être relancé.

**À faire :**
- Ajouter un watchdog / `supervisord` dans le docker-compose
- Healthcheck endpoints (déjà présents sur le Rust, à ajouter sur l'agent FastAPI)
- Script `start.sh` qui lance les 3 services (Rust + Agent + UI)

---

## PRIORITÉ 4 — Tests 🔵

### 4.1 — Tests unitaires pour le pipeline agent

| Module | Tests à écrire |
|--------|---------------|
| `agent_runner.py` | Cycle de vie : start → select_family → plan → exec → done |
| `arch_planner.py` | Planification avec template CNN, transformer, multimodal |
| `topology_validator.py` | Graphes valides/invalides, cycles, fan-in, DAG |
| `snapshot_ops.py` | add_node, connect (cycle detect, fan-in), disconnect |
| `catalogue_store.py` | get_catalogue_for_family, get_family_constraints |

### 4.2 — Test de régression catalogue

- Vérifier que chaque famille a un template correspondant dans `FAMILY_TEMPLATES`
- Vérifier que tous les blocs référencés dans les templates existent dans les catalogues

### 4.3 — Test du parseur Rust

- Tests d'intégration avec tous les `model_type` supportés
- Tests de conversion snapshot → IR (équivalent de `compileToNeuraxIR` en Python)

---

## PRIORITÉ 5 — Documentation 🔵

### 5.1 — API Reference

- Documenter `POST /runs` (SSE events) avec exemples
- Documenter `POST /analyze` avec schéma de la topologie
- Documenter les codes d'erreur

### 5.2 — Architecture Document

- Diagramme des composants (Frontend → Agent → Catalogue → Validator → Rust Parser)
- Flux de données : prompt → SSE events → canvas → IR → analyze
- Comment ajouter une nouvelle famille au catalogue

---

## PRIORITÉ 6 — Déploiement & Infrastructure ⚪

### 6.1 — Docker Compose fonctionnel

**Constat :** `docker-compose.yml` existe mais les services (`neurax-agent`, `neurax-ui`, `neurax-service`) n'ont pas été testés ensemble récemment.

**À faire :**
- Tester `docker compose up` complet
- Vérifier les volumes, les ports, les dépendances entre services
- Ajouter une variable d'environnement `OPENAI_API_KEY` au service agent

### 6.2 — Scripts de démarrage

- `start-dev.sh` : lance les 3 services en mode développement
- `start-prod.sh` : lance via docker-compose
- `healthcheck.sh` : vérifie que les 3 endpoints répondent

---

## Roadmap recommandée

```
Semaine 1 : PRIORITÉ 1 (Pipeline complet)
  ├── Tester UI → Agent → Canvas → Rust
  ├── Ajouter multimodal au parseur Rust
  └── Corriger les bugs découverts

Semaine 2 : PRIORITÉ 2 + 3 (Catalogue + Robustesse)
  ├── Vérifier catalogues (toutes familles)
  ├── Améliorer auto-correction
  └── Ajouter healthcheck + error handling

Semaine 3 : PRIORITÉ 4 (Tests)
  ├── Tests unitaires agent
  ├── Tests intégration Rust
  └── Tests de régression catalogue

Semaine 4 : PRIORITÉ 5 + 6 (Documentation + Déploiement)
  ├── API docs
  ├── Architecture docs
  ├── Docker compose validé
  └── Scripts de démarrage
```

---

## Résumé des acquis (déjà fonctionnel)

| Composant | Statut |
|-----------|--------|
| Agent FastAPI (port 8099) | ✅ Fonctionnel |
| Agent pipeline (3 phases) | ✅ Validé |
| Topology validator | ✅ Valide CNN, ResNet, Multimodal |
| Catalogue CNN (23 blocs) | ✅ Complet |
| Catalogue transformer (17 blocs) | ✅ Complet |
| Catalogue multimodal (24 blocs) | ✅ Ajouté |
| Adaptation de template | ✅ Validé (ResNet → médical) |
| Analyse Rust (port 9098) | ✅ Opérationnel |
| Auto-correction (retry) | ✅ Fonctionnel (2-3 tentatives) |
| Service Rust backend | ✅ Compilé, en ligne |
