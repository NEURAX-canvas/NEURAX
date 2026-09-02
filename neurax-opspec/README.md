# neurax-opspec

**OpSpec-IR — la spécification unique de chaque opération NEURAX.**

> Statut : construit et intégré. La famille des blocs CNN (10 types) est
> migrée, testée, branchée dans `neurax-ir`, `cargo build --workspace` et
> `cargo test --workspace` passent. Le reste de ce document explique le
> raisonnement — pourquoi ce crate existe, sur quoi il s'appuie, ce qui
> reste à faire.

---

## Le problème qui a motivé son existence

Le 1er et 2 septembre 2026, en corrigeant des bugs réels dans NEURAX
(formules CNN non câblées, S4/H3 réutilisant la forme de Mamba, RNN ignorant
`input_size`, GNN retombant sur un placeholder générique), un même constat
est revenu à chaque fois : **le même `LayerType` était défini trois fois,
indépendamment, dans trois fichiers séparés du crate `neurax-ir`** :

| Propriété | Fichier | Taille |
|---|---|---|
| Nombre de paramètres | `neurax-ir/src/architecture/mod.rs` | ~1100 lignes |
| FLOPs | `neurax-ir/src/operator/pass.rs` | ~1400 lignes |
| Forme de sortie | `neurax-ir/src/tensor/shape_inference.rs` | ~250 lignes |

Rien ne liait ces trois définitions entre elles. Un type pouvait avoir des
paramètres corrects et des FLOPs placeholder pendant des mois sans qu'aucun
mécanisme ne le signale — c'est exactement ce qui s'est produit pour six
types de blocs CNN, découvert seulement par relecture manuelle du code.

**OpSpec-IR existe pour rendre ce type de bug structurellement impossible**,
pas seulement détectable par audit.

---

## Rôle

OpSpec-IR n'est **pas** une représentation intermédiaire au sens
traditionnel du terme, et ne sera jamais une étape de plus dans le pipeline
d'analyse de NEURAX (`ArchitectureIR → GraphIR → TensorIR → OperatorIR →
ComputeIR → MemoryIR → ParallelismIR → HardwareIR → CostIR → ReportIR`).
C'est une **table de spécifications, consultée par ces étapes** — une
donnée statique, pas un flux qui les traverse.

NEURAX n'est pas un compilateur qui génère du code exécutable — cette IR
reste fidèle à ce principe : **aucun parseur, aucun format texte, aucune
cible de génération de code, aucun moteur de réécriture de graphe.** Elle ne
fait qu'une chose : garantir qu'un type d'opération se définit à un seul
endroit, avec sa forme, ses paramètres et ses FLOPs nés ensemble.

## Sur quels concepts elle s'appuie — sources vérifiées

Trois projets open source ont été étudiés directement (code source et
documentation officielle) avant de concevoir ce crate :

### ONNX — l'inspiration principale

Chaque opérateur ONNX se déclare **une seule fois**, via
`ONNX_OPERATOR_SET_SCHEMA(Nom, depuis_version, schema)`, avec sa fonction
d'inférence de forme attachée dans le même appel
(`.TypeAndShapeInferenceFunction(...)`). Le schéma qui en résulte est un
objet **interrogeable par programme** (`onnx.defs.get_schema()`), pas
seulement de la documentation. Versionné par snapshots immuables : modifier
un opérateur archive d'abord l'ancienne définition dans `old.cc`, si bien
qu'un graphe déjà sauvegardé continue de résoudre contre le schéma qui
était vrai au moment de sa création.

C'est le seul des projets étudiés avec un schéma véritablement
machine-lisible — c'est le patron directement repris ici : un seul point
d'enregistrement par type, regroupant forme et coût.

Sources : [ONNX AddNewOp](https://onnx.ai/onnx/repo-docs/AddNewOp.html),
[onnx.defs API](https://onnx.ai/onnx/api/defs.html),
[Versioning.md](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)

### StableHLO — la décomposition comme donnée

StableHLO définit un opérateur `composite` de première classe, dont la
sémantique vient d'un attribut `decomposition` — l'opérateur composite peut
être remplacé par sa décomposition sans changer le sens du programme. C'est
formaliser ce que `decompose_layer_to_ops`
(`neurax-ir/src/operator/pass.rs`) fait déjà chez NEURAX de façon
impérative : un futur champ `decomposition` sur `OpSpec` rendra cette
décomposition interrogeable comme donnée plutôt que cachée dans du code.

StableHLO garantit aussi une compatibilité stricte (5 ans arrière, 2 ans
avant) pour ses artefacts sérialisés, via un dialecte versionné compagnon
(VHLO) où un numéro de version d'opérateur n'augmente que si son
comportement change réellement. Point de vigilance retenu : la spec de
StableHLO elle-même n'est que du prose/markdown, pas un schéma
machine-lisible — contrairement à ONNX, ce n'est donc pas le modèle à
suivre pour la partie "interrogeable par programme".

Sources : [StableHLO spec](https://openxla.org/stablehlo/spec),
[Compatibility RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20230623-compatibility.md)

### MLIR — la philosophie de vérification

Un opérateur MLIR est identité + opérandes + résultats + attributs +
régions. La vérification structurelle passe avant les vérificateurs
sémantiques propres à chaque opérateur ; l'inférence de type est optionnelle
via une interface (`InferTypeOpInterface`) — le résultat se déduit des
opérandes plutôt que d'être redéclaré à chaque site d'appel. Cette
philosophie (vérifier plutôt que faire confiance) motive un futur champ
`verify_fn` sur `OpSpec`, pas encore construit.

Source : [MLIR LangRef](https://mlir.llvm.org/docs/LangRef/)

### Ce qui n'a pas été retenu, et pourquoi

- **IREE (Stream/HAL)** — sépare "quoi calculer" de "comment/où
  l'ordonnancer". NEURAX a déjà cette même coupure structurelle
  (`ParallelismIR` → `HardwareIR`) ; la leçon confirme que le pipeline
  existant est déjà sain, elle ne s'applique pas à OpSpec-IR lui-même.
- **LMHLO** — la forme "bufferisée" de MHLO (valeurs → buffers avec
  liveness). NEURAX a déjà cette coupure aussi (`TensorIR` → `MemoryIR`).
  Même conclusion : validant, pas actionnable ici.
- **Le patron SSA complet d'un vrai compilateur (LIFT, MLIR)** — régions,
  contrôle de flux, réécriture de graphe. Délibérément exclu : les graphes
  de NEURAX sont des DAG d'opérations atomiques décrivant une architecture,
  pas des programmes avec des boucles, et NEURAX ne réécrit jamais le
  design d'un utilisateur (voir `DiagnosticCode::W008` : on signale, on ne
  corrige jamais silencieusement).

---

## Architecture

```
neurax-opspec/
├── README.md       // ce document
├── Cargo.toml      // deps : neurax-parser + neurax-formulas (serde_json en dev-dependency, tests GNN)
└── src/
    ├── lib.rs       // OpSpec, ParamsFn, FlopsFn, FlopsContext, ActivationMemoryFn, ré-export de op_spec()/attention_rope_flops
    └── registry.rs  // la table statique, un OpSpec par LayerType migré (Custom excepté)
```

```rust
pub struct FlopsContext<'a> {
    pub global_params: &'a GlobalParams,       // GNN : num_nodes/num_edges
    pub image_channels: Option<usize>,         // Conv : repli quand pas d'input_shape
    pub image_height: Option<usize>,
    pub image_width: Option<usize>,
}

pub struct OpSpec {
    pub layer_type: LayerType,
    pub params_fn: fn(&Layer) -> u64,
    pub flops_fn: fn(&Layer, usize, usize, &FlopsContext) -> f64,   // layer, batch, seq, ctx
    pub activation_memory_fn: Option<fn(&Layer, usize, usize, &str) -> u64>, // layer, batch, seq, dtype
}

pub fn op_spec(layer_type: LayerType) -> Option<&'static OpSpec>;
```

`op_spec()` retourne `None` uniquement pour `Custom` (exception permanente,
voir plus haut) — c'est ce qui garde la migration sûre : un type absent de
la table continuerait de résoudre exactement comme avant, dans les anciens
fichiers de `neurax-ir`, s'il existait encore un tel type.

`FlopsFn` a été étendu deux fois pendant la migration complète, exactement
comme le prémortem l'anticipait (« ne pas figer la signature avant une
deuxième famille structurellement différente ») :

- `&LayerParams` → `&Layer` : `Dense`/LoRA/DoRA ont besoin de la vraie
  `input_shape`/`output_shape` de la couche, pas seulement de ses
  hyperparamètres.
- Ajout de `&FlopsContext` : GNN lit `num_nodes`/`num_edges` depuis
  `global_params.extra` (rien sur la couche elle-même ne les porte), et
  `Conv` retombe sur `data.image_channels/height/width` quand aucune
  `input_shape` à 4 dimensions n'est connue.

Un troisième axe, `activation_memory_fn`, a été ajouté en `Option` plutôt
que dans `FlopsFn` : seuls six types (`Embedding`, `Attention`, `Mlp`,
`Dense`, `LoraLinear`, `DoraLinear`) suivaient un vrai coût d'activation
avant la migration — tous les autres retournaient `0` en dur. `None`
reproduit exactement ce `0` sans forcer chaque entrée à fournir une
fermeture inutile.

## Comment ça s'intègre à la structure actuelle

Deux points d'entrée dans `neurax-ir` consultent la table — `Custom` est
seul à garder sa logique historique en filet de repli :

```rust
// neurax-ir/src/architecture/mod.rs::calculate_layer_params
pub fn calculate_layer_params(layer: &Layer) -> u64 {
    if let Some(spec) = neurax_opspec::op_spec(layer.layer_type) {
        return (spec.params_fn)(layer);
    }
    match layer.layer_type { /* Custom uniquement */ }
}
```

```rust
// neurax-ir/src/operator/pass.rs::decompose_layer_to_ops
fn decompose_layer_to_ops(layer: &LayerDef, batch: usize, seq: usize, dtype: &str, ctx: &NeuraxContext) -> Vec<AtomOp> {
    if let Some(spec) = neurax_opspec::op_spec(layer.layer_type) {
        let parser_layer = to_parser_layer(layer);
        let flops_ctx = neurax_opspec::FlopsContext { /* depuis ctx.config */ };
        let flops = (spec.flops_fn)(&parser_layer, batch, seq, &flops_ctx);
        let activation_memory = spec.activation_memory_fn
            .map(|f| f(&parser_layer, batch, seq, dtype))
            .unwrap_or(0);
        // OpType reste décidé ici, côté neurax-ir (op_type_for()), qui en
        // est le seul propriétaire : ce crate ne dépend d'aucun type de
        // neurax-ir.
        let mut ops = vec![AtomOp { op_type: op_type_for(&parser_layer), flops, activation_memory, ... }];
        // Attention est la seule exception à "un type migré = un AtomOp" :
        // RoPE reste un second op réel, voir plus bas.
        if layer.layer_type == LayerType::Attention {
            ops.push(/* AtomOp RoPE, via neurax_opspec::attention_rope_flops */);
        }
        return ops;
    }
    match layer.layer_type { /* Custom uniquement */ }
}
```

Pour chaque type migré, l'ancien bras de `match` correspondant a été
**supprimé**, remplacé par un bras `unreachable!()` explicite documentant
pourquoi il ne devrait jamais s'exécuter — jamais laissé en double (code
mort interdit dans ce projet).

`tensor/shape_inference.rs` n'est toujours pas branché : aucun type migré
n'a de forme de sortie propre aujourd'hui (ils retombent sur un
passthrough) — un `shape_fn` rejoindrait `OpSpec` si une famille en avait
réellement besoin, ce qu'aucune n'a exigé pendant cette migration.

### État de la migration

**Migration complète — chaque `LayerType` réel, sauf `Custom`, est dans la
table.**

| Famille | Types |
|---|---|
| Transformer | `Embedding`, `Attention`, `Mlp`, `Dense`, `LoraLinear`, `DoraLinear`, `Normalization`, `Conv`, `Pooling` |
| Blocs CNN | `ResidualBlock`, `ResnetBottleneck`, `Mbconv`, `Inception`, `DenseBlock`, `ConvnextBlock`, `ShuffleUnit`, `C2f`, `Detection`, `Transition` |
| MoE | `MoE`, `MoeRouter`, `MoeCombine`, `MoeSharedExpert` |
| SSM | `MambaBlock`, `S4Block`, `StateSpace`, `H3Block`, `RwkvBlock`, `RetentionBlock` |
| GAN | `GeneratorBlock`, `DiscriminatorBlock`, `ProgressiveBlock`, `SelfAttention`, `StyleMod`, `AdaIN`, `PixelNorm`, `MinibatchStd`, `SpectralNorm` |
| RNN | `LstmBlock`, `GruBlock`, `RnnCell`, `Bidirectional`, `EncoderBlock`, `DecoderBlock` |
| Diffusion | `UnetBlock`, `ResnetBlock`, `TimeEmbedding`, `TimestepBlock`, `CrossAttention`, `DownBlock`, `UpBlock`, `MidBlock`, `ConditionBlock`, `NoisePredictor`, `VaeEncoder`, `VaeDecoder` |
| GNN | `GraphConvNet`, `MessagePassing`, `GraphAttentionNet`, `RgcnConv` |
| — | `Custom` : exception permanente (évalue une équation utilisateur, rapporte des diagnostics `neurax-ir` — hors du périmètre de ce crate) |

### Bugs réels trouvés pendant la migration

- **`ResidualBlock` / `ResnetBottleneck`** (CNN, première vague). Un seul
  bras de `match` partagé côté FLOPs utilisait `out_channels.unwrap_or(256)`
  — correct pour `ResnetBottleneck`, faux pour `ResidualBlock` seul (dont le
  côté paramètres utilise `unwrap_or(64)`). Corrigé en séparant les deux en
  entrées distinctes avec leurs propres valeurs par défaut.
- **`Conv`** (migration complète). Le bras FLOPs pré-migration dérivait
  `kernel_w` de `kernel_h` — lui-même toujours égal à `kernel_size`, sans
  jamais lire `params.kernel_h` — alors que le bras paramètres lisait
  `kernel_h`/`kernel_w` directement. Une couche `Conv` avec un noyau
  explicitement non carré (`kernel_h=7, kernel_w=7, kernel_size` absent)
  était donc silencieusement coûtée comme un noyau `(3, 7)`. Corrigé en
  faisant lire aux deux formules exactement les deux mêmes champs, dans le
  même ordre — voir `conv_flops_reads_kernel_h_and_kernel_w_not_just_kernel_size`
  dans `registry.rs` et `conv_flops_reads_the_fixed_kernel_shape` dans le
  test avant/après de `neurax-ir`.

Une approximation préexistante a été **délibérément préservée**, pas
« corrigée » : `RwkvBlock`/`RetentionBlock` partagent la formule FLOPs de
`MambaBlock` (`neurax_formulas::ssm` n'a pas encore de `rwkv_flops`/
`retention_flops` propres), alors qu'ils ont chacun leur vraie formule de
paramètres. Inventer une formule FLOPs correcte pour RWKV/RetNet est un
vrai chantier de `neurax-formulas`, pas quelque chose que cette migration
devait trancher silencieusement.

### Test avant/après

`neurax-ir/tests/opspec_full_migration_before_after.rs` fixe, famille par
famille, les valeurs (params, FLOPs) obtenues pour une couche représentative
de chaque type migré. Ces valeurs ont été capturées depuis le code
pré-migration (en isolant temporairement `op_spec()` aux seuls blocs CNN, le
temps de la capture) puis comparées octet pour octet aux valeurs
post-migration : identiques partout, sauf `Conv` — la seule différence
attendue, documentée ci-dessus.

---

## Risques identifiés par prémortem, et ce qui en est advenu

*(Exercice mené avant d'écrire du code : « on est dans 12 mois, ce chantier
a échoué — pourquoi ? » La colonne "Advenu" est remplie a posteriori,
migration complète.)*

| Risque | Réponse proposée | Advenu |
|---|---|---|
| Migration partielle éternelle (deux systèmes cohabitent indéfiniment) | Chaque famille migrée indépendamment, testée et committée séparément — jamais un big-bang. | ✅ Résolu — les 9 familles (Transformer, CNN, MoE, SSM, GAN, RNN, Diffusion, GNN) sont toutes migrées ; `Custom` reste seul, par conception permanente. |
| La signature ne passe pas à l'échelle dès qu'une famille non uniforme arrive | Ne pas figer `FlopsFn` avant une deuxième famille structurellement différente. | ✅ Confirmé nécessaire, deux fois : `&LayerParams` → `&Layer` (Dense/LoRA ont besoin d'`input_shape`/`output_shape`) puis ajout de `FlopsContext` (GNN a besoin de `global_params.extra`, `Conv` de `data.image_*`). Un troisième axe (`activation_memory_fn`) a aussi dû être ajouté, en `Option` — voir "Comment ça s'intègre". |
| Le registre devient lui-même une source obsolète | Anciens bras de `match` supprimés, pas laissés en doublon — `unreachable!()`. | ✅ Fait pour les deux fichiers (`architecture/mod.rs`, `operator/pass.rs`) sur toute la migration. |
| Faux sentiment de sécurité — centraliser élimine le signal de désaccord qui trouvait des bugs par accident | Garder le vérificateur natif indépendant en place, non couplé. | ✅ `neurax-core/tests/native_flops_cross_check.rs` inchangé. Le signal a d'ailleurs encore fonctionné une fois pendant cette migration : le bug `Conv` (kernel_h/kernel_w) a été trouvé précisément en confrontant les deux anciens bras l'un à l'autre. |
| Effort de migration sous-estimé | Déjà confirmé pendant le prototypage CNN. | ✅ Confirmé une seconde fois : le cas `Attention` (voir "Comment ça s'intègre") a exigé qu'on revienne sur la simplification "un type migré = un `AtomOp`" après qu'un test de régression existant (`kernel_launch_count_reflects_real_ops.rs`) a révélé qu'elle changeait un vrai comportement (le nombre de lancements de kernel, qui alimente un modèle de latence réel). |
| Rien n'empêche la régression humaine (nouveau type ajouté hors du registre) | Différé jusqu'à deux familles migrées. | ✅ Résolu — `every_real_layer_type_except_custom_is_registered` (dans `registry.rs`) échoue si un futur `LayerType` est ajouté au parseur sans être enregistré ici. |
| Nouvelle dépendance inter-crates mal calibrée | Sens de dépendance fixé dès la conception (`neurax-opspec` en amont, jamais l'inverse). | ✅ Tenu sur toute la migration — `neurax-opspec` dépend uniquement de `neurax-parser` + `neurax-formulas` (+ `serde_json` en dev-dependency pour les tests GNN). |

---

## Ce que ce crate ne fera jamais

- Générer du code exécutable ou une IR textuelle.
- Réécrire le design d'un utilisateur (NEURAX signale, il ne corrige jamais
  silencieusement — voir `DiagnosticCode::W008`).
- Remplacer le pipeline en 10 passes de `neurax-ir` — il reste consulté PAR
  lui, jamais une étape de plus dedans.
- Décider d'un `OpType` ou construire un `AtomOp` — ce sont des concepts
  `neurax-ir`, décidés à l'endroit où ils sont consommés (`op_type_for()`
  dans `operator/pass.rs`).

## License

Propriétaire — logiciel fermé, commercial. Tous droits réservés. Non
publié sur crates.io (`publish = false`).
