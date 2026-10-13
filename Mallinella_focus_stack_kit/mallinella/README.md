# Mallinella synthétiques : focus stacks + profondeur vraie

Kit Blender pour générer des araignées *Mallinella* (Zodariidae) à l'échelle réelle et rendre, pour des orientations que vous choisissez, des **focus stacks macro physiquement cohérents** avec la **carte de profondeur exacte**. L'objectif est d'entraîner un modèle de depth-from-focus / depth estimation.

Testé avec Blender 5.2 (module `bpy`). Le code n'utilise que des API présentes depuis Blender 4.2. Tout est en **millimètres** (1 unité Blender = 1 mm).

## Contenu

| Fichier | Rôle |
|---|---|
| `specimens/*.blend` | 8 spécimens prêts : `sphaerica_f_001`, `sphaerica_m_002`, `pluma_m_003`, `random_004…008` |
| `mallinella_generator.py` | Générateur paramétrique (presets + variantes aléatoires, graine reproductible) |
| `render_focus_stack.py` | Rendu des focus stacks + vérité terrain, piloté par un JSON |
| `config_example.json` | Optique, pile, fond, éclairage et **liste des orientations** à modifier |
| `run_all.sh` | Rend tous les spécimens (et peut en générer de nouveaux) |
| `load_dataset.py` | Chargeur numpy/Pillow (pile, profondeurs de mise au point, profondeur, masque) |
| `sample_output/` | Exemple de sortie (1 vue, résolution réduite) |
| `tests/aperture_test.py` | Test qui valide la conversion de l'ouverture dans Cycles |

## Démarrage rapide

```bash
# 1. Rendre un spécimen avec les vues de la config
blender -b specimens/mallinella_sphaerica_m_002.blend -P render_focus_stack.py -- --config config_example.json

# Seulement certaines vues, ou un test rapide (¼ de résolution, 16 samples, ≤ 12 tranches)
blender -b specimens/mallinella_pluma_m_003.blend -P render_focus_stack.py -- --config config_example.json --views dorsal,frontal --quick

# 2. Générer de nouveaux spécimens
blender -b -P mallinella_generator.py -- --preset random --seed 42 --out specimens/mallinella_random_042.blend
#   presets : sphaerica_f, sphaerica_m, pluma_m, random ; --hair 0 = sans setae ; --hair 1.5 = plus velu

# 3. Tout rendre (en générant d'abord 20 nouveaux individus)
N_NEW=20 ./run_all.sh config_example.json
```

Sur GPU, mettez `"device": "OPTIX"` (ou `CUDA`, `HIP`, `METAL`) dans la config.

## Choisir les orientations

Chaque entrée de `views` définit une prise de vue. La caméra orbite autour du spécimen :

```json
{ "name": "oblique", "azimuth": 45, "elevation": 40, "roll": 0 }
```

- **`azimuth`** : 0 = face à la tête, 90 = flanc gauche de l'araignée (+X), 180 = arrière, 270 = flanc droit.
- **`elevation`** : 90 = vue dorsale (tête en haut de l'image), 0 = vue rasante, −90 = vue ventrale. Le substrat est masqué automatiquement sous 2°.
- **`roll`** : rotation de l'image autour de l'axe optique (°).
- **`magnification`** : grandissement (`"auto"` cadre tout le spécimen, pattes comprises, ou un nombre, ex. `2.5`). Avec `"center": "body"`, la vue est centrée sur le corps et les pattes peuvent sortir du cadre.
- **`specimen_rotation_deg`** : `[rx, ry, rz]` incline le spécimen lui-même, par exemple pour imiter un spécimen épinglé de travers.
- **`target_offset_mm`** : décale le point visé.

## Modèle optique

Le modèle est celui de la lentille mince, comme sur un banc macro réel.

- **`"mode": "lens"`** (objectif macro) : `focal_length_mm`, `f_number`, `magnification`.
  - Distance de travail : s = f(1 + 1/m).
  - Ouverture : D = f/N.
  - N effectif = N(1 + m).
  - Profondeur de champ par tranche : 2·N·c·(1 + m)/m², avec c = 2 pixels capteur par défaut (`coc_mm`).
- **`"mode": "objective"`** (objectif de microscope à l'infini) : `magnification`, `na`, `working_distance_mm`.
  - Profondeur de champ : λ/NA² + p/(m·NA).
- **`stack.mode`** :
  - `"rail"` : toute la caméra avance, comme un StackShot. Le grandissement est constant, avec un léger changement de perspective.
  - `"focus"` : la caméra est fixe et seule la distance de mise au point varie.
- **Pas de la pile** : `step_mm` explicite, sinon `overlap × DOF` (0,7 par défaut). La pile couvre la profondeur du spécimen ± `margin_mm`. Elle est plafonnée par `max_slices`.

**Détail technique important.** Cycles convertit le f-stop en ouverture en supposant 1 unité = 1 m, même quand l'échelle de scène vaut 0,001. Le script règle donc `fstop_blender = focale_blender[mm]·10⁻³ / D[mm]`. `tests/aperture_test.py` le vérifie : le flou mesuré est de 29 px pour 27,8 px attendus.

## Sorties (par spécimen et par vue)

```
renders/<specimen>/<vue>/
  slices/slice_000.png …   pile défocalisée (8 ou 16 bits, outputs.bit_depth)
  all_in_focus.png         rendu sténopé depuis la caméra de référence
  depth.npy                profondeur planaire (mm) le long de l'axe optique de référence, float32
  depth.exr                R = profondeur mm, G = index objet (1 = araignée, 2 = substrat), B = couverture
  depth_vis.png            visualisation 16 bits (proche = clair)
  mask.png                 araignée, setae comprises
  normals.npy              normales en repère caméra (float16)
  meta.json                intrinsèques (fx, fy, cx, cy en px), extrinsèques OpenCV (référence + chaque tranche),
                           optique, profondeur de mise au point de chaque tranche, taille du pixel objet
```

La profondeur vraie est rendue en un seul échantillon par pixel, au centre du pixel, sans antialiasing. Les valeurs ne sont donc jamais des moyennes entre avant-plan et arrière-plan, y compris sur les poils. J'ai vérifié qu'elle est planaire : le substrat donne exactement la hauteur de la caméra aux quatre coins.

## Anatomie des spécimens

Les modèles suivent les descriptions de *M. sphaerica* et *M. pluma* et le diagnostic du genre.

- **Céphalothorax** : ovale, allongé, brun-noir, granuleux, avec un sillon médian noir.
- **Yeux** : 8 yeux groupés en deux rangées procurvées. Les AME, les plus grands, sont sombres ; les yeux secondaires sont clairs.
- **Chélicères et pédipalpes** : présents ; les palpes des mâles ont un tarse un peu renflé.
- **Abdomen** : noir, avec 3 paires de taches blanches dorso-latérales (la 3ᵉ plus ou moins reliée) et une tache transverse postérieure. Les mâles portent un scutum brun-rouge brillant sur la moitié antérieure. La face ventrale est gris-noir avec 3 bandes blanches. Les filières sont jaune pâle.
- **Pattes** : longues, formule 4123, peu épineuses. Les hanches sont jaunâtres et les fémurs plus sombres.
- **Setae** : environ 15 000 à 40 000 poils de 0,1 à 0,2 mm. Leur couleur suit celle du tégument, donc ils sont blancs sur les taches. Ce détail fin est difficile pour un réseau de profondeur, ce qui est voulu.
- **Topologie** : un mesh en quads connecté, avec Subsurf, des UV et des groupes de vertex nommés pour un éventuel rig (`leg1.femur.L`, …).

## Limites connues

- Blender n'a pas de vraie lentille épaisse. La DOF est celle d'une lentille mince avec projection sténopé, sans « focus breathing » en mode `focus`, sans aberrations et sans diffraction. Pour plus de réalisme, ajoutez au moment de l'entraînement du bruit capteur, une compression JPEG et un léger flou de diffraction.
- Les profondeurs des tranches sont exactes. Un vrai rail a en plus une erreur de pas : vous pouvez la simuler en bruitant `focus_depth_mm`.
- Le spécimen est un adulte « au repos » sur un substrat. La posture des pattes est aléatoire, mais elle n'est pas animée.
