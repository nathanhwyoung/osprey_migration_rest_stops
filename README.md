# OSPREY Migration Stopover Analysis

Automated detection of stopover sites used by ospreys (_Pandion haliaetus_) during fall and spring migration, reanalyzing a legacy Movebank satellite telemetry dataset (Martell & Douglas, 1995–2002).

---

## Research Questions

1. Can automated analysis detect discrete stopovers from coarse-resolution (date-only) satellite telemetry?
2. Do ospreys from different breeding populations share stopover sites during migration?

---

## Dataset

**Source:** Martell & Douglas (1995–2002) via Movebank  
**Tracking records:** 86,383 Argos Doppler-shift location fixes across 131 deployments  
**Files:**

- `Osprey in North and South America 1995-2002 (Martell).csv` — raw tracking data (19 MB, 47 columns)
- `Osprey in North and South America 1995-2002 (Martell)-reference-data.csv` — deployment metadata (131 records)

### Data Caveats

- **Argos positional error** ranges from ~150 m (Class 3) to several km (Class A/B/Z). Class Z fixes have no valid location and are excluded from analysis.
- **Date-only timestamps** prevent fine-grained step-speed filtering; analysis relies on NSD curves instead.
- **Tag redeployment issue:** 41 of 80 tags were redeployed on multiple birds in the original study. Analyses using tag ID alone blend trajectories across different individuals. All scripts from v2 onward use a composite `(tag_id, animal_id)` key to correctly separate deployments.

---

## Methods

### Stopover Detection

**Pipeline steps:**

1. Filter to `visible=TRUE` records only
2. Compute **Net Squared Displacement (NSD)** from each bird's breeding site over time
3. Identify flat segments in the NSD curve as stopover candidates (stationary periods)
4. Cluster candidate fixes spatially using **DBSCAN** (eps = 75 km, min_samples = 3, Haversine metric)
5. Filter out pre-departure staging at breeding sites
6. Flag stopover sites shared by 2+ birds across deployments

---

## Results

| Metric                           | Value                                       |
| -------------------------------- | ------------------------------------------- |
| Total stopovers detected         | 78                                          |
| Birds with at least one stopover | 58 of 126 deployments                       |
| Shared stopovers (2+ birds)      | 24 (31%)                                    |
| Highest-use site                 | Central Cuba — 4 birds, 3 years             |
| Second-highest site              | Northern Colombian coast — 4 birds, 4 years |

Stopover duration increases markedly at continental barriers: median ~4 days inland vs. ~12 days at coastal barriers.

---

## Dependencies

```
pandas
numpy
scikit-learn   # DBSCAN
```

Install with:

```bash
pip install pandas numpy scikit-learn
```

---

## Usage

# Run stopover detection

python detect_stopovers.py

```
## Citation

Raw data: Martell M, Douglas D. 2017. _Osprey in North and South America 1995–2002_. Movebank Data Repository.
```
