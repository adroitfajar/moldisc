# Small-data demonstration datasets

The active MolDisc examples intentionally use **100 labelled molecules and
1,000 unlabelled molecules per task**. They were selected reproducibly from the
larger user-supplied tables derived from datasets distributed through the
MoleculeNet/DeepChem collection. These compact inputs are used in the article
calculations and the public tutorial notebooks.

MolDisc's MIT software license applies to the software, not to the molecular
records in these CSV files. The bundled records remain subject to applicable
upstream dataset terms and citation requirements; the DeepChem download
endpoints do not by themselves establish one uniform license for all five
underlying collections. Review the source-specific notes below before reuse,
commercial use, or redistribution.

## `data_1`: aqueous-solubility regression

- `labeled.csv`: 100 canonical molecules with measured aqueous solubility in
  log10(mol/L). Ten molecules were sampled from each of ten target-rank bins
  after canonical duplicates were aggregated by their mean.
- `unlabeled.csv`: 1,000 canonical structures selected proportionally from
  ESOL, FreeSolv, and Lipophilicity (186, 55, and 759 molecules,
  respectively). Property columns were withheld.

## `data_2`: BACE-1 activity classification

- `labeled.csv`: 100 canonical BACE molecules, sampled in proportion to the
  parent class distribution (54 inactive and 46 active; `1` is the inhibitor
  class).
- `unlabeled.csv`: 1,000 canonical structures selected proportionally from
  BACE and Tox21 (159 and 841 molecules, respectively). Before sampling, this
  pool was restricted to one connected component and the disclosed atom set
  B, C, N, O, F, Si, P, S, Cl, Br, and I. These two pre-sampling checks match
  the corresponding fragment and allowed-element constraints in the paper
  campaign. Activity/property columns were withheld.

## Reproducible selection

The selector uses seed 42 with NumPy's PCG64 generator. Structures are
sanitized and converted to canonical isomeric SMILES before deduplication.
Labelled regression examples are target-rank-stratified; labelled
classification examples are class-stratified; unlabelled examples are
source-stratified. Only the selected 100 labelled structures are excluded from
each unlabelled pool. Thus, molecules with measurements in the parent datasets
may appear in the unlabelled input, but those measurements are not used by
MolDisc. For `data_2`, the chemistry-domain filter excluded 244 disconnected
structures and 104 additional structures containing atoms outside the stated
set before source-proportional sampling.

Run the following command to rebuild the active files only when the preserved
full tables are already available locally under `data/full_source/`. The
selector fails before changing active files if those full tables or their
expected row counts are unavailable:

```bash
python scripts/select_small_data.py
```

`subset_manifest.json` records the source hashes, cleaning audit, quotas,
counts, target/class summaries, scaffold counts, and hashes of all four active
CSVs. The full source tables are retained locally for audit but excluded from
the public Git repository.

Original source endpoints:

- ESOL: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv`
- FreeSolv: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv`
- Lipophilicity: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv`
- BACE: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv`
- Tox21: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`

## Dataset citations and terms

Please cite MoleculeNet for the curated benchmark collection and the original
source appropriate to each record pool:

- MoleculeNet: Wu *et al.*, *Chemical Science* **9** (2018) 513–530,
  [doi:10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A).
- ESOL: Delaney, *Journal of Chemical Information and Computer Sciences* **44**
  (2004) 1000–1005,
  [doi:10.1021/ci034243x](https://doi.org/10.1021/ci034243x).
- FreeSolv: Mobley and Guthrie, *Journal of Computer-Aided Molecular Design*
  **28** (2014) 711–720,
  [doi:10.1007/s10822-014-9747-x](https://doi.org/10.1007/s10822-014-9747-x).
  The [upstream FreeSolv repository](https://github.com/MobleyLab/FreeSolv#license)
  applies CC BY 4.0 to data to the extent possible and explicitly notes that
  source-derived records may retain other restrictions.
- Lipophilicity: Hersey, *ChEMBL Deposited Data Set – AZ dataset* (2015),
  [doi:10.6019/CHEMBL3301361](https://doi.org/10.6019/CHEMBL3301361).
- BACE: Subramanian *et al.*, *Journal of Chemical Information and Modeling*
  **56** (2016) 1936–1949,
  [doi:10.1021/acs.jcim.6b00290](https://doi.org/10.1021/acs.jcim.6b00290).
- Tox21: Huang *et al.*, *Frontiers in Environmental Science* **3** (2016) 85,
  [doi:10.3389/fenvs.2015.00085](https://doi.org/10.3389/fenvs.2015.00085),
  and the [NCATS challenge data page](https://tripod.nih.gov/tox21/challenge/data.jsp).

No single explicit data license was identified that covers every upstream
collection. Repository maintainers should resolve any venue- or
institution-specific redistribution requirement before publishing these CSV
extracts. Where a single permissive data license is mandatory, distribute the
selection script, source URLs, and hashes and require local reconstruction
instead of implying that the software's MIT license covers the data.
