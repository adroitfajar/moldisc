# Contributing to MolDisc

Thank you for helping improve MolDisc. Changes should be scientifically traceable and should not weaken data or stopping-rule safeguards.

## Development setup

Create the two environments described in `README.md`, then install their development requirements:

```powershell
conda activate moldisc_main
python -m pip install -r requirements-dev.txt
python -m pytest

conda activate subGPT
python -m pip install -r requirementsGPT-dev.txt
python -m pytest tests/test_gpt_generation.py
```

## Pull requests

1. Explain the scientific or engineering motivation.
2. Add or update tests for behavior changes.
3. Preserve backward compatibility or document the migration.
4. Record any change that affects validity, uniqueness, novelty, property selection, or stopping counts.
5. Do not commit pretrained models, generated campaigns, private data, credentials, or notebook caches.
6. Update the README and notebooks when the public interface or citation changes.

For model or dataset changes, include data hashes, random seeds, split methodology, software versions, and the metrics needed to reproduce the claim.
