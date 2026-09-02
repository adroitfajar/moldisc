import pytest

torch = pytest.importorskip("torch")

from models.GPTGenerator import GPTGenerator


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, num_return_sequences, **kwargs):
        return torch.zeros((num_return_sequences, 1), dtype=torch.long)


class _FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0

    def batch_decode(self, output, skip_special_tokens=True):
        candidates = ["CCO", "OCC", "not_smiles", "CCN"]
        return candidates[: len(output)]


class _StructuralFilterTokenizer(_FakeTokenizer):
    def batch_decode(self, output, skip_special_tokens=True):
        candidates = ["O", "CO", "CCN", "CCC"]
        return candidates[: len(output)]


class _UpperBoundFilterTokenizer(_FakeTokenizer):
    def batch_decode(self, output, skip_special_tokens=True):
        candidates = ["CCCCCCCC", "CCN"]
        return candidates[: len(output)]


class _ElementFilterTokenizer(_FakeTokenizer):
    def batch_decode(self, output, skip_special_tokens=True):
        candidates = ["CC[SiH3]", "CCCl", "CCN"]
        return candidates[: len(output)]


def test_generation_excludes_known_and_reports_quality_metrics():
    generator = GPTGenerator(random_seed=7)
    generator.model = _FakeModel()
    generator.tokenizer = _FakeTokenizer()

    generated = generator.generate_smiles(
        target_num_samples=1,
        num_attempts=4,
        generation_batch_size=4,
        max_new_tokens=10,
        remove_ionics="all",
        excluded_smiles={"CCO"},
    )

    assert generated == ["CCN"]
    stats = generator.last_generation_stats
    assert stats["attempts"] == 4
    assert stats["valid_decodes"] == 3
    assert stats["unique_valid"] == 2
    assert stats["unique_known"] == 1
    assert stats["accepted"] == 1
    assert stats["yield_rate"] == 1.0


def test_generation_applies_configurable_organic_size_filters():
    generator = GPTGenerator(random_seed=9)
    generator.model = _FakeModel()
    generator.tokenizer = _StructuralFilterTokenizer()

    generated = generator.generate_smiles(
        target_num_samples=1,
        num_attempts=4,
        generation_batch_size=4,
        max_new_tokens=10,
        remove_ionics="all",
        min_carbon_atoms=2,
        min_heavy_atoms=3,
    )

    assert generated == ["CCN"]
    stats = generator.last_generation_stats
    assert stats["min_carbon_filtered"] == 2
    assert stats["min_heavy_atom_filtered"] == 0


def test_generation_applies_upper_chemical_domain_filters():
    generator = GPTGenerator(random_seed=11)
    generator.model = _FakeModel()
    generator.tokenizer = _UpperBoundFilterTokenizer()

    generated = generator.generate_smiles(
        target_num_samples=1,
        num_attempts=2,
        generation_batch_size=2,
        remove_ionics="all",
        max_heavy_atoms=5,
        max_molecular_weight=100,
        max_logp=3,
    )

    assert generated == ["CCN"]
    assert generator.last_generation_stats["max_heavy_atom_filtered"] == 1


def test_generation_applies_allowed_element_filter_and_reports_rejections():
    generator = GPTGenerator(random_seed=13)
    generator.model = _FakeModel()
    generator.tokenizer = _ElementFilterTokenizer()

    generated = generator.generate_smiles(
        target_num_samples=1,
        num_attempts=3,
        generation_batch_size=3,
        remove_ionics="all",
        allowed_elements=["C", "N"],
    )

    assert generated == ["CCN"]
    assert generator.last_generation_stats["element_filtered"] == 2
    assert generator.last_generation_stats["disallowed_element_counts"] == {
        "Cl": 1,
        "Si": 1,
    }
