"""Tests for metis.registry.ArtifactRegistry (I8)."""
import json

import pytest

from metis.registry import ArtifactEntry, ArtifactRegistry


@pytest.fixture
def reg(tmp_path):
    return ArtifactRegistry(tmp_path / "artifacts")


def test_register_fills_provenance_and_persists(reg, tmp_path):
    e = reg.register("ds1", "dataset", metrics={"n_cases": 11})
    assert e.status == "experimental"
    assert e.created_at and e.git_commit and e.metis_version
    assert e.metrics["n_cases"] == 11
    # persisted immediately
    data = json.loads((tmp_path / "artifacts" / "registry.json").read_text())
    assert data["entries"][0]["id"] == "ds1"
    # a fresh registry sees it
    assert "ds1" in ArtifactRegistry(tmp_path / "artifacts")


def test_register_rejects_bad_kind_and_status(reg):
    with pytest.raises(ValueError, match="kind"):
        reg.register("x", "notakind")
    with pytest.raises(ValueError, match="status"):
        reg.register("x", "model", status="bogus")


def test_duplicate_id_raises(reg):
    reg.register("m1", "model")
    with pytest.raises(ValueError, match="already registered"):
        reg.register("m1", "model")


def test_list_filters_by_kind_and_status(reg):
    reg.register("d1", "dataset")
    reg.register("d2", "dataset", scope="s", status="validated")
    reg.register("m1", "model")
    assert [e.id for e in reg.list(kind="dataset")] == ["d1", "d2"]
    assert [e.id for e in reg.list(status="validated")] == ["d2"]
    assert len(reg.list()) == 3


def test_promote_requires_scope_then_succeeds(reg):
    reg.register("m1", "model")
    with pytest.raises(ValueError, match="scope"):
        reg.promote("m1")
    reg.update("m1", scope="reconstruct case10 xy_slice_1 RMS fields")
    assert reg.promote("m1").status == "validated"


def test_deprecate_and_set_status(reg):
    reg.register("m1", "model", scope="x")
    assert reg.deprecate("m1").status == "deprecated"
    assert reg.set_status("m1", "experimental").status == "experimental"


def test_update_rejects_immutable_fields(reg):
    reg.register("m1", "model")
    with pytest.raises(ValueError, match="cannot update"):
        reg.update("m1", id="m2")
    with pytest.raises(ValueError, match="cannot update"):
        reg.update("m1", created_at="2000-01-01")


def test_unknown_id_raises_keyerror_listing_known(reg):
    reg.register("m1", "model")
    with pytest.raises(KeyError, match="m1"):
        reg["nope"]


def test_entry_dataclass_validates_on_construction():
    with pytest.raises(ValueError):
        ArtifactEntry(id="x", kind="dataset", status="bad")
