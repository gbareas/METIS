"""Tests for metis.config — DNS data-root resolution (milestone R1)."""
import pytest

from metis.config import DATA_ROOT_ENV_VAR, load_config, resolve_data_root


def test_cli_value_wins_over_everything(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path / "from_env"))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data:\n  root: /from/config\n")
    assert resolve_data_root(cli_value="/from/cli", config_path=cfg).as_posix() == "/from/cli"


def test_config_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path / "from_env"))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data:\n  root: /from/config\n")
    assert resolve_data_root(config_path=cfg).as_posix() == "/from/config"


def test_env_used_when_no_cli_or_config(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, "/from/env")
    empty = tmp_path / "empty.yaml"
    empty.write_text("runtime:\n  seed: 0\n")
    assert resolve_data_root(config_path=empty).as_posix() == "/from/env"


def test_raises_with_helpful_message_when_nothing_configured(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n")
    with pytest.raises(RuntimeError, match="--data-root"):
        resolve_data_root(config_path=empty)


def test_user_home_is_expanded(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    resolved = resolve_data_root(cli_value="~/dns_data")
    assert "~" not in resolved.as_posix()


def test_missing_explicit_config_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
