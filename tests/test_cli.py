"""Tests for CLI smoke tests."""

from typer.testing import CliRunner

from reflex.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Deploy any VLA" in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_targets():
    result = runner.invoke(app, ["targets"])
    assert result.exit_code == 0
    assert "orin-nano" in result.output
    assert "Jetson Thor" in result.output


def test_export_help():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "HuggingFace model ID" in result.output


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "inference server" in result.output.lower() or "POST /act" in result.output


def test_serve_missing_dir():
    result = runner.invoke(app, ["serve", "/nonexistent/path"])
    assert result.exit_code == 1
