def test_import():
    import sentinel
    assert sentinel.__version__ == "0.1.0"


def test_cli_main_runs(capsys, monkeypatch):
    import sys

    from sentinel.cli import main

    monkeypatch.setattr(sys, "argv", ["sentinel", "--help"])
    try:
        main()
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "sentinel" in captured.out.lower()
