def test_import():
    import sentinel
    assert sentinel.__version__ == "0.1.0"


def test_cli_main_runs(capsys):
    from sentinel.cli import main
    main()
    captured = capsys.readouterr()
    assert "sentinel" in captured.out.lower()
