def test_import_module() -> None:
    import abinslib

    assert isinstance(abinslib.__version__, str)
