def test_package_exposes_version():
    import budgettrace

    assert budgettrace.__version__ == "0.1.0"
