import voxflow


def test_version_is_nonempty_string():
    assert isinstance(voxflow.__version__, str)
    assert voxflow.__version__


def test_version_looks_semver():
    parts = voxflow.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])
