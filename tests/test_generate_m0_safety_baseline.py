import json

from generate_m0_safety_baseline import read_manifest


def test_read_manifest_requires_nonempty_records(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"records": []}))
    try:
        read_manifest(path)
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty M0 manifest should fail")


def test_read_manifest_preserves_frozen_order(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"records": [{"id": "b"}, {"id": "a"}]}))
    assert [row["id"] for row in read_manifest(path)] == ["b", "a"]
