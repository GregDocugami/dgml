# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`Workspace.open(configuration=…)`: no create step, meta written once, never re-identified."""

from __future__ import annotations

import gc
import logging
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from dgml_core import DocSetStore, Workspace
from dgml_core.configuration import (
    Configuration,
    Identity,
    Model,
    Models,
    ProviderSpec,
    Storage,
)
from dgml_core.errors import ConflictError, InvalidArgument
from dgml_core.migrations import WORKSPACE_SCHEMA_VERSION, workspace_schema_version
from dgml_core.storage import user_config_path
from dgml_core.storage_local import LocalStore
from dgml_core.storage_service import BlobStore, DocStore, StorageConfig

_LOCAL = Storage.combined(ProviderSpec("dgml_core.storage_local:LocalStore"))


def _cfg(workspace_id: str = "tenant-1") -> Configuration:
    return Configuration.build(
        identity=Identity(workspace_id=workspace_id, name="Tenant", organization="TenantOrg"),
        storage=_LOCAL,
        models=Models(advanced=Model("cfg/advanced")),
    )


def test_first_open_writes_meta_and_stamps_version(tmp_path: Path) -> None:
    ws = Workspace.open(configuration=_cfg(), root=tmp_path)
    assert ws.read_meta() == {
        "name": "Tenant",
        "organization": "TenantOrg",
        "workspace_id": "tenant-1",
        "schema_version": WORKSPACE_SCHEMA_VERSION,
    }
    assert workspace_schema_version(ws) == WORKSPACE_SCHEMA_VERSION
    assert ws.is_initialized()
    assert ws.organization == "TenantOrg" and ws.display_name == "Tenant"
    assert ws.workspace_id == "tenant-1"


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    Workspace.open(configuration=_cfg(), root=tmp_path)
    again = Workspace.open(configuration=_cfg(), root=tmp_path)
    assert again.read_meta()["workspace_id"] == "tenant-1"


def test_another_workspace_id_on_the_same_backend_is_a_conflict(tmp_path: Path) -> None:
    Workspace.open(configuration=_cfg("tenant-1"), root=tmp_path)
    with pytest.raises(ConflictError, match="already holds workspace 'tenant-1'") as info:
        Workspace.open(configuration=_cfg("tenant-2"), root=tmp_path)
    assert info.value.existing_id == "tenant-1"
    assert Workspace(root=tmp_path, configuration=_cfg("tenant-1")).read_meta()["name"] == "Tenant"


def test_changed_organization_reorganizes_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Same rule as `workspace create --organization`: not refused, but loud."""
    Workspace.open(configuration=_cfg(), root=tmp_path)
    renamed = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="NewOrg"),
        storage=_LOCAL,
    )
    with caplog.at_level(logging.WARNING, logger="dgml_core.storage"):
        ws = Workspace.open(configuration=renamed, root=tmp_path)
    assert "'NewOrg' differs from the 'TenantOrg'" in caplog.text
    meta = ws.read_meta()
    assert meta["organization"] == "NewOrg" and ws.organization == "NewOrg"
    assert meta["schema_version"] == WORKSPACE_SCHEMA_VERSION  # merge-preserving rewrite


def test_changed_name_is_updated_quietly(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    Workspace.open(configuration=_cfg(), root=tmp_path)
    renamed = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant v2", organization="TenantOrg"),
        storage=_LOCAL,
    )
    with caplog.at_level(logging.WARNING, logger="dgml_core.storage"):
        ws = Workspace.open(configuration=renamed, root=tmp_path)
    assert caplog.text == "" and ws.read_meta()["name"] == "Tenant v2"


def _pinned(tmp_path: Path) -> Configuration:
    """A local store told where to live, so the Workspace itself needs no root."""
    return Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="TenantOrg"),
        storage=Storage.combined(
            ProviderSpec("dgml_core.storage_local:LocalStore", {"workspace_path": str(tmp_path)})
        ),
    )


def test_workspace_path_becomes_the_root(tmp_path: Path) -> None:
    """A local store told where to live makes that the workspace root — not a scratch
    dir beside it — so `files_dir` and every other direct reader of `ws.root` look at
    the tree the store actually writes."""
    ws = Workspace.open(configuration=_pinned(tmp_path))
    assert ws.root == tmp_path.resolve()
    assert ws.files_dir == tmp_path.resolve() / "files"
    assert (tmp_path / "workspace.json").exists()


def test_local_store_without_a_root_is_refused(tmp_path: Path) -> None:
    """The local store's data *is* the root; a scratch dir would vanish with the object."""
    with pytest.raises(InvalidArgument, match=r"pass root=.*or set 'workspace_path'"):
        Workspace.open(configuration=_cfg())
    Workspace.open(configuration=_cfg(), root=tmp_path)  # with a root: fine


def test_a_workspace_path_root_is_not_removed_with_the_workspace(tmp_path: Path) -> None:
    """The caller's directory is the root, and it is not ours to remove."""
    ws = Workspace.open(configuration=_pinned(tmp_path))
    root = ws.root
    del ws
    gc.collect()
    assert root.exists() and (root / "workspace.json").exists()


def test_a_callers_root_is_left_alone(tmp_path: Path) -> None:
    ws = Workspace.open(configuration=_cfg(), root=tmp_path)
    del ws
    gc.collect()
    assert tmp_path.is_dir()


def test_path_or_id_is_refused_with_a_configuration(tmp_path: Path) -> None:
    with pytest.raises(InvalidArgument, match="whole workspace"):
        Workspace.open(tmp_path, configuration=_cfg())
    with pytest.raises(InvalidArgument, match="root applies only"):
        Workspace.open(tmp_path, root=tmp_path)


def test_nothing_on_disk_is_read_or_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the feature: user config, env and store of workspaces play no part."""
    user_config_path().parent.mkdir(parents=True, exist_ok=True)
    user_config_path().write_text('[models]\nadvanced = "user/advanced"\n', encoding="utf-8")
    monkeypatch.setenv("DGML_MODELS__ADVANCED", "env/advanced")
    from dgml_core.grounded import load_grounded_config

    ws = Workspace.open(configuration=_cfg(), root=tmp_path / "ws")
    assert load_grounded_config(ws).values_model == "cfg/advanced"
    assert ws.config_text is None and not (tmp_path / "ws" / "config.toml").exists()
    assert ws.config_location == "in-memory configuration"
    # And the usual store calls work against it.
    ds = DocSetStore(ws).create(name="Bills")
    assert DocSetStore(ws).get(ds.id).name == "Bills"


def test_writers_refuse(tmp_path: Path) -> None:
    from dgml_core import workspace_config

    ws = Workspace.open(configuration=_cfg(), root=tmp_path)
    with pytest.raises(InvalidArgument, match="never written"):
        workspace_config.write_identity(ws, name="x")
    with pytest.raises(InvalidArgument, match="never written"):
        workspace_config.reseal(ws)


def test_local_store_alias_without_a_root_is_refused() -> None:
    """The guard decides on the resolved class, not the provider string: another
    spelling of the same class is still the local store."""
    alias = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="TenantOrg"),
        storage=Storage.combined(ProviderSpec("dgml_core:LocalStore")),
    )
    with pytest.raises(InvalidArgument, match=r"pass root=.*or set 'workspace_path'"):
        Workspace.open(configuration=alias)


def test_blobs_and_docs_with_different_workspace_paths_are_refused(tmp_path: Path) -> None:
    split = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="TenantOrg"),
        storage=Storage(
            blobs=ProviderSpec(
                "dgml_core.storage_local:LocalStore", {"workspace_path": str(tmp_path / "a")}
            ),
            docs=ProviderSpec(
                "dgml_core.storage_local:LocalStore", {"workspace_path": str(tmp_path / "b")}
            ),
        ),
    )
    with pytest.raises(InvalidArgument, match="different 'workspace_path'"):
        Workspace.open(configuration=split)


# ---- the scratch-root path needs a store with no directory at all ------------


class _InMemoryStore(BlobStore, DocStore):
    """A store with no directory — what "every store is remote" looks like to the
    guard — so the scratch-root path can be exercised without a real backend."""

    name = "in-memory"
    config_fields = frozenset[str]()

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        cls._check_no_extra_fields(config.options)
        return config

    def __init__(self, config: StorageConfig) -> None:
        self._blobs: dict[str, bytes] = {}
        self._docs: dict[str, dict[str, dict[str, Any]]] = {}
        self._log: dict[str, list[dict[str, Any]]] = {}

    def put_blob(self, key: str, data: bytes) -> None:
        self._blobs[key] = data

    def get_blob(self, key: str) -> bytes:
        try:
            return self._blobs[key]
        except KeyError:
            raise FileNotFoundError(key) from None

    def delete_blob(self, key: str) -> None:
        self._blobs.pop(key, None)

    def blob_exists(self, key: str) -> bool:
        return key in self._blobs

    def list_blobs(self, prefix: str) -> list[str]:
        return sorted(k for k in self._blobs if k.startswith(prefix))

    def upload_blob(self, key: str, src: Path) -> None:
        self._blobs[key] = src.read_bytes()

    def download_blob(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.get_blob(key))

    def delete_blobs(self, prefix: str) -> None:
        for key in self.list_blobs(prefix):
            del self._blobs[key]

    def append_doc(self, collection: str, doc: dict[str, Any]) -> None:
        self._log.setdefault(collection, []).append(dict(doc))

    def get_doc(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        found = self._docs.get(collection, {}).get(doc_id)
        return dict(found) if found is not None else None

    def find_docs(self, collection: str, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows = list(self._docs.get(collection, {}).values()) + self._log.get(collection, [])
        return [dict(d) for d in rows if all(d.get(k) == v for k, v in query.items())]

    def put_doc(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        self._docs.setdefault(collection, {})[doc_id] = dict(doc)

    def delete_doc(self, collection: str, doc_id: str) -> None:
        self._docs.get(collection, {}).pop(doc_id, None)

    def delete_docs(self, collection: str, query: Mapping[str, Any]) -> int:
        col = self._docs.get(collection, {})
        hits = [i for i, d in col.items() if all(d.get(k) == v for k, v in query.items())]
        for i in hits:
            del col[i]
        return len(hits)


def _provider(monkeypatch: pytest.MonkeyPatch, cls: type, attr: str) -> str:
    """Register ``cls`` under an importable module name; return its dotted path."""
    module = types.ModuleType("fake_stores")
    setattr(module, attr, cls)
    monkeypatch.setitem(sys.modules, "fake_stores", module)
    return f"fake_stores:{attr}"


def _with_storage(provider: str) -> Configuration:
    return Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="TenantOrg"),
        storage=Storage.combined(ProviderSpec(provider)),
    )


def test_local_store_subclass_without_a_root_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subclass is still the local store: its data is still the root."""

    class _Sub(LocalStore):
        pass

    with pytest.raises(InvalidArgument, match=r"pass root=.*or set 'workspace_path'"):
        Workspace.open(configuration=_with_storage(_provider(monkeypatch, _Sub, "Sub")))


def test_scratch_dir_goes_with_its_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every store remote there is no local data, so the root is a scratch dir
    that holds nothing and is removed with the object."""
    cfg = _with_storage(_provider(monkeypatch, _InMemoryStore, "InMemoryStore"))
    ws = Workspace.open(configuration=cfg)
    root = ws.root
    assert root.is_dir() and root.name.startswith("dgml-ws-")
    assert ws.read_meta()["workspace_id"] == "tenant-1"  # went to the store, not to disk
    assert list(root.iterdir()) == []
    del ws
    gc.collect()
    assert not root.exists()
