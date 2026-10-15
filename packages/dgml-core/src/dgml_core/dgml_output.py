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

"""The DGML the pipeline produced for a file in a DocSet.

Each (DocSet, file) pair has one core DGML file,
``docsets/<did>/files/<fid>/<stem>.dgml.xml``. ``docset generate`` writes the
grounded document tree there, and ``extraction extract`` adds a
``dg:extraction`` element to it (or writes a standalone one when no tree exists
yet). DGML is always per DocSet: its tags come from the DocSet's schema, so a
file has no DGML of its own outside a DocSet.

:func:`read_dgml` returns that file's stored bytes and blob key: the full DGML,
tree and extraction alike. :func:`dgml_core.extraction.get_extracted_values`
builds on it to return just the extracted values as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import layout
from .docsets import DocSetStore
from .errors import (
    DgmlNotFound,
    DocSetNotFound,
    FileNotFound,
    InvalidArgument,
    ValuesNotFound,
)
from .models import FileRecord
from .storage import Workspace


@dataclass(frozen=True)
class DgmlDocument:
    """A pair's core DGML file as stored."""

    key: str
    """Blob key, ``docsets/<did>/files/<fid>/<stem>.dgml.xml``."""
    xml: bytes
    """The stored bytes, unmodified."""


def read_dgml(
    workspace: Workspace,
    docset_id: str,
    file_id: str,
    *,
    require_extraction: bool = False,
) -> DgmlDocument:
    """Return the core DGML file for ``(docset_id, file_id)``.

    With ``require_extraction``, the file must also carry a ``dg:extraction``
    element; a missing file then counts as "not extracted" too.

    Raises:
        :class:`InvalidArgument` — empty ids.
        :class:`DocSetNotFound` — unknown docset.
        :class:`FileNotFound` — unknown file, or a file not assigned to the docset.
        :class:`DgmlNotFound` — assigned, but no DGML written yet.
        :class:`ValuesNotFound` — ``require_extraction`` and no ``dg:extraction``
            (or no DGML at all).
    """
    if not docset_id.strip():
        raise InvalidArgument("docset id must not be empty")
    if not file_id.strip():
        raise InvalidArgument("file id must not be empty")
    if workspace.docs.get_doc(layout.Collection.DOCSETS, docset_id) is None:
        raise DocSetNotFound(f"docset '{docset_id}' not found")
    record_data = workspace.docs.get_doc(layout.Collection.FILES, file_id)
    if record_data is None:
        raise FileNotFound(f"file '{file_id}' not found")
    if not DocSetStore(workspace).is_assigned(docset_id, file_id):
        raise FileNotFound(f"file '{file_id}' is not assigned to docset '{docset_id}'")

    stem = Path(FileRecord.from_json(record_data).original_filename).stem
    key = layout.dgml_xml_key(docset_id, file_id, stem)
    try:
        xml = workspace.blobs.get_blob(key)
    except FileNotFoundError:
        if require_extraction:
            raise _values_not_found(docset_id, file_id) from None
        raise DgmlNotFound(
            f"no DGML for file '{file_id}' in docset '{docset_id}'; "
            "run 'dgml docset generate' or 'dgml extraction extract' first"
        ) from None

    if require_extraction:
        # Deferred: extraction_xml pulls in the generation and matching modules,
        # which a plain ``import dgml_core`` should not pay for.
        from .extraction_xml import has_extraction

        if not has_extraction(xml.decode("utf-8")):
            raise _values_not_found(docset_id, file_id)
    return DgmlDocument(key=key, xml=xml)


def _values_not_found(docset_id: str, file_id: str) -> ValuesNotFound:
    return ValuesNotFound(
        f"no extracted values for file '{file_id}' in docset '{docset_id}'; "
        "run 'dgml extraction extract' first"
    )
