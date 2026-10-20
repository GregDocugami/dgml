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

"""Which config keys hold a credential, and tables with those values hidden.

Shared by the reprs of the configuration objects (so a ``Configuration`` can be logged)
and by the storage seal (which leaves credentials out of a store's identity hash).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: A key naming a credential carries one of these: ``api_key``,
#: ``aws_secret_access_key``, ``auth_token``. The storage seal matches them as
#: substrings; :func:`is_secret_name` as whole words (``max_tokens`` is a count).
SECRET_HINTS = ("key", "secret", "token", "password", "credential")

#: Matched as a whole ``_``-separated word, not a substring: a connection URI
#: (``mongo_uri``) carries credentials inline, but ``security_…`` is not a secret.
SECRET_WORDS = frozenset({"uri"})

#: What a hidden value prints as.
REDACTED = "***"

_SECRET_NAME_WORDS = frozenset(SECRET_HINTS) | SECRET_WORDS


def is_secret_name(name: str) -> bool:
    """Whether a config key names a credential *value*: one of its ``_``-separated
    words is a secret hint. A ``*_env`` key names the variable holding one, which is
    not itself a secret."""
    lowered = name.lower()
    if lowered.endswith("_env"):
        return False
    return not _SECRET_NAME_WORDS.isdisjoint(lowered.split("_"))


def redacted(table: Mapping[str, Any]) -> dict[str, Any]:
    """*table* with every secret-named value replaced by :data:`REDACTED`, nested
    tables included. ``None`` stays ``None``: unset is not a secret."""
    out: dict[str, Any] = {}
    for key, value in table.items():
        if isinstance(value, Mapping):
            out[key] = redacted(value)
        elif value is not None and is_secret_name(str(key)):
            out[key] = REDACTED
        else:
            out[key] = value
    return out
