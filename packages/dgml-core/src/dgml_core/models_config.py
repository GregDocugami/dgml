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

"""The ``[models]`` tier block — the simplified model entry point.

Four tiers, cheapest to strongest, each mapped to a set of tasks:

* ``light`` ...... classification, style
* ``standard`` ... transcription, text extraction
* ``advanced`` ... labeling, value extraction
* ``expert`` ..... schema generation

Each tier is a :class:`Model`: a model id plus optional credentials
(``<tier>_api_key`` / ``<tier>_api_key_env`` / ``<tier>_api_base``). **A model and
its credentials are one group, written together in one config layer.** A tier's
credentials are the ``<tier>_*`` keys beside it; a task section's model
(``generation.label_model``, ``grounded.schema_model``) takes only that section's
``{prefix}api_key`` and friends; a credential in a layer that does not set its
model is dropped with a warning, so nothing is ever inherited from another tier,
section or layer. A model with no credentials leaves litellm to read its
per-provider env var — the one fallback there is. See :func:`resolve_tiered_model`
and :func:`normalize_models_layer`.

``<provider>_api_key`` / ``_env`` (``anthropic_``, ``google_``, ``openai_``) is a
setting like any other: its pair merges as one group, the highest layer that sets
either form winning. Once the layers are merged, :func:`apply_provider_keys` writes it
onto every model of that provider — tier, task section or per-call override — that
carries neither its own key nor an ``api_base``. Matching is by model-id prefix, so a
key only ever reaches its own provider's models, and a model pointed at a custom
endpoint never receives a key meant for the provider's own service.

``family`` picks a whole provider family's defaults: one of the
:data:`~dgml_core.default_config.PROVIDER_MODELS` keys. It is shorthand for that
family's four tiers *within its config layer* (see :func:`expand_family`): tiers
the same layer sets win, and the expanded tiers override any a lower layer set.
A family-based config tracks dgml's shipped defaults across upgrades; explicit
tiers are the pinning mechanism.

A tier that is unset falls back to the nearest set tier (nearest *lower* first,
then higher), emitting a warning — so a minimal config that sets only, say,
``standard`` still resolves every task.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .default_config import PROVIDER_MODELS, PROVIDERS
from .errors import DgmlError, ModelsConfigInvalid

logger = logging.getLogger(__name__)


class Tier(StrEnum):
    """A ``[models]`` capability tier. Definition order is cheapest → strongest,
    which :meth:`ModelsConfig._nearest_set` relies on for fallback ordering.

    A ``StrEnum``, so a member is usable directly as a config key, in f-strings,
    and anywhere a plain tier string was expected."""

    LIGHT = "light"
    STANDARD = "standard"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ConfigSection(StrEnum):
    """A top-level config section (a ``[section]`` table in ``config.toml``).

    A ``StrEnum`` whose value is the literal TOML section name, so it doubles as
    the lookup key into the merged config mapping and formats to the bare name in
    error messages. The tier-backed sections (``classification``, ``style``,
    ``text_extraction``, ``generation``, ``grounded``) are the ones passed to
    :func:`resolve_tiered_model`; the rest (``models``, ``ocr``, ``pdf``,
    ``conversion``, ``clustering``) configure non-LLM or tier-source settings."""

    MODELS = "models"
    GENERATION = "generation"
    GROUNDED = "grounded"
    CLASSIFICATION = "classification"
    OCR = "ocr"
    PDF = "pdf"
    STYLE = "style"
    TEXT_EXTRACTION = "text_extraction"
    CONVERSION = "conversion"
    CLUSTERING = "clustering"
    STORAGE = "storage"


# Cheapest → strongest. Fallback searches lower (cheaper) neighbours first.
TIERS: tuple[Tier, ...] = tuple(Tier)

# Tier fallbacks already reported this process, so a per-file loop (e.g. bulk
# extract) doesn't repeat the same warning. Keyed by (requested, used).
_WARNED_TIER_FALLBACKS: set[tuple[Tier, Tier]] = set()

# Same idea for "configured but not enabled" advisories (see `section_enabled`):
# every loader re-reads the merged config, so without this the same line would be
# written several times per command.
_WARNED_DISABLED: set[ConfigSection] = set()

# Provider keys withheld from a keyless model at a custom api_base (see
# `apply_provider_keys`), keyed by the message.
_WARNED_PROVIDER_KEY_NOT_APPLIED: set[str] = set()

# Credentials dropped from a layer that does not set their model (see
# `_drop_stray_group`), keyed by the message.
_WARNED_DROPPED_CREDENTIAL: set[str] = set()


@dataclass(frozen=True)
class Model:
    """A model id with its credentials — the typed form of one ``model`` / ``api_key``
    / ``api_key_env`` / ``api_base`` key group, wherever it appears: a ``[models]``
    tier, a task section's model, or a task's resolved model. The key itself is left
    out of the repr; the variable name and endpoint are not secrets."""

    model: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    api_base: str | None = None

    def table(self, model_key: str, prefix: str) -> dict[str, Any]:
        """The flat keys: ``model_key`` for the id, ``{prefix}api_key`` and so on for
        the rest (``light`` / ``light_``, ``label_model`` / ``label_``)."""
        out = {model_key: self.model}
        for name in GROUP_SUFFIXES:
            value = getattr(self, name)
            if value is not None:
                out[f"{prefix}{name}"] = value
        return out


Credentials = tuple[str | None, str | None]
"""An ``(api_key, api_key_env)`` pair; at most one is set."""


def provider_of(model: str) -> str | None:
    """The :data:`~dgml_core.default_config.PROVIDERS` key serving ``model``'s
    litellm prefix, or ``None`` for an unprefixed or unlisted one."""
    prefix, sep, _ = model.partition("/")
    if not sep:
        return None
    return next((p for p, prefixes in PROVIDERS.items() if prefix in prefixes), None)


@dataclass(frozen=True)
class ModelsConfig:
    """Parsed ``[models]`` block: one :class:`Model` per tier, all optional, each
    carrying exactly the credentials written beside it."""

    light: Model | None = None
    standard: Model | None = None
    advanced: Model | None = None
    expert: Model | None = None

    def resolve(self, tier: Tier) -> Model | None:
        """Resolve ``tier`` to its :class:`Model`.

        If ``tier`` has no model, fall back to the nearest set tier — lower
        (cheaper) neighbours first, then higher — and log a WARNING (once per
        process per fallback). The fallback tier's credentials come with its
        model. Returns ``None`` when no tier is set at all (the caller then
        surfaces the appropriate config error)."""
        if tier not in TIERS:
            raise ValueError(f"unknown model tier {tier!r}")
        actual = self._nearest_set(tier)
        if actual is None:
            return None
        model: Model = getattr(self, actual)
        if actual != tier and (tier, actual) not in _WARNED_TIER_FALLBACKS:
            _WARNED_TIER_FALLBACKS.add((tier, actual))
            logger.warning(
                f"[dgml] model tier '{tier}' is not set; falling back to '{actual}' "
                f"('{model.model}'). Set [models].{tier} to silence this."
            )
        return model

    def _nearest_set(self, tier: Tier) -> Tier | None:
        idx = TIERS.index(tier)
        # Lower (cheaper) neighbours nearest-first, then higher neighbours.
        order = list(range(idx - 1, -1, -1)) + list(range(idx + 1, len(TIERS)))
        if getattr(self, tier) is not None:
            return tier
        for i in order:
            if getattr(self, TIERS[i]) is not None:
                return TIERS[i]
        return None


def _opt_str(value: Any, label: str, invalid: type[DgmlError]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise invalid(f"'{label}' must be a non-empty string if set")
    return value


def _credentials(
    table: dict[str, Any], prefix: str, section: str, invalid: type[DgmlError]
) -> Credentials:
    """``{prefix}api_key`` / ``{prefix}api_key_env`` from ``table``; setting both is an
    error."""
    key_label, env_label = f"{section}.{prefix}api_key", f"{section}.{prefix}api_key_env"
    api_key = _opt_str(table.get(f"{prefix}api_key"), key_label, invalid)
    api_key_env = _opt_str(table.get(f"{prefix}api_key_env"), env_label, invalid)
    if api_key is not None and api_key_env is not None:
        raise invalid(f"set at most one of '{key_label}' / '{env_label}', not both")
    return api_key, api_key_env


def expand_family(models: dict[str, Any]) -> dict[str, Any]:
    """Return one config layer's ``[models]`` table with its ``family`` expanded
    into the tiers the table leaves unset. Called per layer *before* the merge, so
    a family overrides lower-layer tiers. A malformed or unknown family is left
    as-is for :func:`load_models_config` to reject."""
    family = models.get("family")
    defaults = PROVIDER_MODELS.get(family) if isinstance(family, str) else None
    return models if defaults is None else {**defaults, **models}


#: The credential keys that travel with a model key as one group.
GROUP_SUFFIXES = ("api_key", "api_key_env", "api_base")

#: Sections whose ``model`` / ``<prefix>_model`` keys head a credential group. Named
#: rather than inferred: ``ocr`` also carries an ``api_key``, but no model it belongs to.
LLM_SECTIONS = frozenset(
    {
        ConfigSection.GENERATION,
        ConfigSection.GROUNDED,
        ConfigSection.CLASSIFICATION,
        ConfigSection.STYLE,
        ConfigSection.TEXT_EXTRACTION,
    }
)

#: Sections that are off unless ``enabled = true`` (see :func:`section_enabled`).
OPT_IN_SECTIONS = frozenset({ConfigSection.STYLE, ConfigSection.TEXT_EXTRACTION})


def _model_keys(table: Mapping[str, Any]) -> list[str]:
    """The keys of ``table`` that head a model group: ``model`` and every ``*_model``."""
    return [k for k in table if k == "model" or k.endswith("_model")]


def _stray_credential(table: Mapping[str, Any], prefix: str) -> str | None:
    """The first of ``{prefix}api_key`` / ``_env`` / ``api_base`` set in ``table``."""
    return next((s for s in GROUP_SUFFIXES if table.get(f"{prefix}{s}") is not None), None)


def _complete_group(table: dict[str, Any], prefix: str) -> None:
    """Write ``None`` for the credentials of one model group the layer leaves unset.

    A ``None`` survives the layer merge where an absent key would not, so a higher
    layer that sets a model replaces its whole group instead of pairing the new model
    with a lower layer's key. The merge drops those ``None``s afterwards."""
    for suffix in GROUP_SUFFIXES:
        table.setdefault(f"{prefix}{suffix}", None)


def _drop_stray_group(
    table: dict[str, Any], prefix: str, section: str, model_key: str, remedy: str, layer: str
) -> None:
    """Drop the credentials of a model group whose model *layer* does not set, warning
    once per message. A key beside no model has no model it was meant for: paired with
    another layer's it would follow that model wherever that layer moves it, so it is
    withheld — never sent — and the user is told how to place it."""
    stray = _stray_credential(table, prefix)
    if stray is None:
        return
    for suffix in GROUP_SUFFIXES:
        table.pop(f"{prefix}{suffix}", None)
    message = (
        f"[dgml] '{section}.{prefix}{stray}' in {layer} has no '{section}.{model_key}' beside "
        f"it; it is ignored. Set {remedy} in {layer} too, or use 'models.<provider>_api_key' "
        f"/ '_env', which reaches every keyless model of its provider."
    )
    if message not in _WARNED_DROPPED_CREDENTIAL:
        _WARNED_DROPPED_CREDENTIAL.add(message)
        logger.warning(message)


def _unknown_family(family: str) -> ModelsConfigInvalid:
    return ModelsConfigInvalid(
        f"'models.family' must be one of {', '.join(sorted(PROVIDER_MODELS))}; got {family!r}"
    )


def provider_keys(models: dict[str, Any]) -> dict[str, Credentials]:
    """A ``[models]`` table's ``<provider>_api_key`` / ``_env`` pairs, validated; a
    provider with neither is left out."""
    pairs = {p: _credentials(models, f"{p}_", "models", ModelsConfigInvalid) for p in PROVIDERS}
    return {p: pair for p, pair in pairs.items() if pair != (None, None)}


def _take_provider_key(
    table: dict[str, Any], model_key: str, prefix: str, providers: Mapping[str, Credentials]
) -> None:
    """Write the provider key onto one model group that carries no credentials of its
    own and no ``api_base`` — a model at a custom endpoint is not its provider's service."""
    model = table.get(model_key)
    if not isinstance(model, str):
        return
    if any(table.get(f"{prefix}{suffix}") is not None for suffix in GROUP_SUFFIXES):
        return
    api_key, api_key_env = providers.get(provider_of(model) or "", (None, None))
    if api_key is not None:
        table[f"{prefix}api_key"] = api_key
    if api_key_env is not None:
        table[f"{prefix}api_key_env"] = api_key_env


def normalize_models_layer(
    models: dict[str, Any], layer: str = "this config layer"
) -> dict[str, Any]:
    """One config layer's ``[models]`` table with its groups made atomic.

    ``family`` is expanded (:func:`expand_family`); then every set tier's group is
    completed with explicit ``None``s. A tier credential whose tier this layer does not
    set is dropped with a warning (:func:`_drop_stray_group`): a key is written beside
    its model, and a layer boundary must not separate them — that is how a key ends up
    following a model it was never meant for. Runs per layer, before the merge.

    Each provider's key pair is a group too: a layer that sets either form gets
    ``None`` for the other, so the highest layer's pair wins whole and one form in a
    lower layer never clashes with the other in a higher one. Both forms in one layer
    are a clash. The keys stay in the table; :func:`apply_provider_keys` writes them
    onto models after the merge.

    An unknown ``family`` is not expanded but skips none of the rest; it stays in the
    table for :func:`load_models_config` to report, before any tier — so a stray tier
    credential beside it is left alone rather than blamed for the family's typo.
    *layer* names the layer in warnings (``the user config``)."""
    family = models.get("family")
    unknown_family = isinstance(family, str) and family not in PROVIDER_MODELS
    out = dict(expand_family(models))
    for p in provider_keys(out):
        out.setdefault(f"{p}_api_key", None)
        out.setdefault(f"{p}_api_key_env", None)
    for t in TIERS:
        prefix = f"{t}_"
        model = out.get(t.value)
        if model is None:
            if not unknown_family:
                remedy = f"'models.{t}' (or 'models.family')"
                _drop_stray_group(out, prefix, "models", t.value, remedy, layer)
            if t.value in out:
                _complete_group(out, prefix)  # an explicit null unsets the whole group
            continue
        if not isinstance(model, str):
            continue  # malformed; load_models_config reports it
        _complete_group(out, prefix)
    return out


def normalize_task_layer(
    section: dict[str, Any],
    section_name: ConfigSection,
    layer: str = "this config layer",
) -> dict[str, Any]:
    """One layer of an LLM task section with its groups made atomic: every model key
    it sets (``model``, ``label_model``, ``schema_model``, …) has its credential group
    completed with explicit ``None``s. A credential whose model this layer does not set
    is dropped with a warning — the same rule as :func:`normalize_models_layer`. A
    model key set to ``null`` sets no model and unsets its whole group, so a lower
    layer's credentials do not outlive the model they were written beside."""
    out = dict(section)
    model_keys = _model_keys(section)
    prefixes = {k[: -len("model")] for k in model_keys if section.get(k) is not None}
    null_prefixes = {k[: -len("model")] for k in model_keys if section.get(k) is None}
    strays = sorted(
        {
            key[: -len(suffix)]
            for key, value in section.items()
            for suffix in GROUP_SUFFIXES
            if key.endswith(suffix) and value is not None
        }
        - prefixes
    )
    for prefix in strays:
        model_key = f"{prefix}model"
        remedy = f"'{section_name}.{model_key}'"
        _drop_stray_group(out, prefix, section_name.value, model_key, remedy, layer)
    for prefix in prefixes:
        if isinstance(section.get(f"{prefix}model"), str):
            _complete_group(out, prefix)
    for prefix in null_prefixes:
        _complete_group(out, prefix)
    return out


def _keyless_models(merged: Mapping[ConfigSection, Any]) -> Iterator[tuple[str, str, bool]]:
    """``(config key, model id, has api_base)`` for every model in the merged config
    that carries no key. A disabled ``style`` / ``text_extraction`` never calls its
    model, so it is skipped."""
    groups: list[tuple[str, dict[str, Any], str, str]] = []
    models = merged.get(ConfigSection.MODELS)
    if isinstance(models, dict):
        groups += [(f"models.{t}", models, t.value, f"{t}_") for t in TIERS]
    for section in sorted(LLM_SECTIONS):
        table = merged.get(section)
        if not isinstance(table, dict):
            continue
        if section in OPT_IN_SECTIONS and table.get("enabled") is not True:
            continue
        for key in sorted(_model_keys(table)):
            groups.append((f"{section}.{key}", table, key, key[: -len("model")]))
    for label, table, model_key, prefix in groups:
        model = table.get(model_key)
        if isinstance(model, str) and all(
            table.get(f"{prefix}{suffix}") is None for suffix in ("api_key", "api_key_env")
        ):
            yield label, model, table.get(f"{prefix}api_base") is not None


def apply_provider_keys(merged: Mapping[ConfigSection, Any]) -> None:
    """Write the merged ``[models]`` provider keys onto the models that take them.

    Every tier, task-section and per-call override model with no key of its own and
    no ``api_base`` takes its provider's ``<provider>_api_key`` / ``_env``, in place.
    Called once per read, by :func:`~dgml_core.config.load_merged_config` after the
    last layer (the per-call overrides) is in — not before, since a model keyed here
    keeps that key, and a higher layer's provider key would then never reach it.

    A model at a custom ``api_base`` is left keyless on purpose, but then litellm sends
    its own provider env var there instead, so that is warned about — once per process
    per message."""
    models = merged.get(ConfigSection.MODELS)
    if not isinstance(models, dict):
        return
    keys = provider_keys(models)
    if not keys:
        return
    for t in TIERS:
        _take_provider_key(models, t.value, f"{t}_", keys)
    for section in sorted(LLM_SECTIONS):
        table = merged.get(section)
        if isinstance(table, dict):
            for key in _model_keys(table):
                _take_provider_key(table, key, key[: -len("model")], keys)
    at_api_base: dict[str, list[str]] = {}
    for label, model, has_api_base in _keyless_models(merged):
        provider = provider_of(model)
        if has_api_base and provider in keys:
            at_api_base.setdefault(provider, []).append(label)
    for provider, labels in at_api_base.items():
        form = "api_key" if keys[provider][0] is not None else "api_key_env"
        message = (
            f"[dgml] 'models.{provider}_{form}' is not sent to {', '.join(labels)}: a "
            f"provider key never goes to a custom api_base, so litellm will send its own "
            f"{provider} env var there instead. Set a key beside each api_base (or "
            f'"EMPTY" if that server checks none).'
        )
        if message not in _WARNED_PROVIDER_KEY_NOT_APPLIED:
            _WARNED_PROVIDER_KEY_NOT_APPLIED.add(message)
            logger.warning(message)


def load_models_config(merged: dict[ConfigSection, Any]) -> ModelsConfig:
    """Build a :class:`ModelsConfig` from the merged config mapping's
    ``[models]`` section (an empty section yields an all-``None`` config).

    ``family`` is validated here but not expanded (:func:`normalize_models_layer` did
    that per layer), and provider keys are not read — :func:`apply_provider_keys` has
    already written them onto the tiers, so every tier's credentials are the
    ``<tier>_*`` keys beside it. A tier credential with no tier is an error here: the
    merge drops one a layer leaves stray, so this only meets a hand-built mapping."""
    section = merged.get(ConfigSection.MODELS)
    if section is None:
        return ModelsConfig()
    if not isinstance(section, dict):
        raise ModelsConfigInvalid("'models' must be a table")
    invalid = ModelsConfigInvalid
    family = _opt_str(section.get("family"), "models.family", invalid)
    if family is not None and family not in PROVIDER_MODELS:
        raise _unknown_family(family)
    tiers: dict[str, Model] = {}
    for t in TIERS:
        model = _opt_str(section.get(t.value), f"models.{t}", invalid)
        api_key, api_key_env = _credentials(section, f"{t}_", "models", invalid)
        api_base = _opt_str(section.get(f"{t}_api_base"), f"models.{t}_api_base", invalid)
        if model is None:
            stray = _stray_credential(section, f"{t}_")
            if stray is not None:
                raise invalid(f"'models.{t}_{stray}' is set but 'models.{t}' is not")
            continue
        tiers[t.value] = Model(model, api_key, api_key_env, api_base)
    return ModelsConfig(**tiers)


def section_enabled(
    section: dict[str, Any],
    *,
    section_name: ConfigSection,
    invalid: type[DgmlError],
) -> bool:
    """Whether an opt-in feature section carries ``enabled = true``.

    The ``style`` and ``text_extraction`` sections are switches: they configure a
    feature that is off unless explicitly turned on. ``enabled`` is that switch —
    the section's mere *presence* means nothing, so the shipped ``config.toml``
    can name both features (with comments explaining them) without enabling
    either.

    Warns — once per section per process — when a section is disabled but carries
    real configuration beyond ``enabled`` itself. A section holding only
    ``enabled = false`` is the shipped default and says nothing about intent; one
    that also names a model or credentials was written by someone who expects it
    to run. Configs predating the ``enabled`` switch look exactly like that, and
    turning them off silently is the failure mode this switch exists to prevent.

    Raises ``invalid`` when ``enabled`` is present but not a boolean.
    """
    enabled: object = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise invalid(f"'{section_name}.enabled' must be true or false")
    if not enabled and set(section) - {"enabled"} and section_name not in _WARNED_DISABLED:
        _WARNED_DISABLED.add(section_name)
        logger.warning(
            f"[dgml] the [{section_name}] config section is configured but not enabled; "
            f"it will be ignored. Set {section_name}.enabled = true to use it."
        )
    return enabled


def resolve_tiered_model(
    merged: dict[ConfigSection, Any],
    *,
    section_name: ConfigSection,
    tier: Tier,
    invalid: type[DgmlError],
    missing: type[DgmlError],
    prefix: str = "",
) -> Model:
    """Resolve one task's :class:`Model` from the ``[{section_name}]`` section of
    *merged*, or — when the section names no model — from its ``[models]`` *tier*.

    ``prefix`` names the section's key group: ``{prefix}model``, ``{prefix}api_key``,
    ``{prefix}api_key_env``, ``{prefix}api_base`` (``schema_`` / ``values_`` for
    grounded, ``label_`` for generation labeling, empty for the rest).

    A model and its credentials are one group. A section that names a model takes
    exactly that section's ``{prefix}api_key`` / ``_env`` / ``api_base`` — never a
    tier's (the provider key was already written there by the merge) — and
    ``[models]`` is not consulted at all, so a
    malformed tier value or family cannot break a task that names its own model. A
    section that names none takes the tier's :class:`Model` whole. Section credentials
    with no section model are an error, not a partial override: a key written beside
    no model has no model it was meant for (the merge drops one a layer leaves stray,
    so this only meets a hand-built mapping). A model with no credentials leaves
    litellm to read its per-provider env var.

    Raises ``invalid`` for a malformed value, a literal+env-name clash, or
    credentials without a model; ``missing`` when neither the section nor the tier
    resolves a model. Callers that treat a section's mere presence as a feature
    switch (``style`` / ``text_extraction``) check that themselves before calling
    this.
    """
    section = merged.get(section_name)
    sec: dict[str, Any] = section if isinstance(section, dict) else {}
    model = _opt_str(sec.get(f"{prefix}model"), f"{section_name}.{prefix}model", invalid)
    api_key, api_key_env = _credentials(sec, prefix, section_name, invalid)
    api_base = _opt_str(sec.get(f"{prefix}api_base"), f"{section_name}.{prefix}api_base", invalid)

    if model is not None:
        return Model(model, api_key, api_key_env, api_base)
    stray = _stray_credential(sec, prefix)
    if stray is not None:
        raise invalid(
            f"'{section_name}.{prefix}{stray}' is set but '{section_name}.{prefix}model' is "
            f"not: credentials belong to the model written beside them"
        )
    tiered = load_models_config(merged).resolve(tier)
    if tiered is None:
        raise missing(
            f"no {prefix}model for {section_name}: set [models].family, [models].{tier}, "
            f"or '{section_name}.{prefix}model' in the config"
        )
    return tiered
