"""YAML deserialisation for Fiddle configs.

This module is the inverse of ``fiddle._src.experimental.yaml_serialization``,
which serialises ``fdl.Config`` and ``fdl.Partial`` objects to YAML using
custom ``yaml.SafeDumper`` representers.  That module only provides
``dump_yaml`` — there is no corresponding ``load_yaml`` in Fiddle — so we
implement one here by mirroring each representer as a ``yaml.SafeLoader``
constructor.

Representer → Constructor mapping
----------------------------------
``_config_representer``   (tag ``!fdl.Config``)
    Writes a mapping with ``__fn_or_cls__: {module, name}`` plus the config
    arguments.  We reverse this by resolving the class via ``importlib`` and
    constructing a ``fdl.Config``.

``_partial_representer``  (tag ``!fdl.Partial``)
    Identical structure to ``!fdl.Config``; we construct a ``fdl.Partial``
    instead.

``_type_representer``     (tag ``!type``, registered in ``orchestrator.py``)
    Writes a scalar string ``"module.ClassName"``.  We reverse this by
    splitting on the last dot and resolving via ``importlib``.

``_taggedvalue_representer`` (tag ``!fdl.TaggedValue``)
    Not used in any mlcast config; raises ``NotImplementedError`` if
    encountered so the caller gets a clear error rather than silent data loss.
"""

import importlib
from pathlib import Path

import fiddle as fdl
import yaml


def _resolve_class(module: str, name: str):
    """Resolve a class from its module path and qualified name.

    Mirrors the inverse of the ``__fn_or_cls__`` dict written by
    ``_config_representer`` in ``fiddle._src.experimental.yaml_serialization``.
    Handles nested classes expressed as ``OuterClass.InnerClass`` via
    successive ``getattr`` calls.
    """
    mod = importlib.import_module(module)
    obj = mod
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


class _FiddleLoader(yaml.SafeLoader):
    """A YAML loader that reconstructs ``fdl.Config`` / ``fdl.Partial`` objects.

    Registers constructors for the three tags emitted by
    ``fiddle._src.experimental.yaml_serialization`` plus the ``!type`` tag
    registered in ``mlcast.config.orchestrator``.  All constructors mirror
    their corresponding representer in reverse.
    """


def _construct_fdl_config(loader: _FiddleLoader, node: yaml.MappingNode) -> fdl.Config:
    """Constructor for ``!fdl.Config`` — mirrors ``_config_representer``.

    ``_config_representer`` serialises a ``fdl.Config`` as a YAML mapping with
    a special ``__fn_or_cls__: {module, name}`` key alongside the config
    arguments.  We extract and resolve that key, then forward the remaining
    items as keyword arguments to ``fdl.Config``.
    """
    mapping = loader.construct_mapping(node, deep=True)
    fn_or_cls_ref = mapping.pop("__fn_or_cls__")
    cls = _resolve_class(fn_or_cls_ref["module"], fn_or_cls_ref["name"])
    cfg = fdl.Config(cls)
    for key, value in mapping.items():
        setattr(cfg, key, value)
    return cfg


def _construct_fdl_partial(loader: _FiddleLoader, node: yaml.MappingNode) -> fdl.Partial:
    """Constructor for ``!fdl.Partial`` — mirrors ``_partial_representer``.

    ``_partial_representer`` delegates to ``_config_representer`` with
    ``type_name="fdl.Partial"``, so the YAML structure is identical to
    ``!fdl.Config``.  We reconstruct a ``fdl.Partial`` instead of a
    ``fdl.Config``.
    """
    mapping = loader.construct_mapping(node, deep=True)
    fn_or_cls_ref = mapping.pop("__fn_or_cls__")
    cls = _resolve_class(fn_or_cls_ref["module"], fn_or_cls_ref["name"])
    partial = fdl.Partial(cls)
    for key, value in mapping.items():
        setattr(partial, key, value)
    return partial


def _construct_type(loader: _FiddleLoader, node: yaml.ScalarNode) -> type:
    """Constructor for ``!type`` — mirrors ``_type_representer`` in orchestrator.py.

    ``_type_representer`` serialises a Python type as a scalar string of the
    form ``"module.ClassName"``.  We split on the last dot and resolve via
    ``importlib``.
    """
    value = loader.construct_scalar(node)
    module, name = value.rsplit(".", 1)
    return _resolve_class(module, name)


def _construct_fdl_tagged_value(loader: _FiddleLoader, node: yaml.MappingNode) -> None:
    """Constructor for ``!fdl.TaggedValue`` — mirrors ``_taggedvalue_representer``.

    ``fdl.TaggedValue`` is not used in any mlcast config.  Raise a clear error
    rather than silently failing or returning ``None``.
    """
    raise NotImplementedError(
        "load_yaml_config does not support fdl.TaggedValue nodes. "
        "The YAML file contains a !fdl.TaggedValue tag, which is not used in "
        "any mlcast config and has no deserialisation implementation here."
    )


# Register constructors on _FiddleLoader — these are the inverses of the
# representers registered on yaml.SafeDumper in:
#   fiddle._src.experimental.yaml_serialization  (!fdl.Config, !fdl.Partial, !fdl.TaggedValue)
#   mlcast.config.orchestrator                   (!type)
_FiddleLoader.add_constructor("!fdl.Config", _construct_fdl_config)
_FiddleLoader.add_constructor("!fdl.Partial", _construct_fdl_partial)
_FiddleLoader.add_constructor("!type", _construct_type)
_FiddleLoader.add_constructor("!fdl.TaggedValue", _construct_fdl_tagged_value)


def load_yaml_config(path: str | Path) -> fdl.Config:
    """Load a Fiddle config from a YAML file produced by ``dump_yaml``.

    This is the deserialisation counterpart to
    ``fiddle.experimental.yaml_serialization.dump_yaml``, which Fiddle itself
    does not provide.  The YAML file must have been written by ``dump_yaml``
    (or be structurally equivalent), using the ``!fdl.Config``, ``!fdl.Partial``
    and ``!type`` tags.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    cfg : fdl.Config
        The reconstructed Fiddle configuration object.

    Raises
    ------
    NotImplementedError
        If the YAML contains a ``!fdl.TaggedValue`` node, which is not
        supported.
    """
    path = Path(path)
    with path.open() as f:
        return yaml.load(f, Loader=_FiddleLoader)  # noqa: S506 — _FiddleLoader is a safe subclass
