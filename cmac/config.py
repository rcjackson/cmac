"""
cmac.config
===========
CMAC 2.0 Configuration.

    get_metadata
    get_field_names
    get_cmac_values
    get_plot_values
    get_zs_relationships

All getters accept an optional ``config_file`` argument. When supplied, the
YAML file at that path is loaded and any values it contains override the
defaults; anything not specified in the YAML falls back to the built-in
defaults defined in ``cmac.default_config``.

The YAML file mirrors the structure of the default dictionaries, e.g.::

    metadata:
      my_radar:
        site_id: sgp
        ...
    field_names:
      my_radar:
        reflectivity: reflectivity
        ...
    cmac_values:
      my_radar:
        save_name: my_radar.c1
        site_alt: 300
        ...
    plot_values:
      my_radar:
        sweep: 3
        ...
    zs_relationships:
      My Relationship:
        A: 100
        B: 2
        abbreviation: my_rel

"""

import os
from copy import deepcopy

from .default_config import (_DEFAULT_METADATA, _DEFAULT_FIELD_NAMES,
                             _DEFAULT_CMAC_VALUES, _DEFAULT_PLOT_VALUES,
                             _DEFAULT_ZS_RELATIONSHIPS)


# Cache of parsed YAML files keyed by (absolute_path, mtime) so repeated calls
# during a single processing run don't re-read or re-parse the file.
_YAML_CACHE = {}


def _load_yaml_config(config_file):
    """Return the parsed YAML config as a dict, using a small mtime cache."""
    try:
        import yaml
    except ImportError as err:
        raise ImportError(
            "Loading a CMAC config from YAML requires PyYAML. "
            "Install it with `pip install pyyaml` or `conda install pyyaml`."
        ) from err

    abs_path = os.path.abspath(config_file)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            "CMAC config file not found: %s" % config_file)

    mtime = os.path.getmtime(abs_path)
    cached = _YAML_CACHE.get(abs_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with open(abs_path, 'r') as fh:
        parsed = yaml.safe_load(fh) or {}
    if not isinstance(parsed, dict):
        raise ValueError(
            "CMAC config file %s must contain a mapping at the top level."
            % config_file)

    _YAML_CACHE[abs_path] = (mtime, parsed)
    return parsed


def _resolve(section, radar, defaults, config_file):
    """Merge YAML overrides for ``radar`` on top of ``defaults[radar]``.

    Returns a fresh dict so callers can mutate it without affecting either
    the defaults or the YAML cache.
    """
    base = deepcopy(defaults.get(radar, {}))
    if config_file is None:
        return base

    yaml_cfg = _load_yaml_config(config_file)
    section_cfg = yaml_cfg.get(section) or {}
    radar_overrides = section_cfg.get(radar)
    if radar_overrides is None:
        return base

    if not isinstance(radar_overrides, dict):
        raise ValueError(
            "YAML config section '%s' for radar '%s' must be a mapping."
            % (section, radar))

    base.update(deepcopy(radar_overrides))
    return base


def get_metadata(radar, config_file=None):
    """
    Return a dictionary of metadata for a given radar. An empty dictionary
    will be returned if no metadata exists for ``radar`` in either the YAML
    config or the defaults.
    """
    return _resolve('metadata', radar, _DEFAULT_METADATA, config_file)


def get_field_names(radar, config_file=None):
    """
    Return the field name mapping for a given radar. When ``config_file``
    is provided, values from the YAML file override the defaults.
    """
    if config_file is None:
        return _DEFAULT_FIELD_NAMES[radar]
    merged = _resolve('field_names', radar, _DEFAULT_FIELD_NAMES, config_file)
    if not merged:
        raise KeyError(radar)
    return merged


def get_cmac_values(radar, config_file=None):
    """
    Return the values specific to a radar for processing the radar data,
    using CMAC 2.0. When ``config_file`` is provided, values from the YAML
    file override the defaults.
    """
    if config_file is None:
        return _DEFAULT_CMAC_VALUES[radar].copy()
    merged = _resolve('cmac_values', radar, _DEFAULT_CMAC_VALUES, config_file)
    if not merged:
        raise KeyError(radar)
    return merged


def get_plot_values(radar, config_file=None):
    """
    Return the values specific to a radar for plotting the radar fields.
    When ``config_file`` is provided, values from the YAML file override
    the defaults.
    """
    if config_file is None:
        return _DEFAULT_PLOT_VALUES[radar].copy()
    merged = _resolve('plot_values', radar, _DEFAULT_PLOT_VALUES, config_file)
    if not merged:
        raise KeyError(radar)
    return merged


def get_zs_relationships(config_file=None):
    """
    Return the set of Z-S relationships to use. When ``config_file`` is
    provided, any relationships defined under the ``zs_relationships``
    section of the YAML file override (or add to) the defaults.
    """
    base = deepcopy(_DEFAULT_ZS_RELATIONSHIPS)
    if config_file is None:
        return base

    yaml_cfg = _load_yaml_config(config_file)
    zs_overrides = yaml_cfg.get('zs_relationships') or {}
    if not isinstance(zs_overrides, dict):
        raise ValueError(
            "YAML config section 'zs_relationships' must be a mapping.")
    base.update(deepcopy(zs_overrides))
    return base
