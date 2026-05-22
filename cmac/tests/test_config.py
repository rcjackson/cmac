""" Unit Tests for CMAC 2.0's config.py module. """

import os
import textwrap

import pytest

from cmac.config import (get_cmac_values, get_field_names,
                         get_plot_values, get_metadata,
                         get_zs_relationships)


def test_get_cmac_values():
    cmac_config = get_cmac_values('xsapr_i5_ppi')
    assert type(cmac_config) == dict

    assert cmac_config['save_name'] == 'sgpxsaprcmacsecI5.c1'
    assert cmac_config['site_alt'] == 328
    assert cmac_config['ref_offset'] == 0.0
    assert cmac_config['self_const'] == 60000.00
    assert cmac_config['attenuation_a_coef'] == 0.17

def test_get_field_names():
    field_config = get_field_names('xsapr_i5_ppi')
    assert type(field_config) == dict

    assert field_config['reflectivity'] == 'reflectivity'
    assert field_config['temperature'] == 'tdry'


def test_get_metadata():
    meta_config = get_metadata('xsapr_i5_ppi')
    assert type(meta_config) == dict

    assert meta_config['site_id'] == 'sgp'
    assert meta_config['facility_id'] == 'I5: Garber, OK'
    assert meta_config['version'] == '2.0 lite'


def test_get_plot_values():
    plot_config = get_plot_values('xsapr_i5_ppi')
    assert type(plot_config) == dict

    assert plot_config['sweep'] == 3
    assert plot_config['max_lat'] == 37.0
    assert plot_config['min_lon'] == -98.3
    assert plot_config['site_i5_dms_lat'] == (36, 29, 29.4)
    assert plot_config['site_i5_dms_lon'] == (-97, 35, 37.68)


# -- YAML overrides ---------------------------------------------------------

YAML_BODY = textwrap.dedent("""
    cmac_values:
      xsapr_i5_ppi:
        ref_offset: 1.25
        zdr_offset: 2.5
      my_new_radar:
        save_name: my_radar.c1
        site_alt: 500
        ref_offset: 0.5
    metadata:
      xsapr_i5_ppi:
        developers: "Override Author"
    field_names:
      xsapr_i5_ppi:
        reflectivity: my_reflectivity
    plot_values:
      xsapr_i5_ppi:
        sweep: 7
    zs_relationships:
      My Test Relationship:
        A: 99
        B: 1.5
        abbreviation: mytest
    """)


@pytest.fixture
def yaml_config(tmp_path):
    path = tmp_path / "user_config.yaml"
    path.write_text(YAML_BODY)
    return str(path)


def test_yaml_overrides_cmac_values(yaml_config):
    cfg = get_cmac_values('xsapr_i5_ppi', config_file=yaml_config)
    # Overridden values come from YAML.
    assert cfg['ref_offset'] == 1.25
    assert cfg['zdr_offset'] == 2.5
    # Untouched values still come from the defaults.
    assert cfg['save_name'] == 'sgpxsaprcmacsecI5.c1'
    assert cfg['site_alt'] == 328


def test_yaml_overrides_metadata(yaml_config):
    meta = get_metadata('xsapr_i5_ppi', config_file=yaml_config)
    assert meta['developers'] == 'Override Author'
    # Fallback for un-overridden keys.
    assert meta['site_id'] == 'sgp'


def test_yaml_overrides_field_names(yaml_config):
    fields = get_field_names('xsapr_i5_ppi', config_file=yaml_config)
    assert fields['reflectivity'] == 'my_reflectivity'
    assert fields['temperature'] == 'tdry'


def test_yaml_overrides_plot_values(yaml_config):
    plot = get_plot_values('xsapr_i5_ppi', config_file=yaml_config)
    assert plot['sweep'] == 7
    assert plot['min_lon'] == -98.3


def test_yaml_adds_new_radar(yaml_config):
    cfg = get_cmac_values('my_new_radar', config_file=yaml_config)
    assert cfg['save_name'] == 'my_radar.c1'
    assert cfg['site_alt'] == 500
    assert cfg['ref_offset'] == 0.5


def test_yaml_adds_zs_relationship(yaml_config):
    rels = get_zs_relationships(config_file=yaml_config)
    assert 'My Test Relationship' in rels
    assert rels['My Test Relationship']['A'] == 99
    # Defaults still present.
    assert 'Wolf and Snider (2012)' in rels


def test_yaml_does_not_mutate_defaults(yaml_config):
    # Mutate the dict returned via YAML and confirm defaults stay clean.
    cfg = get_cmac_values('xsapr_i5_ppi', config_file=yaml_config)
    cfg['ref_offset'] = 999
    default_cfg = get_cmac_values('xsapr_i5_ppi')
    assert default_cfg['ref_offset'] == 0.0


def test_yaml_missing_file_raises(tmp_path):
    missing = str(tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError):
        get_cmac_values('xsapr_i5_ppi', config_file=missing)
