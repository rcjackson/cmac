"""Integration tests covering cmac() and the per-field PPI/RHI quicklooks.

The quicklook tests use ``pytest-mpl`` to compare each generated figure
against a baseline PNG stored under ``cmac/tests/baseline_images/``.
PPI baselines are named ``ppi_<plot>.png`` and RHI baselines
``rhi_<plot>.png``; the prefix is part of the filename (not the
directory) so pytest-mpl's single-directory ``--mpl-generate-path``
mode produces both sets in one pass without collisions.

The bnfcsapr2cfrS3.a1 datastream returns two files per request: the
first is an RHI scan, the second is a PPI scan. Both are processed
through ``cmac()`` and then through the matching ``quicklooks_*``
function.

To (re)generate the baseline images after a deliberate visual change::

    pytest cmac/tests/test_processing.py \\
        --mpl-generate-path=cmac/tests/baseline_images

To run the comparison (the default once baselines exist)::

    pytest cmac/tests/test_processing.py --mpl

Without the ``--mpl`` flag the baseline tests still execute, but
pytest-mpl skips the image comparison and only checks that a Figure was
returned. ARM_USERNAME / ARM_PASSWORD must be set in the environment
because the fixtures download a sample radar volume and sonde from ARM
Data Discovery.
"""

import glob
import os

import act
import pyart
import pytest
import xarray as xr

from cmac import cmac, quicklooks_ppi, quicklooks_rhi


# cmac() processes both scan strategies the same way, so we feed the PPI
# config (the only one with full metadata/field_names/cmac_values entries)
# to both radars. quicklooks_rhi reuses the same config; the PPI plot_values
# has extra lat/lon keys that quicklooks_rhi ignores.
RADAR_CONFIG = "bnf_csapr2_ppi"

BASELINE_DIR = "baseline_images"

# Every plot name produced by quicklooks_ppi(return_figs=True). Must stay
# in lockstep with the dict keys assigned inside cmac_ppi_quicklooks.py.
# snow_rate_ws2012 is conditional on the field being present, so it's not
# in this list — add an explicit test for it once the test data triggers
# the snow-rate branch.
PPI_FIGURE_NAMES = [
    "reflectivity",
    "cmac_four_panel_plot",
    "masked_corrected_reflectivity",
    "corrected_reflectivity",
    "differential_phase",
    "specific_attenuation",
    "corrected_differential_phase",
    "corrected_specific_diff_phase",
    "corrected_velocity",
    "rain_rate_A",
    "rain_rate_Z",
    "rain_rate_Kdp",
    "filtered_corrected_differential_phase",
    "filtered_corrected_specific_diff_phase",
    "specific_differential_attenuation",
    "path_integrated_differential_attenuation",
    "corrected_differential_reflectivity",
    "normalized_coherent_power",
    "signal_to_noise_ratio",
]

# Every plot name produced by quicklooks_rhi(return_figs=True). Same
# co-evolution rule as PPI_FIGURE_NAMES. The RHI module produces a
# subset — no rain_rate_Z, rain_rate_Kdp, or snow_rate_ws2012.
RHI_FIGURE_NAMES = [
    "reflectivity",
    "cmac_four_panel_plot",
    "masked_corrected_reflectivity",
    "corrected_reflectivity",
    "differential_phase",
    "specific_attenuation",
    "corrected_differential_phase",
    "corrected_specific_diff_phase",
    "corrected_velocity",
    "rain_rate_A",
    "filtered_corrected_differential_phase",
    "filtered_corrected_specific_diff_phase",
    "specific_differential_attenuation",
    "path_integrated_differential_attenuation",
    "corrected_differential_reflectivity",
    "normalized_coherent_power",
    "signal_to_noise_ratio",
]


@pytest.fixture(scope="module")
def _downloaded_files(tmp_path_factory):
    """Download the radar + sonde once for the whole module."""
    username = os.getenv("ARM_USERNAME")
    token = os.getenv("ARM_PASSWORD")
    if not username or not token:
        pytest.skip("ARM credentials not set")

    data_dir = tmp_path_factory.mktemp("cmac_data")
    start, end = "2025-05-20T00:00:00", "2025-05-20T00:10:00"
    start_sonde, end_sonde = "2025-05-19T21:00:00", "2025-05-20T03:00:00"

    act.discovery.download_arm_data(
        username, token, "bnfcsapr2cfrS3.a1", start, end,
        output=str(data_dir),
    )
    act.discovery.download_arm_data(
        username, token, "bnfsondewnpnM1.b1", start_sonde, end_sonde,
        output=str(data_dir),
    )

    radar_files = sorted(glob.glob(
        str(data_dir / "**" / "bnfcsapr2cfrS3*"), recursive=True))
    sonde_files = sorted(glob.glob(
        str(data_dir / "**" / "bnfsondewnpnM1*"), recursive=True))
    assert len(radar_files) >= 2, (
        f"expected at least 2 radar files (RHI + PPI), got {len(radar_files)}")
    assert sonde_files, "no sonde file downloaded"

    # The bnfcsapr2cfrS3.a1 datastream returns: [0]=RHI scan, [1]=PPI scan.
    return {
        "rhi": radar_files[0],
        "ppi": radar_files[1],
        "sonde": sonde_files[0],
    }


def _run_cmac(radar_path, sonde_path):
    radar = pyart.io.read(radar_path)
    sonde = xr.open_dataset(sonde_path)
    try:
        return cmac(
            radar, sonde, RADAR_CONFIG,
            meta_append="config", verbose=False,
        )
    finally:
        sonde.close()


@pytest.fixture(scope="module")
def cmac_radar_ppi(_downloaded_files):
    return _run_cmac(_downloaded_files["ppi"], _downloaded_files["sonde"])


@pytest.fixture(scope="module")
def cmac_radar_rhi(_downloaded_files):
    return _run_cmac(_downloaded_files["rhi"], _downloaded_files["sonde"])


@pytest.fixture(scope="module")
def quicklook_figures_ppi(cmac_radar_ppi):
    """Build every PPI quicklook figure once, share across image-compare tests."""
    return quicklooks_ppi(
        cmac_radar_ppi, RADAR_CONFIG, sweep=0, return_figs=True)


@pytest.fixture(scope="module")
def quicklook_figures_rhi(cmac_radar_rhi):
    """Build every RHI quicklook figure once, share across image-compare tests."""
    return quicklooks_rhi(
        cmac_radar_rhi, RADAR_CONFIG, sweep=0, return_figs=True)


def test_cmac_processing_ppi(cmac_radar_ppi):
    expected_fields = {
        "sounding_temperature",
        "height",
        "velocity_texture",
        "gate_id",
        "corrected_reflectivity",
        "corrected_velocity",
    }
    missing = expected_fields - set(cmac_radar_ppi.fields)
    assert not missing, f"cmac() did not add expected fields: {missing}"


def test_cmac_processing_rhi(cmac_radar_rhi):
    expected_fields = {
        "sounding_temperature",
        "height",
        "velocity_texture",
        "gate_id",
        "corrected_reflectivity",
        "corrected_velocity",
    }
    missing = expected_fields - set(cmac_radar_rhi.fields)
    assert not missing, f"cmac() did not add expected fields: {missing}"


def test_quicklook_figure_set_complete_ppi(quicklook_figures_ppi):
    """Catch new PPI plots being added without a baseline test."""
    actual = set(quicklook_figures_ppi) - {"snow_rate_ws2012"}
    expected = set(PPI_FIGURE_NAMES)
    new = actual - expected
    missing = expected - actual
    assert not new, (
        f"new PPI quicklook(s) {new} present in figures dict but no baseline "
        f"test — add them to PPI_FIGURE_NAMES and regenerate baselines.")
    assert not missing, (
        f"PPI quicklook(s) {missing} listed in PPI_FIGURE_NAMES but not "
        f"produced by quicklooks_ppi — remove from PPI_FIGURE_NAMES or restore.")


def test_quicklook_figure_set_complete_rhi(quicklook_figures_rhi):
    """Catch new RHI plots being added without a baseline test."""
    actual = set(quicklook_figures_rhi)
    expected = set(RHI_FIGURE_NAMES)
    new = actual - expected
    missing = expected - actual
    assert not new, (
        f"new RHI quicklook(s) {new} present in figures dict but no baseline "
        f"test — add them to RHI_FIGURE_NAMES and regenerate baselines.")
    assert not missing, (
        f"RHI quicklook(s) {missing} listed in RHI_FIGURE_NAMES but not "
        f"produced by quicklooks_rhi — remove from RHI_FIGURE_NAMES or restore.")


def _make_baseline_test(scan, name, figures_fixture):
    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR,
        filename=f"{scan}_{name}.png",
        tolerance=10,
    )
    def _test(request):
        figures = request.getfixturevalue(figures_fixture)
        return figures[name]
    _test.__name__ = f"test_quicklook_{scan}_{name}"
    _test.__doc__ = (
        f"pytest-mpl baseline comparison for the {name!r} {scan.upper()} "
        f"quicklook.")
    return _test


# Bind one image-compare test per figure into module globals so pytest
# discovers them as test_quicklook_ppi_<name> / test_quicklook_rhi_<name>.
# Baseline filenames are ppi_<name>.png / rhi_<name>.png so the two scan
# strategies share one baseline_dir without colliding.
for _name in PPI_FIGURE_NAMES:
    globals()[f"test_quicklook_ppi_{_name}"] = _make_baseline_test(
        "ppi", _name, "quicklook_figures_ppi")
for _name in RHI_FIGURE_NAMES:
    globals()[f"test_quicklook_rhi_{_name}"] = _make_baseline_test(
        "rhi", _name, "quicklook_figures_rhi")
del _name
