""" Code that plots fields from the CMAC radar object. """

import os
from datetime import datetime
import operator

import cartopy.crs as ccrs
import netCDF4
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyart
import cmweather

from pyart.graph.common import (
    generate_radar_name, generate_radar_time_begin)

from .config import get_plot_values, get_field_names

plt.switch_backend('agg')


def quicklooks_ppi(radar, config, sweep=None, image_directory=None,
                   dd_lobes=True, config_file=None):
    """
    Quicklooks PPI, images produced with regards to CMAC

    Parameter
    ---------
    radar : Radar
        Radar object that has CMAC applied to it.
    config : str
        A string of the radar name found from config.py that contains values
        for plotting, specific to that radar.

    Optional Parameters
    -------------------
    image_directory : str
        File path to the image folder of which to save the CMAC images. If no
        image file path is given, image path defaults to users home directory.
    dd_lobes : bool
        Plot DD lobes between radars if dd_lobes is True.
    config_file : str or None
        Path to a YAML file whose values override the built-in defaults for
        the named ``config`` radar.

    """
    if image_directory is None:
        image_directory = os.path.expanduser('~')

    radar_start_date = netCDF4.num2date(
        radar.time['data'][0], radar.time['units'],
        only_use_cftime_datetimes=False, only_use_python_datetimes=True)

    # Retrieve the plot parameter values based on the radar.
    plot_config = get_plot_values(config, config_file=config_file)
    field_config = get_field_names(config, config_file=config_file)
    save_name = plot_config['save_name']
    date_string = datetime.strftime(radar_start_date, '%Y%m%d.%H%M%S')
    combined_name = '.' + save_name + '.' + date_string

    # Soft-coded plot layout / range defaults — every key falls back to
    # its previous hard-coded value if the YAML config does not provide it.
    figsize_single = plot_config.get('figsize_single', [12, 8])
    figsize_panel = plot_config.get('figsize_panel', [15, 10])
    tick_spacing = plot_config.get('lat_lon_tick_spacing', 0.8)
    dd_grid_spacing = plot_config.get('dd_lobe_grid_spacing', 0.01)
    dd_bca_levels = plot_config.get(
        'dd_lobe_bca_levels', [np.pi / 6, 5 * np.pi / 6])
    sweep_fallback_nsweeps_lt = plot_config.get(
        'sweep_fallback_nsweeps_lt', 4)
    sweep_fallback = plot_config.get('sweep_fallback', 2)
    cat_colors_cfg = dict(plot_config.get(
        'cat_colors', {
            'rain': 'green',
            'multi_trip': 'red',
            'no_scatter': 'gray',
            'snow': 'cyan',
            'melting': 'yellow',
            'clutter': 'black',
            'terrain_blockage': 'brown'}))

    def _range(key, default):
        return plot_config.get(key, default)

    if 'min_lat' in plot_config.keys():
        min_lat = plot_config['min_lat']
    else:
        min_lat = radar.gate_latitude['data'].min() - .1

    if 'max_lat' in plot_config.keys():
        max_lat = plot_config['max_lat']
    else:
        max_lat = radar.gate_latitude['data'].max() + .1

    if 'min_lon' in plot_config.keys():
        min_lon = plot_config['min_lon']
    else:
        min_lon = radar.gate_longitude['data'].min() - .1

    if 'max_lon' in plot_config.keys():
        max_lon = plot_config['max_lon']
    else:
        max_lon = radar.gate_longitude['data'].max() + .1

    #max_lat = radar.gate_latitude['data'].max() + .1
    #min_lat = radar.gate_latitude['data'].min() - .1
    #max_lon = radar.gate_longitude['data'].max() + .1
    #min_lon = radar.gate_longitude['data'].min() - .1

    # Creating a plot of reflectivity before CMAC.
    lal = np.arange(min_lat, max_lat, tick_spacing)
    lol = np.arange(min_lon, max_lon, tick_spacing)

    if dd_lobes:
        grid_lat = np.arange(min_lat, max_lat, dd_grid_spacing)
        grid_lon = np.arange(min_lon, max_lon, dd_grid_spacing)

        facility = plot_config['facility']
        if facility == 'I4':
            dms_radar1_coords = [plot_config['site_i4_dms_lon'],
                                 plot_config['site_i4_dms_lat']]
            dms_radar2_coords = [plot_config['site_i5_dms_lon'],
                                 plot_config['site_i5_dms_lat']]
        elif facility == 'I5':
            dms_radar1_coords = [plot_config['site_i5_dms_lon'],
                                 plot_config['site_i5_dms_lat']]
            dms_radar2_coords = [plot_config['site_i4_dms_lon'],
                                 plot_config['site_i4_dms_lat']]
        elif facility == 'I6':
            dms_radar1_coords = [plot_config['site_i6_dms_lon'],
                                 plot_config['site_i6_dms_lat']]
            dms_radar2_coords = [plot_config['site_i4_dms_lon'],
                                 plot_config['site_i4_dms_lat']]

        dec_radar1 = [_dms_to_decimal(
            dms_radar1_coords[0][0], dms_radar1_coords[0][1],
            dms_radar1_coords[0][2]), _dms_to_decimal(
                dms_radar1_coords[1][0], dms_radar1_coords[1][1],
                dms_radar1_coords[1][2])]
        dec_radar2 = [_dms_to_decimal(
            dms_radar2_coords[0][0], dms_radar2_coords[0][1],
            dms_radar2_coords[0][2]), _dms_to_decimal(
                dms_radar2_coords[1][0], dms_radar2_coords[1][1],
                dms_radar2_coords[1][2])]

        bca = _get_bca(dec_radar2[0], dec_radar2[1], dec_radar1[0],
                       dec_radar1[1], grid_lon, grid_lat)
        grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    if sweep is None:
        if radar.nsweeps < sweep_fallback_nsweeps_lt:
            sweep = sweep_fallback
        else:
            sweep = plot_config['sweep']

    # Plot of the raw reflectivity from the radar.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1,
                           subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map(field_config['reflectivity'], sweep=sweep, resolution='50m', ax=ax,
                         vmin=_range('reflectivity_raw_vmin', -8),
                         vmax=_range('reflectivity_raw_vmax', 64),
                         mask_outside=False,
                         cmap=pyart.graph.cm_colorblind.HomeyerRainbow,
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')

    fig.savefig(
        image_directory
        + '/reflectivity' + combined_name + '.png')
    del fig, ax

    # Four panel plot of gate_id, velocity_texture, reflectivity, and
    # cross_correlation_ratio.
    cat_dict = {}
    print('##')
    print('## Keys for each gate id are as follows:')
    for i, pair_str in enumerate(radar.fields['gate_id']['flag_meanings'].split(' ')):
        print('##   ', str(pair_str))
        cat_dict.update({pair_str: i})
    sorted_cats = sorted(cat_dict.items(), key=operator.itemgetter(1))
    cat_colors = dict(cat_colors_cfg)
    lab_colors = [cat_colors[kitty[0]] for kitty in sorted_cats]
    cmap = matplotlib.colors.ListedColormap(lab_colors)

    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(2, 2,
              figsize=figsize_panel, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax[0, 0].set_aspect('auto')
    display.plot_ppi_map('gate_id', sweep=sweep, min_lon=min_lon, ax=ax[0, 0],
                         max_lon=max_lon, min_lat=min_lat,
                         max_lat=max_lat, resolution='50m',
                         lat_lines=lal, lon_lines=lol, cmap=cmap,
                         vmin=0, vmax=6, projection=ccrs.PlateCarree())

    if dd_lobes:
        ax[0, 0].contour(grid_lon, grid_lat, bca,
                        levels=dd_bca_levels, linewidths=2,
                        colors='k')

    cbax = ax[0, 0]
    if 'ground_clutter' in radar.fields.keys() or 'terrain_blockage' in radar.fields['gate_id']['notes']:
        tick_locs = np.linspace(
            0, len(sorted_cats) - 1, len(sorted_cats)) + 0.5
    else:
        tick_locs = np.linspace(
            0, len(sorted_cats), len(sorted_cats)) + 0.5
    display.cbs[-1].locator = matplotlib.ticker.FixedLocator(tick_locs)
    catty_list = [sorted_cats[i][0] for i in range(len(sorted_cats))]
    display.cbs[-1].formatter = matplotlib.ticker.FixedFormatter(catty_list)
    display.cbs[-1].update_ticks()
    ax[0, 1].set_aspect('auto')
    display.plot_ppi_map(field_config['reflectivity'], sweep=sweep,
                         vmin=_range('reflectivity_vmin', -8),
                         vmax=_range('reflectivity_vmax', 40.0),
                         ax=ax[0, 1], min_lon=min_lon, max_lon=max_lon,
                         min_lat=min_lat,
                         max_lat=max_lat, lat_lines=lal, lon_lines=lol,
                         resolution='50m',
                         cmap=pyart.graph.cm_colorblind.HomeyerRainbow,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax[0, 1].contour(grid_lon, grid_lat, bca,
                        levels=dd_bca_levels, linewidths=2,
                        colors='k')
    ax[1, 0].set_aspect('auto')
    display.plot_ppi_map('velocity_texture', sweep=sweep,
                         vmin=_range('velocity_texture_vmin', 0),
                         vmax=_range('velocity_texture_vmax', 14),
                         min_lon=min_lon, max_lon=max_lon, min_lat=min_lat,
                         max_lat=max_lat, lat_lines=lal, lon_lines=lol,
                         resolution='50m', ax=ax[1, 0],
                         title=_generate_title(
                             radar, 'velocity_texture', sweep),
                         cmap=pyart.graph.cm.NWSRef,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax[1, 0].contour(grid_lon, grid_lat, bca, latlon='True',
                        levels=dd_bca_levels, linewidths=2,
                         colors='k')
    
    rhv_field = field_config['cross_correlation_ratio']
    ax[1, 1].set_aspect('auto')
    display.plot_ppi_map(rhv_field, sweep=sweep,
                         vmin=_range('cross_correlation_ratio_vmin', .5),
                         vmax=_range('cross_correlation_ratio_vmax', 1),
                         min_lon=min_lon, max_lon=max_lon,
                         min_lat=min_lat, max_lat=max_lat, lat_lines=lal,
                         lon_lines=lol, resolution='50m', ax=ax[1, 1],
                         cmap=pyart.graph.cm.Carbone42,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax[1, 1].contour(grid_lon, grid_lat, bca,
                        levels=dd_bca_levels, linewidths=2,
                        colors='k')
    fig.savefig(
        image_directory
        + '/cmac_four_panel_plot' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot with reflectivity corrected with gate ids.
    cmac_gates = pyart.correct.GateFilter(radar)
    cmac_gates.exclude_all()
    cmac_gates.include_equal('gate_id', cat_dict['rain'])
    cmac_gates.include_equal('gate_id', cat_dict['melting'])
    cmac_gates.include_equal('gate_id', cat_dict['snow'])

    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map(field_config['reflectivity'],
                         sweep=sweep, resolution='50m',
                         vmin=_range('reflectivity_vmin', -8),
                         vmax=_range('reflectivity_vmax', 40),
                         mask_outside=False,
                         cmap=pyart.graph.cm_colorblind.HomeyerRainbow,
                         title=_generate_title(
                             radar, 'masked_corrected_reflectivity',
                             sweep), ax=ax,
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol,
                         gatefilter=cmac_gates,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/masked_corrected_reflectivity' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display


    # Creating a plot with reflectivity corrected with attenuation.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('corrected_reflectivity', sweep=sweep,
                         vmin=_range('corrected_reflectivity_vmin', 0),
                         vmax=_range('corrected_reflectivity_vmax', 40.0),
                         resolution='50m',
                         title=_generate_title(
                             radar, 'corrected_reflectivity',
                             sweep),
                         cmap=pyart.graph.cm_colorblind.HomeyerRainbow,
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol, ax=ax,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/corrected_reflectivity' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot with differential phase.
    phase_field = field_config['input_phidp_field']
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map(phase_field, sweep=sweep,
                         resolution='50m', ax=ax,
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol,
                         projection=ccrs.PlateCarree())
    fig.savefig(
        image_directory
        + '/differential_phase' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display


    # Creating a plot of specific attenuation.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('specific_attenuation', sweep=sweep,
                         vmin=_range('specific_attenuation_vmin', 0),
                         vmax=_range('specific_attenuation_vmax', 1.0),
                         resolution='50m', ax=ax,
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                    levels=dd_bca_levels, linewidths=2,
                    colors='k')
    fig.savefig(
        image_directory
        + '/specific_attenuation' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot of corrected differential phase.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('corrected_differential_phase', sweep=sweep,
                         title=_generate_title(
                             radar, 'corrected_differential_phase',
                             sweep), ax=ax,
                         resolution='50m', min_lat=min_lat,
                         min_lon=min_lon, max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/corrected_differential_phase' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot of corrected specific differential phase.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('corrected_specific_diff_phase', sweep=sweep,
                         vmin=_range('corrected_specific_diff_phase_vmin', 0),
                         vmax=_range('corrected_specific_diff_phase_vmax', 6),
                         resolution='50m',
                         title=_generate_title(
                             radar, 'corrected_specific_diff_phase',
                             sweep), ax=ax,
                         min_lat=min_lat, min_lon=min_lon, max_lat=max_lat,
                         max_lon=max_lon, lat_lines=lal, lon_lines=lol,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/corrected_specific_diff_phase' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot with region dealias corrected velocity.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('corrected_velocity', sweep=sweep, resolution='50m',
                         cmap='balance',
                         vmin=_range('corrected_velocity_vmin', -60),
                         vmax=_range('corrected_velocity_vmax', 60),
                         ax=ax, min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, max_lon=max_lon, lat_lines=lal,
                         lon_lines=lol, projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/corrected_velocity' + combined_name + '.png')
    plt.close(fig) 
    del fig, ax, display

    # Creating a plot of rain rate A
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('rain_rate_A', sweep=sweep, resolution='50m',
                         vmin=_range('rain_rate_vmin', 0),
                         vmax=_range('rain_rate_vmax', 120),
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, ax=ax, max_lon=max_lon, lat_lines=lal,
                         lon_lines=lol, projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/rain_rate_A' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display
    # Creating a plot of rain rate Z
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('rain_rate_Z', sweep=sweep, resolution='50m',
                         vmin=_range('rain_rate_vmin', 0),
                         vmax=_range('rain_rate_vmax', 120),
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, ax=ax, max_lon=max_lon, lat_lines=lal,
                         lon_lines=lol, projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/rain_rate_Z' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display
    # Creating a plot of rain rate A
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('rain_rate_Kdp', sweep=sweep, resolution='50m',
                         vmin=_range('rain_rate_vmin', 0),
                         vmax=_range('rain_rate_vmax', 120),
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, ax=ax, max_lon=max_lon, lat_lines=lal,
                         lon_lines=lol, projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/rain_rate_Kdp' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot of snowfall rate from Wolf and Snider
    if 'snow_rate_ws2012' in radar.fields.keys():
        display = pyart.graph.RadarMapDisplay(radar)
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                              figsize=figsize_single)
        ax.set_aspect('auto')
        display.plot_ppi_map('snow_rate_ws2012', sweep=sweep, resolution='50m',
                             vmin=_range('snow_rate_vmin', 0),
                             vmax=_range('snow_rate_vmax', 50),
                             min_lat=min_lat, min_lon=min_lon,
                             max_lat=max_lat, ax=ax, max_lon=max_lon, lat_lines=lal,
                             lon_lines=lol, projection=ccrs.PlateCarree())
        if dd_lobes:
            ax.contour(grid_lon, grid_lat, bca,
                       levels=dd_bca_levels, linewidths=2,
                       colors='k')
        fig.savefig(
             image_directory
            + '/snow_rate_ws2012' + combined_name + '.png')
        plt.close(fig)
        del fig, ax, display

    # Creating a plot of filtered corrected differential phase.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('filtered_corrected_differential_phase', sweep=sweep,
                         title=_generate_title(
                             radar, 'filtered_corrected_differential_phase',
                             sweep),
                         resolution='50m', min_lat=min_lat, ax=ax,
                         min_lon=min_lon, max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol,
                         vmin=_range('filtered_corrected_differential_phase_vmin', 0),
                         vmax=_range('filtered_corrected_differential_phase_vmax', 360),
                         cmap=pyart.graph.cm.Theodore16,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                    levels=dd_bca_levels, linewidths=2,
                    colors='k')
    fig.savefig(
        image_directory
        + '/filtered_corrected_differential_phase' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot of filtered corrected specific differential phase.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('filtered_corrected_specific_diff_phase', sweep=sweep,
                         title=_generate_title(
                             radar, 'filtered_corrected_specific_diff_phase',
                             sweep), ax=ax,
                         resolution='50m', min_lat=min_lat,
                         min_lon=min_lon, max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol,
                         vmin=_range('filtered_corrected_specific_diff_phase_vmin', -2),
                         vmax=_range('filtered_corrected_specific_diff_phase_vmax', 10),
                         cmap=pyart.graph.cm.Theodore16,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                    levels=dd_bca_levels, linewidths=2,
                    colors='k')
    fig.savefig(
        image_directory
        + '/filtered_corrected_specific_diff_phase' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot of corrected differential phase.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('specific_differential_attenuation', sweep=sweep,
                         title=_generate_title(
                             radar, 'specific_differential_attenuation',
                             sweep), ax=ax,
                         resolution='50m', min_lat=min_lat,
                         min_lon=min_lon, max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol, gatefilter=cmac_gates,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/specific_differential_attenuation' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot of corrected differential phase.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('path_integrated_differential_attenuation',
                         sweep=sweep,
                         title=_generate_title(
                             radar, 'path_integrated_differential_attenuation',
                             sweep), ax=ax,
                         resolution='50m', min_lat=min_lat,
                         min_lon=min_lon, max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol, gatefilter=cmac_gates,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/path_integrated_differential_attenuation' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot of corrected differential phase.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                          figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('corrected_differential_reflectivity', sweep=sweep,
                         title=_generate_title(
                             radar, 'corrected_differential_reflectivity',
                             sweep), ax=ax,
                         resolution='50m', min_lat=min_lat,
                         min_lon=min_lon, max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol, gatefilter=cmac_gates,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                    levels=dd_bca_levels, linewidths=2,
                    colors='k')
    fig.savefig(
        image_directory
        + '/corrected_differential_reflectivity' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

    # Creating a plot with reflectivity corrected with attenuation.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map(field_config['normalized_coherent_power'], sweep=sweep,
                         resolution='50m',
                         title=_generate_title(
                             radar, field_config['normalized_coherent_power'],
                             sweep),
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol, ax=ax,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/normalized_coherent_power' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display

   # Creating a plot with reflectivity corrected with attenuation.
    display = pyart.graph.RadarMapDisplay(radar)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=figsize_single)
    ax.set_aspect('auto')
    display.plot_ppi_map('signal_to_noise_ratio', sweep=sweep,
                         resolution='50m',
                         title=_generate_title(
                             radar, 'signal_to_noise_ratio',
                             sweep),
                         min_lat=min_lat, min_lon=min_lon,
                         max_lat=max_lat, max_lon=max_lon,
                         lat_lines=lal, lon_lines=lol, ax=ax,
                         projection=ccrs.PlateCarree())
    if dd_lobes:
        ax.contour(grid_lon, grid_lat, bca,
                   levels=dd_bca_levels, linewidths=2,
                   colors='k')
    fig.savefig(
        image_directory
        + '/signal_to_noise_ratio' + combined_name + '.png')
    plt.close(fig)
    del fig, ax, display


def _generate_title(radar, field, sweep):
    """ Generates a title for each plot. """
    time_str = generate_radar_time_begin(radar).isoformat() + 'Z'
    fixed_angle = radar.fixed_angle['data'][sweep]
    line_one = "%s %.1f Deg. %s " % (generate_radar_name(radar), fixed_angle,
                                     time_str)
    field_name = str(field)
    field_name = field_name.replace('_', ' ')
    field_name = field_name[0].upper() + field_name[1:]
    return line_one + '\n' + field_name


def _get_bca(rad1_lon, rad1_lat, rad2_lon, rad2_lat,
             grid_lon, grid_lat):
    # Beam crossing angle needs cartesian coordinate.
    p = ccrs.PlateCarree()
    p = p.as_geocentric()
    rad1 = p.transform_points(ccrs.PlateCarree().as_geodetic(),
                              np.array(rad1_lon),
                              np.array(rad1_lat))
    rad2 = p.transform_points(ccrs.PlateCarree().as_geodetic(),
                              np.array(rad2_lon),
                              np.array(rad2_lat))
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    grid = p.transform_points(ccrs.PlateCarree().as_geodetic(),
                              grid_lon, grid_lat,
                              np.zeros(grid_lon.shape))

    # Create grid with Radar 1 in center.
    x = grid[:, :, 0] - rad1[0, 0]
    y = grid[:, :, 1] - rad1[0, 1]
    rad2 = rad2 - rad1
    a = np.sqrt(np.multiply(x, x) + np.multiply(y, y))
    b = np.sqrt(pow(x - rad2[0, 0], 2) + pow(y - rad2[0, 1], 2))
    c = np.sqrt(rad2[0, 0] * rad2[0, 0] + rad2[0, 1] * rad2[0, 1])
    theta_1 = np.arccos(x/a)
    theta_2 = np.arccos((x - rad2[0, 1]) / b)
    return np.arccos((a*a + b*b - c*c) / (2*a*b))


def _dms_to_decimal(degrees, minutes, seconds):
    if degrees > 0:
        return degrees + minutes/60 + seconds/3600
    else:
        return degrees - minutes/60 - seconds/3600
