import json
from pykml import parser
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import openeo
import geojson

# PATH_TO_KML = 'lands/Corn.kml'
# PATH_TO_KML = 'lands/machuhi.kml'\
PATH_TO_KML = 'lands/mach_test.kml'

def plot_timeseries(filename, metadata, figsize=(6, 3)):
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    parcel_ids = df['feature_index'].unique()

    for parcel_id in parcel_ids:
        parcel_data = df[df['feature_index'] == parcel_id]
        parcel_data = parcel_data.sort_index()

        fig, ax = plt.subplots(figsize=figsize, dpi=90)
        plt.subplots_adjust(right=0.7)
        ax.plot(parcel_data.index, parcel_data["band_unnamed"], 'o-', linewidth=1, markersize=4)
        plt.xticks(rotation=90)

        max_value_row = parcel_data.loc[parcel_data["band_unnamed"].idxmax()]
        ax.plot(max_value_row.name, max_value_row["band_unnamed"], "ro")
        ax.annotate(
            f"Max NDVI: {max_value_row['band_unnamed']:.2f}",
            xy=(max_value_row.name, max_value_row["band_unnamed"]),
            xytext=(10, -10),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        )

        name = metadata[str(parcel_id)]['name']
        ax.set_title(f"name: {name}, Parcel ID: {parcel_id}")
        ax.set_ylabel("NDVI")
        ax.set_ylim(0, 1)

        text_ax = fig.add_axes([0.75, 0.1, 0.2, 0.8])
        text_ax.axis('off')
        metadata_text = "\n".join([
            f"{key}: {value}" for key, value in metadata[str(parcel_id)].items()
        ])
        text_ax.text(
            0, 0.5, metadata_text, ha="left", va="center", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
        )

        fig.savefig(f'id_{parcel_id}_{name}.pdf', bbox_inches="tight")
        plt.close(fig)

kml_file = path.join(PATH_TO_KML)
fields = {
    'type': "FeatureCollection",
    'features': []
}
metadata = {}

with open(kml_file, 'r', encoding='utf-8') as f:
    doc = parser.parse(f).getroot()
    parcel_id_idx = 0
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    for item in doc.findall('.//kml:Placemark', namespaces=ns):
        placemark_name_elem = item.find('.//kml:name', namespaces=ns)
        placemark_name = placemark_name_elem.text if placemark_name_elem is not None else None

        coords_elem = item.find('.//kml:LinearRing/kml:coordinates', namespaces=ns)
        if coords_elem is None:
            continue  # Skip if there are no coordinates
        coords_text = coords_elem.text
        cords = [
            [float(coord) for coord in c.split(',')[:2]]
            for c in coords_text.strip().split() if c
        ]
        if cords[0] != cords[-1]:
            cords.append(cords[0])

        extended_data = item.find('.//kml:ExtendedData', namespaces=ns)
        parcel_metadata = {}
        if extended_data is not None:
            for data in extended_data.findall('.//kml:Data', namespaces=ns):
                name = data.get('name')
                value_elem = data.find('.//kml:value', namespaces=ns)
                value = value_elem.text if value_elem is not None else None
                parcel_metadata[name] = value

        if placemark_name:
            metadata[str(parcel_id_idx)] = {
                "name": placemark_name, **parcel_metadata
            }
            polygon = geojson.Polygon([cords])
            feature = geojson.Feature(
                geometry=polygon,
                properties={"name": placemark_name, **parcel_metadata}
            )
            fields['features'].append(feature)
            parcel_id_idx += 1

if not fields['features']:
    raise ValueError("No features found in the provided KML file.")

# Pass the GeoJSON dictionary directly to aggregate_spatial
geometries = fields

connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()

s2cube = connection.load_collection(
    "SENTINEL2_L2A",
    temporal_extent=["2023-01-01", "2024-09-18"],
    bands=["B04", "B08", "SCL"],
)

red = s2cube.band("B04")
nir = s2cube.band("B08")
ndvi = (nir - red) / (nir + red)

scl = s2cube.band("SCL")
mask = ~((scl == 4) | (scl == 5))

# 2D gaussian kernel
g = scipy.signal.windows.gaussian(11, std=1.6)
kernel = np.outer(g, g)
kernel = kernel / kernel.sum()

# Morphological dilation of mask: convolution + threshold
mask = mask.apply_kernel(kernel)
mask = mask > 0.1

ndvi_masked = ndvi.mask(mask)

# Pass the GeoJSON dictionary directly
timeseries_masked = ndvi_masked.aggregate_spatial(
    geometries=geometries,
    reducer="mean"
)

udf_code = """
from scipy.signal import savgol_filter
from openeo.udf import XarrayDataCube
import xarray

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    array = cube.get_array()
    filled = array.interpolate_na(dim='t')
    smoothed_array = savgol_filter(filled.values, 5, 2, axis=0)
    return XarrayDataCube(xarray.DataArray(
        smoothed_array, dims=array.dims, coords=array.coords
    ))
"""

udf = openeo.UDF(udf_code)
ndvi_smoothed = ndvi_masked.apply_dimension(code=udf, dimension="t")

timeseries_smoothed = ndvi_smoothed.aggregate_spatial(
    geometries=geometries,
    reducer="mean"
)

job = timeseries_smoothed.execute_batch(
    out_format="CSV", title="Smoothed NDVI timeseries"
)
job.get_results().download_file("ndvi-results/timeseries-masked.csv")

plot_timeseries("ndvi-results/timeseries-masked.csv", metadata)
