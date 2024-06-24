from pykml import parser
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import openeo


PATH_TO_KML = 'legit4in1.kml'


def plot_timeseries(filename, metadata, figsize=(6, 3)):

    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    parcel_ids = df['feature_index'].unique()

    for parcel_id in parcel_ids:
        parcel_data = df[df['feature_index'] == parcel_id]

        # Ensure the data is sorted by date to avoid random connections
        parcel_data = parcel_data.sort_index()

        fig, ax = plt.subplots(figsize=figsize, dpi=90)
        plt.subplots_adjust(right=0.7)

        # Plotting with both dots and lines by specifying 'o-' as the style
        ax.plot(parcel_data.index, parcel_data["avg(band_0)"], 'o-', linewidth=1, markersize=4)

        plt.xticks(rotation=90)

        max_value_row = parcel_data.loc[parcel_data["avg(band_0)"].idxmax()]
        ax.plot(max_value_row.name, max_value_row["avg(band_0)"], "ro")  # Highlight the max value
        ax.annotate(f"Max NDVI: {max_value_row['avg(band_0)']:.2f}",
                    xy=(max_value_row.name, max_value_row["avg(band_0)"]),
                    xytext=(10, -10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        name = metadata[str(parcel_id)]['name']
        ax.set_title(f"name: {name}, Parcel ID: {parcel_id}")
        ax.set_ylabel("NDVI")
        ax.set_ylim(0, 1)

        text_ax = fig.add_axes([0.75, 0.1, 0.2, 0.8])
        text_ax.axis('off')
        metadata_text = "\n".join([f"{key}: {value}" for key, value in metadata[str(parcel_id)].items()])
        text_ax.text(0, 0.5, metadata_text, ha="left", va="center", fontsize=9,
                     bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})

        # Save the figure with parcel_id and KOATUU value in the filename
        fig.savefig(f'id_{parcel_id}_{name}.pdf', bbox_inches="tight")
        plt.close(fig)


kml_file = path.join(PATH_TO_KML)
fields = {
    'type': "FeatureCollection",
    'features': []
}
metadata = {}

with open(kml_file) as f:
    doc = parser.parse(f).getroot()
    parcel_id_idx = 0
    for item in doc.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
        placemark_name = item.find('.//{http://www.opengis.net/kml/2.2}name').text if item.find(
                                                      './/{http://www.opengis.net/kml/2.2}name') is not None else None

        coords_text = item.MultiGeometry.Polygon.outerBoundaryIs.LinearRing.coordinates.text
        cords = [list(map(float, c.split(','))) for c in coords_text.strip().split(' ') if c]
        extended_data = item.ExtendedData
        parcel_metadata = {data.attrib['name']: data.value.text for data in
                           extended_data.findall('.//{http://www.opengis.net/kml/2.2}Data')}
        if placemark_name and placemark_name.startswith("Объект ID"):
            metadata[str(parcel_id_idx)] = {"name": placemark_name, **parcel_metadata}
            fields['features'].append({
                "type": "Feature",
                "properties": {"name": placemark_name, **parcel_metadata},
                "geometry": {"type": "Polygon", "coordinates": [cords]}
            })
            parcel_id_idx += 1

# print('metadata: ', metadata)
# print('fields: ', fields)
# 158, 159

connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()

s2cube = connection.load_collection(
    "SENTINEL2_L2A",
    temporal_extent= ["2023-01-01", "2023-12-31"],
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
timeseries_masked = ndvi_masked.aggregate_spatial(geometries=fields, reducer="mean")


udf = openeo.UDF(
    """
from scipy.signal import savgol_filter
from openeo.udf import XarrayDataCube

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    array = cube.get_array()
    filled = array.interpolate_na(dim='t')
    smoothed_array = savgol_filter(filled.values, 5, 2, axis=0)
    return DataCube(xarray.DataArray(smoothed_array, dims=array. dims,coords=array.coords))
"""
)
ndvi_smoothed = ndvi_masked.apply_dimension(code=udf, dimension="t")
timeseries_smoothed = ndvi_smoothed.aggregate_spatial(geometries=fields, reducer="mean")

job = timeseries_smoothed.execute_batch(
    out_format="CSV", title="Smoothed NDVI timeseries"
)
job.get_results().download_file("ndvi-results/timeseries-masked.csv")

plot_timeseries("ndvi-results/timeseries-masked.csv", metadata)
