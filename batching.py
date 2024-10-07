from pykml import parser
from pykml.factory import KML_ElementMaker as KML
from lxml import etree
import os

kml_file = 'lands/machuhi.kml'  # Path to your KML file
output_folder = 'batches'
batch_size = 10

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(kml_file, 'r', encoding='utf-8') as f:
    doc = parser.parse(f).getroot()

placemarks = doc.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
unique_coords = set()
unique_placemarks = []

for placemark in placemarks:
    coordinates = placemark.findall('.//{http://www.opengis.net/kml/2.2}coordinates')

    if coordinates:
        coords_text = coordinates[0].text.strip()
        coords_tuple = tuple(coords_text.split())

        # If  coordinates are unique, keep the placemark
        if coords_tuple not in unique_coords:
            unique_coords.add(coords_tuple)
            unique_placemarks.append(placemark)

total_batches = len(unique_placemarks) // batch_size + (1 if len(unique_placemarks) % batch_size else 0)

for batch_idx in range(total_batches):
    new_doc = KML.kml(
        KML.Document(
            # Optionally include styles or other elements from the original doc
        )
    )

    batch_placemarks = unique_placemarks[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    for placemark in batch_placemarks:
        new_doc.Document.append(placemark)

    batch_file_path = os.path.join(output_folder, f'batch_{batch_idx + 1}.kml')
    with open(batch_file_path, 'w', encoding='utf-8') as batch_file:
        # Properly write the XML string
        batch_file.write(etree.tostring(new_doc, pretty_print=True, encoding='unicode'))

print(
    f'Successfully removed duplicates and created {total_batches} batches, each containing up to {batch_size} unique placemarks.')
