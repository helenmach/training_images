import fiona
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping
from sklearn.ensemble import RandomForestClassifier
import time


def get_filtered_data():
    shapefile_prodes = r'Recorte_Prodes\df_prodes_recortado_wgs.shp'

    with fiona.open(shapefile_prodes, 'r') as shp:
        print("Projeção do polígono:", shp.crs)

    df_prodes = gpd.read_file(shapefile_prodes)

    uuid = ["6d97fb77-8490-43b8-8d2b-881e948111b8",
            "d0a92248-0f25-48f5-ba36-d17a09400c4c",
            "63a0bdbb-85c0-4f29-8d2f-251d9db242ca",
            "6737421d-1f17-4cd3-9d61-82b74a5d7589"]  # corte raso com vegetação

    filtered_df = df_prodes[df_prodes['uuid'].isin(uuid)]
    return filtered_df[['uuid', 'geometry']]

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min) * 255).astype(np.uint8)

def extract_training_data(rgb_data, seeds, src):
    training_data = []
    training_labels = []

    for _, row in seeds.iterrows():
        mask_geom = [mapping(row['geometry'])]
        try:
            out_image, out_transform = mask(src, mask_geom, crop=True)
            mask_index = (out_image != src.nodata).all(axis=0)
            values = out_image[:, mask_index].reshape(3, -1).T  # RGB bands, reshape to (num_pixels, 3)
            training_data.extend(values)
            training_labels.extend(np.full(values.shape[0], row['uuid'], dtype=str))
        except ValueError:
            print(f"Skipping polygon with UUID {row['uuid']} as it does not intersect with the RGB image.")

    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    print(training_data)
    print(training_labels)
    return training_data, training_labels

def classify_image(image, training_data, training_labels):
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(training_data, training_labels)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration:.2f} seconds.")
    
    classified_image = rf.predict(image.reshape(-1, 3))
    return classified_image.reshape(image.shape[1:])

def save_classified_image(classified_image, profile, output_path):
    profile.update(dtype=rasterio.uint16, count=1, nodata=None)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(classified_image.astype(rasterio.uint16), 1)

def save_mapped_polygon(classified_image, profile, output_path, seeds):
    mapped_mask = np.isin(classified_image, seeds['uuid'].values)
    
    geoms = rasterio.features.shapes(mapped_mask.astype(np.uint8), transform=profile['transform'])
    
    schema = {'geometry': 'Polygon', 'properties': {'class': 'str'}}
    with fiona.open(output_path, 'w', crs=profile['crs'], driver='ESRI Shapefile', schema=schema) as dst:
        for geom, value in geoms:
            if geom['type'] == 'Polygon':
                dst.write({'geometry': geom, 'properties': {'class': str(value)}})

def main():
    seeds = get_filtered_data()
    image_path = 'Imagem_landsat/bands_4_5_6.tif'

    with rasterio.open(image_path) as src:
        print("Projeção do raster:", src.crs)
        raster_data = src.read([1, 2, 3])  # Lê as bandas 4, 5 e 6 como RGB
        raster_transform = src.transform
        raster_nodata = src.nodata
        profile = src.profile

        raster_crs = src.crs
        shapefile_crs = seeds.crs

        if raster_crs != shapefile_crs:
            print("Reprojetando shapefile para corresponder ao CRS do raster")
            seeds = seeds.to_crs(raster_crs)

        # Normalizar as bandas para o intervalo [0, 255]
        raster_data = np.array([normalize(band) for band in raster_data])
        print(raster_data)
        
        fig, ax = plt.subplots(figsize=(10, 10))

        show(raster_data, transform=raster_transform, ax=ax)
        seeds.plot(ax=ax, facecolor='none', edgecolor='red')

        plt.show()

        # Criar a máscara de treinamento
        training_data, training_labels = extract_training_data(raster_data, seeds, src)
        print("Training data shape:", training_data.shape)
        print("Training labels shape:", training_labels.shape)

        classified_image = classify_image(raster_data, training_data, training_labels)

        # Mapear classes para valores numéricos
        unique_labels = np.unique(training_labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_classified_image = np.vectorize(label_mapping.get)(classified_image)

        # Salvar a imagem classificada
        save_classified_image(numeric_classified_image, profile, 'classified_ndvi.tif')

        # Salvar o polígono mapeado
        save_mapped_polygon(classified_image, profile, 'mapped_polygon.shp', seeds)

        # Exibir a imagem classificada
        plt.figure(figsize=(10, 10))
        plt.imshow(numeric_classified_image, cmap='tab20', vmin=0, vmax=len(unique_labels) - 1)  # ajuste vmin e vmax conforme necessário
        plt.colorbar(label='Classified')
        plt.title('Classified NDVI Image')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()