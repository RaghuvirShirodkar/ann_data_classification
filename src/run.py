from datamanipulation import DataManipulation
from intelligence import Intelligence
import warnings
warnings.filterwarnings('ignore')

ARCHIV='../data/Archiv/'
ARCHIV2016 = '../data/Archiv2016/'

def main():
    # Archiv datase
    ## Get data into python
    stoff = DataManipulation(ARCHIV+'Stoff.csv')
    fleisch = DataManipulation(ARCHIV+'Fleisch.csv')
    holz = DataManipulation(ARCHIV+'Holz.csv')
    leder = DataManipulation(ARCHIV+'Leder.csv')

    # Archiv 2016 dataset
    ## Get data into python
    material = DataManipulation(ARCHIV2016+'2016material.csv')
    material_fake = DataManipulation(ARCHIV2016+'2016material-fake.csv')
    skin = DataManipulation(ARCHIV2016+'2016skin.csv')
    referenz = DataManipulation(ARCHIV2016+'Referenz-Haut_6-Klassen.csv')

    ## Align datasets
    material.align_dataset(other=referenz, update_other=False)
    material_fake.align_dataset(other=referenz, update_other=False)
    skin.align_dataset(other=referenz, update_other=True)
    stoff.align_dataset(other=material, update_other=False)
    fleisch.align_dataset(other=material, update_other=False)
    holz.align_dataset(other=material, update_other=False)
    leder.align_dataset(other=material, update_other=False)

    ## Augment dataset
    stoff.augment_dataset(target=0)
    fleisch.augment_dataset(target=0)
    holz.augment_dataset(target=0)
    leder.augment_dataset(target=0)
    material.augment_dataset(target=0)
    material_fake.augment_dataset(target=0)
    skin.augment_dataset(target=1)
    referenz.augment_dataset(target=1)

    ## Extract features
    stoff_features = stoff.extract_features(no_of_features=6)
    fleisch_features = fleisch.extract_features(no_of_features=6)
    holz_features = holz.extract_features(no_of_features=6)
    leder_features = leder.extract_features(no_of_features=6)
    material_features = material.extract_features(no_of_features=6)
    material_fake_features = material_fake.extract_features(no_of_features=6)
    skin_features = skin.extract_features(no_of_features=6)
    referenz_features = referenz.extract_features(no_of_features=6)

    ## Cumulate dataset into non_skin and skin
    non_skin_dataset = DataManipulation.cumulative_dataset([stoff_features, \
                                                            fleisch_features, \
                                                            holz_features, \
                                                            leder_features, \
                                                            material_features, \
                                                            material_fake_features])

    skin_dataset = DataManipulation.cumulative_dataset([skin_features, \
                                                        referenz_features])
    
    # Manipulating data for usage
    material = DataManipulation(ARCHIV2016+'2016material.csv')
    material.augment_dataset(target=0)
    material_features = material.extract_features(no_of_features=15)

    material_fake = DataManipulation(ARCHIV2016+'2016material-fake.csv')
    material_fake.augment_dataset(target=0)
    material_fake_features = material_fake.extract_features(no_of_features=15)

    non_skin_dataset = DataManipulation.cumulative_dataset([material_features, \
                                                            material_fake_features])

    skin = DataManipulation(ARCHIV2016+'2016skin.csv')
    skin.augment_dataset(target=1)
    skin_dataset = skin.extract_features(no_of_features=15)

    # Training and validating anns
    intelligence_client = Intelligence(non_skin_dataset, skin_dataset, 50)
    intelligence_client.svm_(kernel='linear')
    intelligence_client.svm_(kernel='rbf')
    intelligence_client.mlp_()

if __name__ == '__main__':

    main()
