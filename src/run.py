from datamanipulation import DataManipulation
from intelligence import Intelligence
import warnings
warnings.filterwarnings('ignore')

ARCHIV='../data/Archiv/'
ARCHIV2016 = '../data/Archiv2016/'

def main():
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
