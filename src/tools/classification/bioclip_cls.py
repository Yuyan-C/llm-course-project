from bioclip import TreeOfLifeClassifier, Rank
from typing import Any, Dict, List

def run_bioclip(image_path: str) -> Dict[str, float]:
    """
    Run BioCLIP classification on the given image.
    Args:
    image_path: The path to the input image.
    Returns:
    dict: A dictionary containing the predicted species and their corresponding scores.
    """
   
    classifier = TreeOfLifeClassifier()
    predictions = classifier.predict(image_path, Rank.SPECIES)

    for prediction in predictions:
        print(prediction["species"], "-", prediction["score"])

    
    return predictions

if __name__ == "__main__":
    image_path = "/network/scratch/y/yuyan.chen/inquire/train/00261_Animalia_Arthropoda_Insecta_Coleoptera_Cerambycidae_Typocerus_velutinus/4e9c98d9-1fcd-41f3-a1e3-f206982e0210.jpg"
    run_bioclip(image_path)