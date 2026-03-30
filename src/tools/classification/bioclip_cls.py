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

    results = {}
    for prediction in predictions:
        species_name = prediction["species"]
        score = prediction["score"]
        results[species_name] = score


    
    return results

if __name__ == "__main__":
    image_path = "/network/scratch/y/yuyan.chen/inquire/train/00261_Animalia_Arthropoda_Insecta_Coleoptera_Cerambycidae_Typocerus_velutinus/4e9c98d9-1fcd-41f3-a1e3-f206982e0210.jpg"
    print(run_bioclip(image_path))