from bioclip import TreeOfLifeClassifier, Rank
from typing import Any, Dict, List
from PIL import Image

BIOCLIP_CLASSIFIER: TreeOfLifeClassifier | None = None


def _get_bioclip_classifier() -> TreeOfLifeClassifier:
    global BIOCLIP_CLASSIFIER
    if BIOCLIP_CLASSIFIER is None:
        BIOCLIP_CLASSIFIER = TreeOfLifeClassifier()
    return BIOCLIP_CLASSIFIER


def _normalize_bioclip_inputs(
    image_path: str | List[str] | None,
    image: Image.Image | List[Image.Image] | None,
) -> List[Any]:
    if image is None and not image_path:
        raise ValueError("run_bioclip requires image_path or image")
    if image is not None:
        return image if isinstance(image, list) else [image]
    return image_path if isinstance(image_path, list) else [image_path]

def run_bioclip(
    image_path: str | List[str] | None = None,
    image: Image.Image | List[Image.Image] | None = None,
) -> Dict[str, float] | List[Dict[str, float]]:
    """
    Run BioCLIP classification on the given image.
    Args:
        image_path: The path to the input image.
        image: A PIL Image to use instead of loading from image_path.
    Returns:
    dict: A dictionary containing the predicted species and their corresponding scores.
    """
    classifier = _get_bioclip_classifier()
    inputs = _normalize_bioclip_inputs(image_path, image)

    outputs: List[Dict[str, float]] = []
    predictions = classifier.predict(inputs, Rank.SPECIES)
    return predictions
    # for input_image in inputs:
    #     predictions = classifier.predict(input_image, Rank.SPECIES)
    #     results: Dict[str, float] = {}
    #     for prediction in predictions:
    #         species_name = prediction["species"]
    #         score = prediction["score"]
    #         results[species_name] = score
    #     outputs.append(results)

    # if len(outputs) == 1:
    #     return outputs[0]
    # return outputs

if __name__ == "__main__":
    image_path = "/network/scratch/y/yuyan.chen/inquire/train/00261_Animalia_Arthropoda_Insecta_Coleoptera_Cerambycidae_Typocerus_velutinus/4e9c98d9-1fcd-41f3-a1e3-f206982e0210.jpg"
    print(run_bioclip([image_path,image_path]))