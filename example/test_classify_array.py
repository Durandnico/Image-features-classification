import glob
import numpy as np
from PIL import Image
from vision_classifier import VisionClassifier
from vision_classifier.encoders.hf_encoder import HuggingFaceEncoder
from vision_classifier.classifiers.knn import KNNClassifier
from vision_classifier.storage.in_memory import InMemoryStorage

def test_classify_from_array():
    """
    An example demonstrating how to classify an image loaded as a NumPy array.
    """
    print("--- Testing Classification from NumPy Array ---")

    # 1. Initialize and train the classifier
    encoder = HuggingFaceEncoder("google/siglip2-so400m-patch14-224")
    storage = InMemoryStorage()
    classifier = KNNClassifier(k=3)
    vision_classifier = VisionClassifier(encoder, classifier, storage)

    cat_images = glob.glob("example/example_images/cats/*.jpg")
    dog_images = glob.glob("example/example_images/dogs/*.jpg")

    vision_classifier.add_examples("cat", cat_images)
    vision_classifier.add_examples("dog", dog_images)
    vision_classifier.train()

    # 2. Load an image into a NumPy array
    image_to_classify_path = "example/example_images/cats/cat1.jpg"
    print(f"Loading image {image_to_classify_path} into a NumPy array...")
    pil_image = Image.open(image_to_classify_path).convert("RGB")
    image_array = np.array(pil_image)

    # 3. Classify the image using the array
    prediction = vision_classifier.classify_array(image_array)

    # 4. Print the result
    print(f"Classification result for the image array: {prediction}")
    assert prediction['prediction'] == 'cat'
    print("Assertion passed: The classifier correctly identified the image from the array.")


if __name__ == "__main__":
    test_classify_from_array()
