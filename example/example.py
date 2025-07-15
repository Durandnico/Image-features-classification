import glob
from vision_classifier import VisionClassifier
from vision_classifier.encoders.hf_encoder import HuggingFaceEncoder
from vision_classifier.classifiers.knn import KNNClassifier
from vision_classifier.classifiers.nearest_neighbor import NearestNeighborClassifier
from vision_classifier.storage.in_memory import InMemoryStorage

def creating_new_classifier_huggingface():

    # Initialize components
    encoder = HuggingFaceEncoder("google/siglip2-so400m-patch14-224")
    storage = InMemoryStorage()
    classifier = KNNClassifier(k=3)
    classifier = NearestNeighborClassifier()  # You can switch between KNN and Nearest Neighbor

    # Create the main classifier instance
    store_images = False # Set to True if you want to store images in the storage
    vision_classifier = VisionClassifier(encoder, classifier, storage, store_images=store_images)

    # Add examples
    cat_images = glob.glob("example_images/cats/*.jpg")
    dog_images = glob.glob("example_images/dogs/*.jpg")
    
    print(f"Found {len(cat_images)} cat images and {len(dog_images)} dog images.")
    print(cat_images[:2])  # Print first two cat images for verification
    print(dog_images[:2])  # Print first two dog images for verification

    vision_classifier.add_examples("cat", [cat_images[0]])
    vision_classifier.add_examples("dog", dog_images)

    # Train the classifier
    vision_classifier.train()

    # Classify a new image
    new_image_path = "example_images/cats/cat2.jpg"  # Using one of the training images for demonstration
    prediction = vision_classifier.classify_image(new_image_path)
    print(f"Classification result for {new_image_path}: {prediction}")

    # Save the classifier state
    vision_classifier.save("siglip2.pkl")



def loading_pretrained_classifier_huggingface():

    # Save the classifier state
    new_image_path = "example_images/cats/cat2.jpg"  # Using one of the training images for demonstration

    # Load the classifier state
    new_classifier = VisionClassifier.load_from_pretrained("siglip2.pkl")

    # Classify with the loaded classifier
    prediction_after_load = new_classifier.classify_image(new_image_path)
    print(f"Classification result after loading for {new_image_path}: {prediction_after_load}")


if __name__ == "__main__":
    print("Creating a new classifier with HuggingFace encoder...")
    print("=" * 50)
    creating_new_classifier_huggingface()

    print("\n" + "=" * 50)
    print("\nLoading a pretrained classifier with HuggingFace encoder...")
    print("=" * 50)
    # Load a pretrained classifier
    loading_pretrained_classifier_huggingface()
