import glob
from vision_classifier import VisionClassifier
from vision_classifier.encoders.hf_encoder import HuggingFaceEncoder
from vision_classifier.classifiers.knn import KNNClassifier
from vision_classifier.classifiers.nearest_centroid import NearestCentroidClassifier
from vision_classifier.classifiers.svm import SVMClassifier
from vision_classifier.storage.in_memory import InMemoryStorage

def test_classifiers():
    # Initial Setup
    print("--- Initializing and Training KNN Classifier ---")
    encoder = HuggingFaceEncoder("google/siglip2-so400m-patch14-224")
    storage = InMemoryStorage()
    knn_classifier = KNNClassifier(k=3)
    vision_classifier = VisionClassifier(encoder, knn_classifier, storage)

    cat_images = glob.glob("example/example_images/cats/*.jpg")
    dog_images = glob.glob("example/example_images/dogs/*.jpg")

    vision_classifier.add_examples("cat", cat_images)
    vision_classifier.add_examples("dog", dog_images)
    vision_classifier.train()

    prediction = vision_classifier.classify_image(cat_images[0])
    print(f"Prediction for a cat image with KNN: {prediction}")
    # assert prediction['prediction'] == 'cat'

    # Test Nearest Centroid Classifier
    print("\n" + "="*50 + "\n")
    print("--- Testing Nearest Centroid Classifier ---")
    nearest_centroid_classifier = NearestCentroidClassifier()
    vision_classifier.set_classifier(nearest_centroid_classifier)

    prediction = vision_classifier.classify_image(dog_images[0])
    print(f"Prediction for a dog image with Nearest Centroid: {prediction}")
    # assert prediction['prediction'] == 'dog'

    # Test SVM Classifier
    print("\n" + "="*50 + "\n")
    print("--- Testing SVM Classifier ---")
    svm_classifier = SVMClassifier()
    vision_classifier.set_classifier(svm_classifier)

    prediction = vision_classifier.classify_image(cat_images[1])
    print(f"Prediction for a cat image with SVM: {prediction}")
    # assert prediction['prediction'] == 'cat'

if __name__ == "__main__":
    test_classifiers()
