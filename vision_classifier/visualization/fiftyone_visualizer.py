import fiftyone as fo
import numpy as np
from vision_classifier.storage.base import Storage
import fiftyone.brain as fob


class FiftyOneVisualizer:
    def __init__(self, storage: Storage):
        self.storage = storage

    def visualize(
        self,
        dataset_name: str = "image_embeddings",
        launch_app: bool = True,
        method: str = "umap",
        brain_key: str = "img_viz",
        min_samples: int = 4,
    ):
        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
            dataset.delete()

        dataset = fo.Dataset(name=dataset_name, persistent=True)

        all_data = self.storage.get_all_data()
        class_embeddings = all_data["class_embeddings"]
        class_examples = all_data["class_examples"]

        for class_name, examples in class_examples.items():
            embeddings = class_embeddings[class_name]
            for example_path, embedding in zip(examples, embeddings):
                sample = fo.Sample(
                    filepath=example_path,
                    ground_truth=fo.Classification(label=class_name),
                )
                sample["embedding"] = np.squeeze(embedding)
                dataset.add_sample(sample)

        num_samples = len(dataset)
        if num_samples < min_samples:
            print(
                f"Dataset must have at least {min_samples} samples for visualization, but found {num_samples}."
            )
            return

        # Compute visualization
        print(f"Computing {method} visualization for dataset '{dataset_name}'...")
        print(f"Using brain key: {brain_key}")
        print("this is the dataset:")
        print(dataset)

        brain_config = {}
        if method == "umap":
            if num_samples < 15:
                brain_config["n_neighbors"] = max(1, num_samples - 1)
                brain_config["init"] = "pca"
        elif method == "tsne":
            if num_samples < 50:
                brain_config["pca_dim"] = max(1, num_samples - 1)

        results = fob.compute_visualization(
            dataset,
            embeddings="embedding",
            brain_key=brain_key,
            method=method,
            brain_config=brain_config,
        )

        print(f"Computed {method} visualization with brain key '{brain_key}'")

        if launch_app:
            session = fo.launch_app(dataset)
            print("Waiting for FiftyOne session to close...")
            session.wait()

        return dataset
