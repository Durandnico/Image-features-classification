import fiftyone as fo
import numpy as np
from vision_classifier.storage.base import Storage
import fiftyone.brain as fob
from sklearn.manifold import TSNE


class FiftyOneVisualizer:
    def __init__(self, storage: Storage):
        self.storage = storage

    def get_dataset(self, dataset_name: str = "image_embeddings") -> fo.Dataset:
        """
        Retrieves or creates a FiftyOne dataset with image embeddings.

        Args:
            dataset_name (str): Name of the FiftyOne dataset.

        Returns:
            fo.Dataset: The FiftyOne dataset.
        """
        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
            dataset.delete()

        dataset = fo.Dataset(name=dataset_name, persistent=True)

        try:
            all_data = self.storage.get_all_data()
        except Exception as e:
            print(f"Error retrieving data from storage: {e}")
            return dataset

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

        return dataset


    def visualize(
        self,
        dataset_name: str = "image_embeddings",
        launch_app: bool = True,
        method: str = "umap",
        brain_key: str = "img_viz",
    ):
        MIN_SAMPLES = 4
        dataset = self.get_dataset(dataset_name)

        num_samples = len(dataset)
        if num_samples < MIN_SAMPLES:
            print(
                f"Dataset must have at least {MIN_SAMPLES} samples for visualization, but found {num_samples}."
            )
            return

  
        # Compute visualization
        if method not in ["umap", "tsne", "pca", "all"]:
            raise ValueError(f"Invalid method '{method}'. Choose from 'umap', 'tsne', 'pca', or 'all'.")

        if method == "all":
            results = self._tsne_visualization(dataset, brain_key=f"tsne_{brain_key}")

            for meth in ["umap", "pca"]:
                results = fob.compute_visualization(
                    dataset,
                    embeddings="embedding",
                    brain_key=f"{meth}_{brain_key}",
                    method=meth,
                    seed=42,
                )
                print(f"Computing {meth} visualization for dataset '{dataset_name}'...")

        elif method == "tnse":
            brain_key = f"{method}_{brain_key}"
            results = self._tsne_visualization(dataset, brain_key=brain_key)


        else:
            brain_key = f"{method}_{brain_key}"
            results = fob.compute_visualization(
                dataset,
                embeddings="embedding",
                brain_key=brain_key,
                method=method,
                seed=42,
            )
            print(f"Computing {method} visualization for dataset '{dataset_name}'...")



        if launch_app:
            session = fo.launch_app(dataset)
            print("Waiting for FiftyOne session to close...")
            session.wait()

        return dataset


    def _compute_tsne_embeddings(self, dataset: fo.Dataset) -> np.ndarray:
        embeddings = np.array([sample["embedding"] for sample in dataset])
        tsne = TSNE(n_components=2, random_state=42)

        return tsne.fit_transform(embeddings)


    def _tsne_visualization(self, dataset: fo.Dataset, brain_key: str = "tsne_viz"):
        tsne_embeddings = self._compute_tsne_embeddings(dataset)

        results = fob.compute_visualization(
            dataset,
            points=tsne_embeddings,
            brain_key=brain_key,
            method="manual",
            seed=42,
        )

        return results
