import json
from typing import Callable, Literal
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import torch
from scipy.cluster.hierarchy import linkage


from src.data_class import EmbeddingDataClass


def for_each_prompt(
    prompts_file_path: str,
    folder: str,
    setting: str,
    func: Callable[[str, str, str, str, list[str], int], None],
) -> None:
    with open(prompts_file_path, "r") as f:
        prompts = json.load(f)

    for key in prompts.keys():
        prompt_dict = prompts[key]
        prefixes = prompt_dict["prefix"]
        objects = prompt_dict["object"]
        images_per_prompt = prompt_dict["images_per_prompt"]

        for obj in objects:
            func(folder, setting, key, obj, prefixes, images_per_prompt)


def filter_data(
    data: list[EmbeddingDataClass],
    filters: dict[str, list[str]],
    logical_op: Literal["AND", "OR"] = "AND",
) -> list[EmbeddingDataClass]:
    filtered_data = []
    for d in data:
        if logical_op == "AND":
            if all(
                [d.__getattribute__(attr) in filters[attr] for attr in filters.keys()]
            ):
                filtered_data.append(d)
        else:
            if any(
                [d.__getattribute__(attr) in filters[attr] for attr in filters.keys()]
            ):
                filtered_data.append(d)

    return filtered_data


def visualize_keys_w_clusters(embeddings: list[EmbeddingDataClass], keys: list[str]):
    # Define a consistent color palette
    def get_color_map(unique_clusters):
        palette = sns.color_palette("tab10", len(unique_clusters))  # Generate colors
        return {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}

    # Get all unique clusters across the entire dataset
    all_clusters = sorted(set(embedding.cluster for embedding in embeddings))
    cluster_color_map = get_color_map(all_clusters)  # Fixed color mapping

    # Compute global min/max values for fixed scale
    all_embeddings = np.stack(
        [embedding.reduced_dim_embedding for embedding in embeddings]
    )
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()

    def visualize(key):
        out_folder = f"evaluation/clusters"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        filters = {"setting": [key], "object": [key], "prefix": [key]}
        filtered_data = filter_data(embeddings, filters, logical_op="OR")
        filtered_embeddings = np.stack(
            [embedding.reduced_dim_embedding for embedding in filtered_data]
        )
        filtered_clusters = [embedding.cluster for embedding in filtered_data]

        # Map clusters to consistent colors
        colors = [cluster_color_map[cluster] for cluster in filtered_clusters]

        plt.figure(figsize=(10, 10))
        plt.scatter(
            filtered_embeddings[:, 0],
            filtered_embeddings[:, 1],
            c=colors,
            edgecolors="k",
            alpha=0.7,
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(key)
        plt.savefig(f"{out_folder}/{key}.png")
        plt.close()

    for key in keys:
        visualize(key)


def get_all_keys():
    # All classes
    keys = ["work", "home"]
    with open("prompts.json", "r") as f:
        prompts = json.load(f)
        for key in prompts.keys():
            for sub_key in prompts[key]:
                if not isinstance(prompts[key][sub_key], list):
                    continue
                for sub_sub_key in prompts[key][sub_key]:
                    keys.append(sub_sub_key)
    return keys


def calculate_optimal_clusters(embeddings: list[EmbeddingDataClass]):
    embeddings_stacked = (
        torch.stack([embedding.embedding for embedding in embeddings]).squeeze().numpy()
    )
    linked = linkage(embeddings_stacked, method="ward")
    last_dists = linked[-10:, 2]
    diffs = np.diff(last_dists)
    optimal_clusters = np.argmax(diffs) + 1
    return optimal_clusters


def perform_clustering(embeddings: list[EmbeddingDataClass], num_clusters: int):
    embeddings_stacked = (
        torch.stack([embedding.embedding for embedding in embeddings]).squeeze().numpy()
    )
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage="ward")

    cluster_labels = clustering.fit_predict(embeddings_stacked)

    for i, embedding in enumerate(embeddings):
        embedding.cluster = cluster_labels[i]


def perform_dimension_reduction(embeddings: list[EmbeddingDataClass]):
    embeddings_stacked = (
        torch.stack([embedding.embedding for embedding in embeddings]).squeeze().numpy()
    )
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_embeddings = tsne.fit_transform(embeddings_stacked)

    for i, embedding in enumerate(embeddings):
        embedding.reduced_dim_embedding = tsne_embeddings[i]


def key_similarity(key1: str, key2: str, embeddings: list[EmbeddingDataClass]):
    key1_filters = {"setting": [key1], "object": [key1], "prefix": [key1]}
    key1_data = filter_data(embeddings, key1_filters, logical_op="OR")

    key2_filters = {"setting": [key2], "object": [key2], "prefix": [key2]}
    key2_data = filter_data(embeddings, key2_filters, logical_op="OR")

    # Calculate cosine similarity between all pairs of embeddings
    similarities = []
    for embedding1 in key1_data:
        for embedding2 in key2_data:
            emb1_tensor = embedding1.embedding.squeeze().numpy()
            emb2_tensor = embedding2.embedding.squeeze().numpy()
            similarity = np.dot(emb1_tensor, emb2_tensor) / (
                np.linalg.norm(emb1_tensor) * np.linalg.norm(emb2_tensor)
            )
            similarities.append(similarity)

    return np.mean(similarities)


def calculate_sim_matrix(keys: list[str], embeddings: list[EmbeddingDataClass]):
    similarity_matrix = np.ndarray((len(keys), len(keys)))
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            similarity_matrix[i, j] = key_similarity(key1, key2, embeddings)
    return similarity_matrix


def sim_matrix_to_file(sim_matrix: np.ndarray, keys: list[str]):
    # Write all similarities to file (csv)
    with open("evaluation/similarities.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([""] + keys)
        for i, row in enumerate(sim_matrix):
            writer.writerow([keys[i]] + row.tolist())


def visualize_similarity_w_keys(matrix: np.ndarray, name: str, keys: list[str]):
    plt.figure(figsize=(30, 30))  # Increase figure size
    sns.heatmap(matrix, cmap="magma", xticklabels=keys, yticklabels=keys)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(f"evaluation/{name}.png", bbox_inches="tight")
    plt.close()
