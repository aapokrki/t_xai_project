from ConceptExtractor import ConceptExtractor
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def forward_from_layer(model, activation_map, layer):
    """
    helper function for TCAV - continue the forward pass from specific layer with activation map as input
    model: pytorch vgg16 architecture
    activation_map: torch.Tensor
    layer: specific start layer from which the forward pass continues
    """
    model.eval()
    with torch.no_grad():
        remaining_features = model.features[layer:]
        # continue with feature extractor
        x = remaining_features(activation_map)
        # prepare for classifier according to models documentation:
        # https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/vgg.py
        x = torch.flatten(x, 1)
        # apply the classifier
        x = model.classifier(x)
        return x

def calc_concept_number(thresholds: list, A, init_concepts): # --> put in utils file
    """
    thresholds: list of thresholds for maximal similarity
    A: activation map
    init_concepts: initial number of allowed concepts --> will be reduced
    """
    # ideal number of concepts for each threshold; [nr concepts, P]
    concept_dict = {threshold: [None, None] for threshold in thresholds}

    for threshold in thresholds:
        # create concept extractor instance with specific activation layer
        ConceptExtractor_activations = ConceptExtractor(A, seed=1234)
        concepts = init_concepts
        similar_concepts = True
        while similar_concepts and concepts > 0:
            # train NMF
            ConceptExtractor_activations.apply_NMF(nr_concepts=concepts)
            # extract the P matrix where the rows correspond to the CAVs
            P = ConceptExtractor_activations.P
            # calculate the cosine similarity between the CAVs
            cos_sim = cosine_similarity(P)
            # extract the upper triangle, excluding the diagonal to get the similarities beteen CAVs
            cos_sim = np.triu(cos_sim, k=1)
            # check if all the CAVs are below the threshold
            above_threshold = cos_sim > threshold
            if above_threshold.sum() == 0:
                similar_concepts = False
                concept_dict[threshold][0] = concepts
                concept_dict[threshold][1] = P
            else:
                concepts -= 1
    return concept_dict

def TCAV_directional_derivative(activation_layer: np.array, layer_name: int, cav_vector: np.array, epsilon: float, model, class_k: int) -> np.array:
    """
    function calculates the directional derivatives which measure the sensitivity of predictions of the model of class k
    w.r.t. the cav_vector
    Note: we assume for function inputs that activation_layer and cav_vector are calculated for class k and not mixed and
    that all parameters that are influenced by layer_name use the same layer
    Note: this function is only suitable for one image-batch
    :param activation_layer: layer retrieved from the model, has activations for class k,
    shape 4dim: (batch_size, height, width, channels)
    :param layer_name: name of activation_layer
    :param cav_vector: holds the a concept, shape: 1dim (channels)
    :param epsilon: parameter for the directional derivative, should be small
    :param model: model used to calculate the predictions
    :param class_k: index of the cth class [0:10] as model outputs one of the 10 Cifar10 classes
    :return: directional_derivatives_C_l: the directional derviative for cav C from layer l as np.array of size: batch_size
    """
    model.eval()
    with torch.no_grad():
        input1 = activation_layer.clone() + epsilon*torch.tensor(cav_vector).view(1, cav_vector.shape[0], 1, 1)
        logit1 = forward_from_layer(model, input1, layer_name).detach().numpy()[:,class_k]
        logit2 = forward_from_layer(model, activation_layer.clone(), layer_name).detach().numpy()[:,class_k]
    return (logit1 - logit2) / epsilon

def TCAV_score(directional_derivatives_C_l: np.array) -> float:
    """
    function that calculates the fraction of images from class k where the concept C positively influenced the image
    being classified as k
    Note: we assume for all inputs that only class k was used to get actionvations, cavs,... otherwise the ouput
    is not meaningful
    Note: this function is only suitable for one image-batch
    :param directional_derivatives_C_l: I refer to information in TCAV_directional_derivative function
    :return: score, float between 0 and 1
    """
    return np.sum(directional_derivatives_C_l > 0) / directional_derivatives_C_l.shape[0]

def TCAV(activation_layer: np.array, layer_name: int, P: np.array, epsilon: float, model, class_k) -> np.array:
    """
    function that combines TCAV_directional_derivative and TCAV_score and returns TCAV scores for all cavs
    for parameter documentation, I refer to the function documentations
    Note: this function is only suitable for one image-batch
    :param P: is a matrix of shape (c', c) = (concept, channels) retrieved from NMF and holding the cav vectors
    :param class_k: index of the cth class [0:10] as model outputs one of the 10 classes
    :return: TCAV scores for all conecpts Ci for images of class k from layer l
    """
    scores = np.zeros(P.shape[0])
    for concept_idx in range(P.shape[0]):
        cav_vector = P[concept_idx].reshape(-1)
        derivatives = TCAV_directional_derivative(activation_layer, layer_name, cav_vector, epsilon, model, class_k)
        scores[concept_idx] = TCAV_score(derivatives)
    return scores

def plot_TCAV(activations, Concept_Extractor, classes, model, layer_name, epsilon=1e-6, save_path=None):
    # function to plot the TCAV scores
    fig, axes = plt.subplots(nrows=len(classes), ncols=1, figsize=(10, 20))
    # iterate through each class to calculate the concept presence for this specific class
    for i, ax in enumerate(axes):
        class_activations = activations[i]
        cav_scores = TCAV(class_activations.clone(),
                  layer_name = layer_name,
                  P = Concept_Extractor.P,
                  epsilon = epsilon,
                  model = model,
                  class_k = i)

        x_values = range(len(cav_scores))
        x_labels = [i+1 for i in range(len(cav_scores))]
        ax.bar(x_values, cav_scores, color='C0')
        ax.set_xticks(x_values, x_labels)
        ax.set_xlabel('Concepts')
        ax.set_ylabel('TCAV Score')
        ax.set_ylim(0, 1)
        ax.set_title(f'Class {classes[i]} ({class_activations.shape[0]} Samples)', fontsize=13)

    # ajust layout
    plt.tight_layout()
    plt.suptitle(f'TCAV Scores for {len(classes)} Concepts for Each Class', fontsize=20)
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path)

    plt.show()

def calc_avg_concept_presence(top_n, S_list, dataloader, ConceptExtractor): # --> move to utils
    # initialize lists to accumulate data across batches
    all_avg_concept_presence = []

    # iterate over all batches to collect data
    for batch_idx, (data, target) in enumerate(dataloader):
        S = S_list[batch_idx].copy()
        S = S.reshape((
            data.shape[0],
            ConceptExtractor.height,
            ConceptExtractor.width,
            ConceptExtractor.nr_concepts
        ))

        # calculate the average concept presence for the current batch
        avg_concept_presence = np.mean(S, axis=(1, 2))  # Shape (batch_size x nr_concepts)

        all_avg_concept_presence.append(avg_concept_presence)

    # convert the accumulated concept presence data to a single numpy array
    # print(len(all_avg_concept_presence))
    # print(all_avg_concept_presence[0].shape)
    all_avg_concept_presence = np.vstack(all_avg_concept_presence)  # Shape (total_samples x nr_concepts)
    # print("")
    # print(all_avg_concept_presence.shape)

    # dictionary to store the top n filenames and concept presence for each concept
    top_n_imgs = {}

    # calculate the top n indices globally for each concept
    for k in range(all_avg_concept_presence.shape[1]):  # Iterate over each concept (channel)
        sorted_indices_high = np.argsort(all_avg_concept_presence[:, k])[::-1]
        sorted_indices_low = np.argsort(all_avg_concept_presence[:, k])

        # get the top n indices with the highest and lowest values
        top_n_indices_high = sorted_indices_high[:top_n]
        top_n_indices_low = sorted_indices_low[:top_n]

        # store the top n filenames and their corresponding avg concept presence for both high and low
        top_n_imgs[f"concept_{k}"] = {
            'top_n_high': {
                'avg_concept_presence': [all_avg_concept_presence[idx, k] for idx in top_n_indices_high],
                'indices': top_n_indices_high
            },
            'top_n_low': {
                'avg_concept_presence': [all_avg_concept_presence[idx, k] for idx in top_n_indices_low],
                'indices': top_n_indices_low
            }
        }

    return top_n_imgs

def plot_concept_images(concept_dict, dataloader, S=None, threshold=0.5, background_alpha=0.6, plot_lowest=True, save_path=None):
    """
    Plots images for each concept showing the top 5 highest and lowest concept presence.

    concept_dict: Dictionary containing concept presence and indices.
    dataloader: PyTorch dataloader containing the images.
    S: concept strengths for all images
    threshold: threshold for concept strengths
    background_alpha: alpha of image regions without concept presence
    """
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # values for unnormalizing image
    mean=torch.tensor([0.4914, 0.4822, 0.4465])
    std=torch.tensor([0.2023, 0.1994, 0.201])

    num_concepts = len(concept_dict)
    rows_per_concept = 2 if plot_lowest else 1
    size = (12, 60) if plot_lowest else (12, 30)
    total_rows = num_concepts * rows_per_concept
    cols = 5

    fig, axes = plt.subplots(total_rows, cols, figsize=size)

    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for concept_idx, (concept, data) in enumerate(concept_dict.items()):
        # extract high and low indices
        high_indices = data['top_n_high']['indices']
        low_indices = data['top_n_low']['indices']

        # fetch corresponding images from dataloader
        high_images = [dataloader.dataset[idx][0] for idx in high_indices]
        high_labels = [dataloader.dataset[idx][1] for idx in high_indices]
        low_images = [dataloader.dataset[idx][0] for idx in low_indices]
        low_labels = [dataloader.dataset[idx][1] for idx in low_indices]

        if S is not None:
            high_concepts = [S[idx][concept_idx] for idx in high_indices]
            low_concepts = [S[idx][concept_idx] for idx in low_indices]

        row_offset = concept_idx * rows_per_concept

        # plot top 5 high concept presence images
        axes[row_offset, 2].annotate(f"\nConcept {concept_idx + 1}",
                          xy=(0.5, 1.15), xycoords='axes fraction',
                          ha='center', va='bottom', fontsize=14)

        for i, (img, label) in enumerate(zip(high_images, high_labels)):
            ax = axes[row_offset, i]
            # prepare image
            img = img * std[:, None, None] + mean[:, None, None]
            img = img.permute(1, 2, 0)
            # ensure image is between 0 and 1
            img = torch.clamp(img, 0, 1)

            if S is not None:
                concept = high_concepts[i].copy()
                # ensure concept is between 0 and 1
                concept = np.clip(concept, 0, 1)
                # create concept mask where concept is present in image
                mask = concept > threshold
                highlighted_regions = np.zeros_like(concept)
                highlighted_regions[mask] = concept[mask]
                cmap = plt.cm.viridis
                heatmap = cmap(highlighted_regions)
                # make the heatmap of the concept to RGB
                heatmap_rgb = heatmap[..., :3]
                blended_image = img.numpy().copy()
                # change image alpha
                blended_image *= background_alpha
                # replace image pasts of mask with concept
                blended_image[mask] = heatmap_rgb[mask]
                ax.imshow(blended_image)
            else:
                ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"High {i + 1} - {classes[label]}")

        if plot_lowest:
            # plot top 5 low concept presence images
            for i, (img, label) in enumerate(zip(low_images, low_labels)):
                ax = axes[row_offset+1, i]
                img = img * std[:, None, None] + mean[:, None, None]
                img = img.permute(1, 2, 0)
                img = torch.clamp(img, 0, 1)
                ax.imshow(img)  # convert from CHW to HWC
                if S is not None:
                    concept = low_concepts[i].copy()
                    concept = np.clip(concept, 0, 1)
                    mask = concept > threshold
                    highlighted_regions = np.zeros_like(concept)
                    highlighted_regions[mask] = concept[mask]
                    cmap = plt.cm.viridis
                    heatmap = cmap(highlighted_regions)
                    blended_image = img.numpy().copy()
                    blended_image *= background_alpha
                    blended_image[mask] = heatmap_rgb[mask]
                    ax.imshow(blended_image)
                else:
                    ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Low {i + 1} - {classes[label]}")

    fig.suptitle("Prototypical Images for Concepts\n", fontsize=20, y=0.99)
    fig.subplots_adjust(top=0.95, wspace=0.1)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()