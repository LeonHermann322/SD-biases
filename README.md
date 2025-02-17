# SD-biases
Ethics seminar project exploring religious biases in stable diffusion models.

# 1. Experiment pipeline
To test for religious biases we compare embeddings of images generated with religious adjectives versus many different identities which are specified by varying adjectives and professions. We can compare the generated images by computing embeddings of the images and clustering them. By analyzing the relations of embeddings from identities and religiously conditioned images we can identify religious biases in the stable diffusion models.

## 1.1 Image generation
Images are prompted by a combination of personas and settings. The chosen combinations can be found in [prompts.json](prompts.json).

In general the following categories are explored:

1. Settings: at home, at work
2. Personas defined by:
    1. Adjective (specifying religion or character)
    2. Gender identity or profession

Some examples for prompts conditioned by religion:

1. "A jewish man at work"
2. "A muslim non-binary person at work"
3. "A buddhist woman at home"

Some examples for prompts for identities:

1. "A confident painter at work"
2. "A sensitive physical therapist at home"
3. "A stubborn computer programmer at home"

## 1.2 Image embedding generation
We compute embeddings for all generated image with the image encoder of a SOTA Question-Answer-Model for images. This allows conditioning the created embeddings on a prompt. Thus, we can instruct the image encoder to focus on the person shown in the image instead of many details shown in the images that might be irrelevant to the person shown.

The Question-Answer-Model is prompted with:

**"What are the defining characteristics of the person shown in the image?"**

The computed embeddings should contain the information vital to describing those characteristics in the answer of the model.

## 1.3 Embedding evaluation
