# Transcription Factor Transformer Imputation (TFTI)
* Alexander Ku$^*$, Gunjan Bain$^*$, Ashish Vaswani
* Corresponding email: `alexku@berkeley.edu`, `gunjan_baid@berkeley.edu`
* And anyone else who wants to get involved!

## Background

Understanding the interplay between transcription factors (TFs) and DNA is an important step in better understanding differential gene expression. TFs are proteins that bind to specific sequences in the genome to regulate the rate at which DNA is transcribed into RNA and thereby translated into proteins. Recent advancements in both sequencing technologies and wet-lab assays (e.g., ChIP-seq) can accurately identify TF binding sites. However, these methods are time and cost-intensive, and produce a massive amount of raw data, thus motivating the need for an efficient computational approach.
While previous works have had considerable success with TF binding site prediction (DeepSEA, DeepBind, etc.), they have two limitations that we seek to address. First, they often aim to predict TF binding events indepentently. This is problematic, as it is well established that TFs interact to form TF complexes. Second, when partial binding information is present, there is no obvious way to inject it into their model to improve the prediction.


### Problem Setting
Suppose we have $N$ binding events to predict and we only know the ground truth for $K$ of them. We aim to develop an algorithm that can leverage the $K$ observed events to impute the remaining $N-K$.

## Architecture

The TFTI algorithm uses an augmented Transformer network architecture ([Vaswani *et al.*](http://papers.nips.cc/paper/7181-attention-is-all-you-need)) in a semi-generative manner. The key differences are the following:

1. The decoder input is a latent tensor with the same dimensionality as the prediction logits. This tensor will encode the $K$ observed binding events and, through successive passes through the netwtork, will be transformed into our actual logits (presumably with the $N-K$ unobserved binding events imputed). In the pure predictive setting, the latent code is a zero-tensor.
2. The decoder self-attention is no longer masked to be causal. We are interested in the mutual interaction between binding events.
3. No decoder positional embeddings.

<img src="images/transformer.png", height=400></img>

Above is the model architecture diagram of the Transformer. 