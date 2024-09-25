# Context-Aware Predictive Coding: A Representation Learning Framework for WiFi Sensing

This repository contains the PyTorch implementation of **Context-Aware Predictive Coding: A Representation Learning Framework for WiFi Sensing**. 

**Note:** The repository is still under development and the code will be available soon. Please contact me if you have any specific questions.

## Abstract

WiFi sensing is an emerging technology that utilizes wireless signals for various sensing applications. However, the reliance on supervised learning and the scarcity of labelled data and the incomprehensible channel state information (CSI) data pose significant challenges. These issues affect deep learning models’ performance and generalization across different environments. Consequently, self-supervised learning (SSL) is emerging as a promising strategy to extract meaningful data representations with minimal reliance on labelled samples. In this paper, we introduce a novel SSL framework called Context-Aware Predictive Coding (CAPC), which effectively learns from unlabelled data and adapts to diverse environments. CAPC integrates elements of Contrastive Predictive Coding (CPC) and the augmentation-based SSL method, Barlow Twins, promoting temporal and contextual consistency in data representations. This hybrid approach captures essential temporal information in CSI, crucial for tasks like human activity recognition (HAR), and ensures robustness against data distortions. Additionally, we propose a unique augmentation, employing both uplink and downlink CSI to isolate free space propagation effects and minimize the impact of electronic distortions of the transceiver. Our evaluations demonstrate that CAPC not only outperforms other SSL methods and supervised approaches, but also achieves superior generalization capabilities. Specifically, CAPC requires fewer labelled samples while significantly outperforming supervised learning by an average margin of 30.53% and surpassing SSL baselines by 6.5% on average in low-labelled data scenarios. Furthermore, our transfer learning studies on an unseen dataset with a different HAR task and environment showcase an accuracy improvement of 1.8% over other SSL baselines and 24.7% over supervised learning, emphasizing its exceptional cross-domain adaptability. These results mark a significant breakthrough in SSL applications for WiFi sensing, highlighting CAPC’s environmental adaptability and reduced dependency on labelled data.

![CAPC](./CAPC.svg)

## Requirements

To set up the required environment, follow these steps:

1. Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate CAPC
```

**Note:** I suggest use [mamba](https://github.com/mamba-org/mamba) instead of conda for faster environment creation.

## Datasets

1. SignFi Dataset

The expanded version of the dataset can be downloaded from [here](https://drive.google.com/file/d/1ywEbg6bJ5hLO9zeYbBfMlGvsMCz_BHcu/view?usp=sharing). This dataset is similar to the [original SignFi dataset]{https://yongsen.github.io/SignFi/}, but it is in npy format and works with our data loader.

2. UT HAR Dataset

Similar to SignFi, the expanded version of UT HAR can be downloaded from [here](https://drive.google.com/file/d/171IN6D6n81zkD_M0yClBDo3lNvSfAghS/view?usp=sharing).

---

For inquiries, please reach out via [email](mailto:bornab@yorku.ca).
