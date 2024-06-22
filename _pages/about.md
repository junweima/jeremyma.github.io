---
permalink: /
title: ""
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

\*\* denotes solo first author; \* denotes co-first author

# Recent Publications

**[\*Retrieval & Fine-Tuning for In-Context Tabular Models](https://arxiv.org/abs/2406.05207)**

  *Published in **NeurIPS** Workshop, 2023 (Runner-up Best Paper Award)*
  <details>
    <summary>Abstract</summary>

    <small>Tabular data is a pervasive modality spanning a wide range of domains, and the inherent diversity poses a considerable challenge for deep learning. Recent advancements using transformer-based in-context learning have shown promise on smaller and less complex datasets, but have struggled to scale to larger and more complex ones. To address this limitation, we propose a combination of retrieval and fine-tuning: we can adapt the transformer to a local subset of the data by collecting nearest neighbours, and then perform task-specific fine-tuning with this retrieved set of neighbours in context. Using TabPFN as the base model -- currently the best tabular in-context learner -- and applying our retrieval and fine-tuning scheme on top results in what we call a locally-calibrated PFN, or LoCalPFN. We conduct extensive evaluation on 95 datasets curated by TabZilla from OpenML, upon which we establish a new state-of-the-art with LoCalPFN -- even with respect to tuned tree-based models. Notably, we show a significant boost in performance compared to the base in-context model, demonstrating the efficacy of our approach and advancing the frontier of deep learning in tabular data.</small>

  </details>

---

**[\*\*In-Context Data Distillation with TabPFN](https://arxiv.org/abs/2402.06971)**

  *Published in **NeurIPS** Workshop, 2023 (Runner-up Best Paper Award)*
  <details>
    <summary>Abstract</summary>

    <small>Foundation models have revolutionized tasks in computer vision and natural language processing. However, in the realm of tabular data, tree-based models like XGBoost continue to dominate. TabPFN, a transformer model tailored for tabular data, mirrors recent foundation models in its exceptional in-context learning capability, being competitive with XGBoost's performance without the need for task-specific training or hyperparameter tuning. Despite its promise, TabPFN's applicability is hindered by its data size constraint, limiting its use in real-world scenarios. To address this, we present in-context data distillation (ICD), a novel methodology that effectively eliminates these constraints by optimizing TabPFN's context. ICD efficiently enables TabPFN to handle significantly larger datasets with a fixed memory budget, improving TabPFN's quadratic memory complexity but at the cost of a linear number of tuning steps. Notably, TabPFN, enhanced with ICD, demonstrates very strong performance against established tree-based models and modern deep learning methods on 48 large tabular datasets from OpenML.</small>

  </details>

---

**[\*\*TabPFGen–Tabular Data Generation with TabPFN](https://openreview.net/pdf?id=4MkkNsAEmO)**

  *Published in **NeurIPS** Workshop, 2023 (Runner-up Best Paper Award)*
  <details>
    <summary>Abstract</summary>

    <small>Advances in deep generative modelling have not translated well to tabular data. We argue that this is caused by a mismatch in structure between popular generative models and _discriminative_ models of tabular data. We thus devise a technique to turn TabPFN -- a highly performant transformer initially designed for in-context discriminative tabular tasks -- into an energy-based generative model, which we dub _TabPFGen_. This novel framework leverages the pre-trained TabPFN as part of the energy function and does not require any additional training or hyperparameter tuning, thus inheriting TabPFN's in-context learning capability. We can sample from TabPFGen analogously to other energy-based models. We demonstrate strong results on standard generative modelling tasks, including data augmentation, class-balancing, and imputation, unlocking a new frontier of tabular data generation.</small>

  </details>

---

**[\*X-pool: Cross-modal language-video attention for text-video retrieval](https://openaccess.thecvf.com/content/CVPR2022/papers/Gorti_X-Pool_Cross-Modal_Language-Video_Attention_for_Text-Video_Retrieval_CVPR_2022_paper.pdf)**  

   *Published in **CVPR**, 2022*

  <details>
    <summary>Abstract</summary>

    <small>In text-video retrieval, the objective is to learn a cross-modal similarity function between a text and a video that ranks relevant text-video pairs higher than irrelevant pairs. However, videos inherently express a much wider gamut of information than texts. Instead, texts often capture sub-regions of entire videos and are most semantically similar to certain frames within videos. Therefore, for a given text, a retrieval model should focus on the text's most semantically similar video sub-regions to make a more relevant comparison. Yet, most existing works aggregate entire videos without directly considering text. Common text-agnostic aggregations schemes include mean-pooling or self-attention over the frames, but these are likely to encode misleading visual information not described in the given text. To address this, we propose a cross-modal attention model called X-Pool that reasons between a text and the frames of a video. Our core mechanism is a scaled dot product attention for a text to attend to its most semantically similar frames. We then generate an aggregated video representation conditioned on the text's attention weights over the frames. We evaluate our method on three benchmark datasets of MSR-VTT, MSVD and LSMDC, achieving new state-of-the-art results by up to 12% in relative improvement in Recall@1. Our findings thereby highlight the importance of joint text-video reasoning to extract important visual cues according to text. Full code and demo can be found at: https://layer6ai-labs.github.io/xpool/</small>

  </details>

---

**[\*Weakly Supervised Action Selection Learning in Video](https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Weakly_Supervised_Action_Selection_Learning_in_Video_CVPR_2021_paper.pdf)**  
   
   *Published in **CVPR**, 2021*

  <details>
    <summary>Abstract</summary>

    <small>Localizing actions in video is a core task in computer vision. The weakly supervised temporal localization problem investigates whether this task can be adequately solved with only video-level labels, significantly reducing the amount of expensive and error-prone annotation that is required. A common approach is to train a frame-level classifier where frames with the highest class probability are selected to make a video-level prediction. Frame-level activations are then used for localization. However, the absence of frame-level annotations cause the classifier to impart class bias on every frame. To address this, we propose the Action Selection Learning (ASL) approach to capture the general concept of action, a property we refer to as "actionness". Under ASL, the model is trained with a novel class-agnostic task to predict which frames will be selected by the classifier. Empirically, we show that ASL outperforms leading baselines on two popular benchmarks THUMOS-14 and ActivityNet-1.2, with 10.3% and 5.7% relative improvement respectively. We further analyze the properties of ASL and demonstrate the importance of actionness. Full code for this work is available here https://github.com/layer6ai-labs/ASL</small>

  </details>

---

**[Guided similarity separation for image retrieval](https://proceedings.neurips.cc/paper/2019/file/7504adad8bb96320eb3afdd4df6e1f60-Paper.pdf)**  
   
   *Published in **NeurIPS**, 2019* (**oral**)

  <details>
    <summary>Abstract</summary>

    <small>Despite recent progress in computer vision, image retrieval remains a challenging open problem. Numerous variations such as view angle, lighting and occlusion make it difficult to design models that are both robust and efficient. Many leading methods traverse the nearest neighbor graph to exploit higher order neighbor information and uncover the highly complex underlying manifold. In this work we propose a different approach where we leverage graph convolutional networks to directly encode neighbor information into image descriptors. We further leverage ideas from clustering and manifold learning, and introduce an unsupervised loss based on pairwise separation of image similarities. Empirically, we demonstrate that our model is able to successfully learn a new descriptor space that significantly improves retrieval accuracy, while still allowing efficient inner product inference. Experiments on five public benchmarks show highly competitive performance with up to 24\% relative improvement in mAP over leading baselines. Full code for this work is available here: https://github. com/layer6ai-labs/GSS.</small>

  </details>

---

**[Cross-Class Relevance Learning for Temporal Concept Localization](https://arxiv.org/pdf/1911.08548)**  
   
   *Published in **ICCV** Workshop, 2019*

  <details>
    <summary>Abstract</summary>

    <small>We present a novel Cross-Class Relevance Learning approach for the task of temporal concept localization. Most localization architectures rely on feature extraction layers followed by a classification layer which outputs class probabilities for each segment. However, in many real-world applications classes can exhibit complex relationships that are difficult to model with this architecture. In contrast, we propose to incorporate target class and class-related features as input, and learn a pairwise binary model to predict general segment to class relevance. This facilitates learning of shared information between classes, and allows for arbitrary class-specific feature engineering. We apply this approach to the 3rd YouTube-8M Video Understanding Challenge together with other leading models, and achieve first place out of over 280 teams. In this paper we describe our approach and show some empirical results.</small>

  </details>

---

**[Semi-Supervised Exploration in Image Retrieval](https://arxiv.org/pdf/1906.04944)**  
   
   *Published in **CVPR** Workshop, 2019*

  <details>
    <summary>Abstract</summary>

    <small>We present our solution to Landmark Image Retrieval Challenge 2019. This challenge was based on the large Google Landmarks Dataset V2[9]. The goal was to retrieve all database images containing the same landmark for every provided query image. Our solution is a combination of global and local models to form an initial KNN graph. We then use a novel extension of the recently proposed graph traversal method EGT [1] referred to as semi-supervised EGT to refine the graph and retrieve better candidates.</small>

  </details>

---

**[\*Text-to-image-to-text translation using cycle consistent adversarial networks](https://arxiv.org/pdf/1808.04538.pdf)**  
   
   *Published in Arxiv, 2018*

  <details>
    <summary>Abstract</summary>

    <small>Text-to-Image translation has been an active area of research in the recent past. The ability for a network to learn the meaning of a sentence and generate an accurate image that depicts the sentence shows ability of the model to think more like humans. Popular methods on text to image translation make use of Generative Adversarial Networks (GANs) to generate high quality images based on text input, but the generated images don't always reflect the meaning of the sentence given to the model as input. We address this issue by using a captioning network to caption on generated images and exploit the distance between ground truth captions and generated captions to improve the network further. We show extensive comparisons between our method and existing methods.</small>

  </details>

---

# CV

<iframe src="cv.pdf" width="100%" height="770px"></iframe>