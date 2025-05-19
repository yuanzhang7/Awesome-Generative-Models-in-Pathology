# Awesome-Generative-Models-in-Pathology
**[Generative Models in Computational Pathology: A Comprehensive Survey on Methods, Applications, and Challenges](https://arxiv.org/pdf/2505.10993)**[ [arXiv]](https://arxiv.org/abs/2505.10993) 

> *Yuan Zhang<sup>1</sup>, Xinfeng Zhang<sup>2</sup>, Xiaoming Qi<sup>3</sup>, Xinyu Wu<sup>1</sup>, Feng Chen<sup>4</sup>, Guangyu Yang<sup>1</sup>, Huazhu Fu<sup>5</sup>*

> *<sup>1</sup>Southeast University, <sup>2</sup>Tsinghua University, <sup>3</sup>National University of Singapore, <sup>4</sup>Nanjing Medical University, <sup>5</sup>Agency for Science, Technology and Research, Singapore.*
## ⚡News: 
### 16/05/2025: Our survey has been accessible on Arxiv([arxiv](https://arxiv.org/abs/2505.10993).)
```
@misc{zhang2025generativemodelscomputationalpathology,
      title={Generative Models in Computational Pathology: A Comprehensive Survey on Methods, Applications, and Challenges}, 
      author={Yuan Zhang and Xinfeng Zhang and Xiaoming Qi Xinyu Wu and Feng Chen and Guanyu Yang and Huazhu Fu},
      year={2025},
      eprint={2505.10993},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2505.10993}, 
}
```

## 📌 What is This Survey About?
🧬 Generative Models in Computational Pathology
Generative modeling is transforming the landscape of computational pathology, enabling data-efficient learning, synthetic data augmentation, and multimodal representation across a wide range of diagnostic tasks. This repository presents a comprehensive and structured survey of over 150 representative studies in the field, covering:

🖼️ Image Generation

📝 Text Generation

🔀 Multimodal Image–Text Generation

🧭 Other Applications (e.g., spatial simulation, molecular inference)

We trace the evolution of generative architectures, from early GANs to state-of-the-art diffusion models and foundation models, and provide commonly used datasets and evaluation metrics in generative pathology.In addition, we discuss open challenges and future directions. 

This project aims to serve as a foundational reference and open knowledge base for researchers, engineers, and clinicians working at the intersection of AI and pathology.


## 📖 Table of Content

- [📄 Abstract](#-abstract)
- [📘 Introduction](#-introduction)
- [🧠 Overview of Generative Pathology](#-overview-of-generative-pathology)
  - [Development Timeline](#development-timeline)
  - [Main Generative Methods](#main-generative-methods)
    - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
    - [Diffusion Models](#diffusion-models)
    - [Large Language / Vision-Language Models (LLMs/VLMs)](#large-language--vision-language-models-llmsvlms)
    - [Other Generative Models](#other-generative-models)

- [🧬 Generation Tasks](#-generation-tasks)
  - [Image Generation](#image-generation)
    - [Synthetic Image Generation](#synthetic-image-generation)
    - [Mask-Guided Generation](#mask-guided-generation)
    - [Multi-/High-Resolution Generation](#multihigh-resolution-generation)
    - [Text-to-Image Generation](#text-to-image-generation)
    - [Stain Generation](#stain-generation)
  - [Text Generation](#text-generation)
    - [Image Captioning and Description](#image-captioning-and-description)
    - [Visual Question Answering (VQA)](#visual-question-answering-vqa)
    - [Pathology Report Generation](#pathology-report-generation)
    - [Text-to-Text Generation](#text-to-text-generation)
  - [Multimodal Generation](#multimodal-generation)
  - [Other Generation](#other-generation)


- [📊 Datasets and Evaluation Metrics](#-datasets-and-evaluation-metrics)
- [🚧 Open Challenges and Future Directions](#-open-challenges-and-future-directions)


## 🤖 Generative Pathology Paperlist
### Image Generation
- Neural stain-style transfer learning using GAN for histopathological images, <ins>ACML, 2017</ins> [[Paper](https://arxiv.org/abs/1710.08543)]
- Stain normalization of histopathology images using generative adversarial networks, <ins>ISBI, 2018</ins> [[Paper](https://ieeexplore.ieee.org/abstract/document/8363641)]
- Selective synthetic augmentation with HistoGAN for improved histopathology image classification, <ins>MIA, 2020</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841520301808)]
- Seamless virtual whole slide image synthesis and validation using perceptual embedding consistency, <ins>JBHI, 2021</ins> [[Paper](https://ieeexplore.ieee.org/document/9003176)]
- Residual cyclegan for robust domain transformation of histopathological tissue slides, <ins>MIA, 2021</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841521000505)]
- PathologyGAN: Learning deep representations of cancer tissue, <ins>Journal of Machine Learning for Biomedical Imaging, 2021</ins> [[Paper](https://arxiv.org/abs/1907.02644)]
- Unpaired Stain Transfer Using Pathology-Consistent Constrained Generative Adversarial Networks, <ins>IEEE TMI, 2021</ins> [[Paper](https://ieeexplore.ieee.org/document/9389763)]
- Normalization of HE-stained histological images using cycle consistent generative adversarial networks, <ins>Diagnostic Pathology, 2021</ins> [[Paper](https://diagnosticpathology.biomedcentral.com/articles/10.1186/s13000-021-01126-y)]
- Self-Supervised Representation Learning using Visual Field Expansion on Digital Pathology, <ins>ICCV, 2021</ins> [[Paper](https://arxiv.org/abs/2109.03299)]
- Sharp-GAN: Sharpness Loss Regularized GAN for Histopathology Image Synthesis, <ins>ISBI, 2021</ins> [[Paper](https://ieeexplore.ieee.org/abstract/document/9761534)]
- A multi-attribute controllable generative model for histopathology image synthesis, <ins>MICCAI 2021, 2021</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_59)]
- Inference of captions from histopathological patches, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2202.03432)]
- Multi-scale self-attention generative adversarial network for pathology image restoration, <ins>The Visual Computer, 2022</ins> [[Paper](https://link.springer.com/article/10.1007/s00371-022-02592-1)]
- InsMix: Towards Realistic Generative Data Augmentation for Nuclei Instance Segmentation, <ins>MICCAI, 2022</ins> [[Paper](https://arxiv.org/abs/2206.15134)]
- Synthesis of diagnostic quality cancer pathology images, <ins>The Journal of Pathology, 2022</ins> [[Paper](https://www.biorxiv.org/content/early/2020/02/26/2020.02.24.963553)]
- Optimising diffusion models for histopathology image synthesis, <ins>BMVC, 2024</ins> [[Paper](https://pure.qub.ac.uk/en/publications/optimising-diffusion-models-for-histopathology-image-synthesis)]
- A Morphology Focused Diffusion Probabilistic Model for Synthesis of Histopathology Images, <ins>WACV, 2022</ins> [[Paper](https://openaccess.thecvf.com/content/WACV2023/html/Moghadam_A_Morphology_Focused_Diffusion_Probabilistic_Model_for_Synthesis_of_Histopathology_WACV_2023_paper.html)]
- Tackling stain variability using CycleGAN-based stain augmentation, <ins>Journal of Pathology Informatics, 2022</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S2153353922007349)]
- Stain normalization using score-based diffusion model through stain separation and overlapped moving window patch strategies, <ins>Computers in Biology and Medicine, 2022</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S0010482522010435)]
- Enhancing gland segmentation in colon histology images using an instance-aware diffusion model, <ins>Computers in Biology and Medicine, 2023</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S0010482523009927)]
- ViT-DAE: Transformer-driven Diffusion Autoencoder for Histopathology Image Analysis, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.01053)]
- Enhanced Pathology Image Quality with Restore-Generative Adversarial Network, <ins>The American Journal of Pathology, 2023</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S0002944023000275)]
- A Large-scale Synthetic Pathological Dataset for Deep Learning-enabled Segmentation of Breast Cancer, <ins>nature Scientific Data, 2023</ins> [[Paper](https://www.nature.com/articles/s41597-023-02125-y)]
- Diffmix: Diffusion model-based data synthesis for nuclei segmentation and classification in imbalanced pathology image datasets, <ins>MICCAI, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14132)]
- ProGleason-GAN: Conditional progressive growing GAN for prostatic cancer Gleason grade patch synthesis, <ins>Computer Methods and Programs in Biomedicine, 2023</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S0169260723003607)]
- Artifact Restoration in Histology Images with Diffusion Probabilistic Models, <ins>MICCAI, 2023</ins> [[Paper](https://arxiv.org/abs/2307.14262)]
- A federated learning system for histopathology image analysis with an orchestral stain-normalization GAN, <ins>JBHI, 2023</ins> [[Paper](https://ieeexplore.ieee.org/document/9947057)]
- Realistic Data Enrichment for Robust Image Segmentation in Histopathology, <ins>MICCAI Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2304.09534)]
- PathLDM: Text conditioned Latent Diffusion Model for Histopathology, <ins>WACV, 2023</ins> [[Paper](https://arxiv.org/abs/2309.00748)]
- NASDM: Nuclei-Aware Semantic Histopathology Image Generation Using Diffusion Models, <ins>MICCAI, 2023</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_76)]
- DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2306.13384)]
- StainDiff: Transfer Stain Styles of Histology Images with Denoising Diffusion Probabilistic Models and Self-ensemble, <ins>MICCAI, 2023</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_53)]
- Characterizing the Features of Mitotic Figures Using a Conditional Diffusion Probabilistic Model, <ins>MICCAI, 2023</ins> [[Paper](https://arxiv.org/abs/2310.03893)]
- Diffusion-based generation of histopathological whole slide images at a gigapixel scale, <ins>WACV, 2023</ins> [[Paper](https://arxiv.org/abs/2311.08199)]
- Diffusion-based Data Augmentation for Nuclei Image Segmentation, <ins>MICCAI, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_57)]
- Diffusion models for out-of-distribution detection in digital pathology, <ins>MIA, 2024</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841524000136)]
- Generative models improve fairness of medical classifiers under distribution shifts, <ins>Nature Medicine, 2024</ins> [[Paper](https://www.nature.com/articles/s41591-024-02838-6)]
- Generation of synthetic whole-slide image tiles of tumours from RNA-sequencing data via cascaded diffusion models, <ins>Nature Biomedical Engineering, 2024</ins> [[Paper](https://www.nature.com/articles/s41551-024-01193-8)]
- Diffusion Models for Generative Histopathology, <ins>MICCAI, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-53767-7_15)]
- Learned representation-guided diffusion models for large-image generation, <ins>CVPR, 2024</ins> [[Paper](https://arxiv.org/abs/2312.07330)]
- Generative adversarial networks for stain normalisation in histopathology, <ins>Applications of Generative AI, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-46238-2_11)]
- Generating progressive images from pathological transitions via diffusion model, <ins>MICCAI, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_29)]
- Diffusion and Multi-Domain Adaptation Methods for Eosinophil Segmentation, <ins>ICMVA, 2024</ins> [[Paper](https://arxiv.org/abs/2403.11323)]
- StainDiffuser: MultiTask Dual Diffusion Model for Virtual Staining, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.11340)]
- F2FLDM: Latent Diffusion Models with Histopathology Pre-Trained Embeddings for Unpaired Frozen Section to FFPE Translation, <ins>WACV, 2024</ins> [[Paper](https://ieeexplore.ieee.org/abstract/document/10943340)]
- AV-GAN: Attention-Based Varifocal Generative Adversarial Network for Uneven Medical Image Translation, <ins>arXiv, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10649902)]
- DISC: Latent Diffusion Models with Self-Distillation from Separated Conditions for Prostate Cancer Grading, <ins>ISBI, 2024</ins> [[Paper](https://arxiv.org/abs/2404.13097)]
- STAR-RL: Spatial–Temporal Hierarchical Reinforcement Learning for Interpretable Pathology Image Super-Resolution, <ins>IEEE TMI, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10574839)]
- VIMs: Virtual Immunohistochemistry Multiplex Staining via Text-to-Stain Diffusion Trained on Uniplex Stains, <ins>Machine Learning in Medical Imaging (MLMI 2024), 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73284-3_15)]
- Unsupervised Latent Stain Adaptation for Computational Pathology, <ins>MICCAI, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_70)]
- PathUp: Patch-wise Timestep Tracking for Multi-class Large Pathology Image Synthesising Diffusion Model, <ins>ACM Multimedia Conference (MM '24), 2024</ins> [[Paper](https://openreview.net/forum?id=A7VkIoEELI)]
- Histology Image Artifact Restoration with Lightweight Transformer Based Diffusion Model, <ins>MIDL, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-66535-6_9)]
- Accelerating histopathology workflows with generative AI-based virtually multiplexed tumour profiling, <ins>Nature Machine Intelligence, 2024</ins> [[Paper](https://www.nature.com/articles/s42256-024-00889-5)]
- StainFuser: Controlling Diffusion for Faster Neural Style Transfer in Multi-Gigapixel Histology Images, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.09302)]
- URCDM: Ultra-Resolution Image Synthesis in Histopathology, <ins>MICCAI, 2024</ins> [[Paper](https://arxiv.org/abs/2407.13277)]
- Histo-Diffusion: A Diffusion Super-Resolution Method for Digital Pathology with Comprehensive Quality Assessment, <ins>AIME, 2024</ins> [[Paper](https://arxiv.org/abs/2408.15218)]
- Test-Time Stain Adaptation with Diffusion Models for Histopathology Image Classification, <ins>ECCV, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72761-0_15)]
- Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model, <ins>ECCV, 2024</ins> [[Paper](https://arxiv.org/abs/2407.14434)]
- USegMix: Unsupervised Segment Mix for Efficient Data Augmentation in Pathology Images, <ins>DEMI, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73748-0_6)]
- Unified Framework for Histopathology Image Augmentation and Classification via Generative Models, <ins>DICTA, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10869588)]
- PST-Diff: achieving high-consistency stain transfer by diffusion models with pathological and structural constraints, <ins>IEEE TMI, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10601703)]
- Multi-modal Denoising Diffusion Pre-training for Whole-Slide Image Classification, <ins>32nd ACM International Conference on Multimedia (MM '24), 2024</ins> [[Paper](https://dl.acm.org/doi/10.1145/3664647.3680882)]
- Generating Seamless Virtual Immunohistochemical Whole Slide Images with Content and Color Consistency, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2410.01072)]
- Counterfactual Diffusion Models for Mechanistic Explainability of Artificial Intelligence Models in Pathology, <ins>bioRxiv, 2024</ins> [[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11565818/)]
- LatentArtiFusion: An Effective and Efficient Histological Artifacts Restoration Framework, <ins>MICCAI 2024, 2024</ins> [[Paper](https://arxiv.org/abs/2407.20172)]
- Virtual multi-staining in a single-section view for renal pathology using generative adversarial networks, <ins>Computers in Biology and Medicine, 2024</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S0950705123005300)]
- Style-Extracting Diffusion Models for Semi-supervised Histopathology Segmentation, <ins>ECCV, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73226-3_14#citeas)]
- HADiff: hierarchy aggregated diffusion model for pathology image segmentation, <ins>The Visual Computer, 2024</ins> [[Paper](https://doi.org/10.1007/s00371-024-03746-z)]
- Generating and evaluating synthetic data in digital pathology through diffusion models, <ins>nature Scientific Reports, 2024</ins> [[Paper](https://www.nature.com/articles/s41598-024-79602-w)]
- Comparative Analysis of Diffusion Generative Models in Computational Pathology, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2411.15719)]
- Versatile Stain Transfer in Histopathology Using a Unified Diffusion Framework, <ins>bioRxiv, 2024</ins> [[Paper](https://www.biorxiv.org/content/early/2024/11/23/2024.11.23.624680.full.pdf)]
- Unpaired Multi-Domain Histopathology Virtual Staining using Dual Path Prompted Inversion, <ins>AAAI, 2024</ins> [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32949)]
- Mask-guided cross-image attention for zero-shot in-silico histopathologic image generation with a diffusion model, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2407.11664)]
- A Value Mapping Virtual Staining Framework for Large-scale Histological Imaging, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2501.03592)]
- A robust image segmentation and synthesis pipeline for histopathology, <ins>MIA, 2025</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S136184152400269X)]
- GenSelfDiff-HIS: Generative Self-Supervision Using Diffusion for Histopathological Image Segmentation, <ins>IEEE TMI, 2025</ins> [[Paper](https://ieeexplore.ieee.org/abstract/document/10663482)]
- Deeply supervised two stage generative adversarial network for stain normalization, <ins>nature Scientific Reports, 2025</ins> [[Paper](https://www.nature.com/articles/s41598-025-91587-8)]
- Enhancing Image Retrieval Performance with Generative Models in Siamese Networks, <ins> JBHI, 2025</ins> [[Paper](https://ieeexplore.ieee.org/document/10896802)]
- ToPoFM: Topology-Guided Pathology Foundation Model for High-Resolution Pathology Image Synthesis with Cellular-Level Control, <ins>IEEE TMI, 2025</ins> [[Paper](https://ieeexplore.ieee.org/document/10915718)]
- PathoPainter: Augmenting Histopathology Segmentation via Tumor-aware Inpainting, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2503.04634)]
- Diffusion-based Virtual Staining from Polarimetric Mueller Matrix Imaging, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2503.01352)]
- DAMM-Diffusion: Learning Divergence-Aware Multi-Modal Diffusion Model for Nanoparticles Distribution Prediction, <ins>CVPR, 2025</ins> [[Paper](https://arxiv.org/abs/2503.09491)]
- PDSeg: Patch-Wise Distillation and Controllable Image Generation for Weakly-Supervised Histopathology Tissue Segmentation, <ins>ICASSP, 2025</ins> [[Paper](https://ieeexplore.ieee.org/document/10888097)]

### Text Generation
- PathVQA: 30000+ Questions for Medical Visual Question Answering, <ins>arXiv, 2020</ins> [[Paper](https://arxiv.org/abs/2003.10286)]
- Multiple Instance Captioning: Learning Representations from Histopathology Textbooks and Articles, <ins>CVPR, 2021</ins> [[Paper](https://ieeexplore.ieee.org/document/9577950)]
- Vision-Language Transformer for Interpretable Pathology Visual Question Answering, <ins>JBHI, 2023</ins> [[Paper](https://ieeexplore.ieee.org/document/9745795)]
- LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day, <ins>NeurIPS, 2023</ins> [[Paper](https://openreview.net/forum?id=GSuP99u2kR)]
- What a Whole Slide Image Can Tell? Subtype-guided Masked Transformer for Pathological Image Captioning, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.20607)]
- Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos, <ins>CVPR, 2023</ins> [[Paper](https://arxiv.org/abs/2312.04746)]
- Automatic Report Generation for Histopathology Images Using Pre-Trained Vision Transformers and BERT, <ins>ISBI, 2023</ins> [[Paper](https://arxiv.org/abs/2312.01435)]
- PathAsst: A Generative Foundation AI Assistant towards Artificial General Intelligence of Pathology, <ins>AAAI, 2024</ins> [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28308)]
- PathMMU: A Massive Multimodal Expert-Level Benchmark for Understanding and Reasoning in Pathology, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.16355)]
- Generating clinical-grade pathology reports from gigapixel whole slide images with HistoGPT, <ins>bioRxiv, 2024</ins> [[Paper](https://www.medrxiv.org/content/10.1101/2024.03.15.24304211v2)]
- HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction, <ins>MIA, 2024</ins> [[Paper](https://arxiv.org/abs/2403.05396)]
- Improving Mitosis Detection on Histopathology Images Using Large Vision-Language Models, <ins>ISBI, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10635613)]
- WsiCaption: Multiple Instance Generation of Pathology Reports for Gigapixel Whole-Slide Images, <ins>arXiv, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_51)]
- PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.00203)]
- PathAlign: A vision-language model for whole slide images in histopathology, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.19578)]
- CPLIP: Zero-Shot Learning for Histopathology with Comprehensive Vision-Language Alignment, <ins>CVPR, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10655627)]
- A multimodal generative AI copilot for human pathology, <ins>Nature, 2024</ins> [[Paper](https://www.nature.com/articles/s41586-024-07618-3)]
- PathM3: A Multimodal Multi-Task Multiple Instance Learning Framework for Whole Slide Image Classification and Captioning, <ins>CVPR, 2024</ins> [[Paper](https://arxiv.org/abs/2403.08967)]
- Cost-effective Instruction Learning for Pathology Vision and Language Analysis, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.17734)]
- PathInsight: Instruction Tuning of Multimodal Datasets and Models for Intelligence Assisted Diagnosis in Histopathology, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2408.07037)]
- PA-LLaVA: A Large Language-Vision Assistant for Human Pathology Image Understanding, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2408.09530)]
- In-context learning enables multimodal large language models to classify cancer pathology images, <ins>MIDL, 2024</ins> [[Paper](https://doi.org/10.1038/s41467-024-51465-9)]
- WSI-VQA: Interpreting Whole Slide Images by Generative Visual Question Answering, <ins>ECCV, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72764-1_23)]
- SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2410.11761)]
- PolyPath: Adapting a Large Multimodal Model for Multi-slide Pathology Report Generation, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2502.10536)]
- Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2502.19293)]
- Pathology report generation from whole slide images with knowledge retrieval and multi-level regional feature selection, <ins>Computer Methods and Programs in Biomedicine, 2025</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S016926072500094X)]
- PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2502.08916)]
- Pathologyvlm: a large vision-language model for pathology image understanding, <ins>Artificial Intelligence Review, 2025</ins> [[Paper](https://doi.org/10.1007/s10462-025-11190-1)]
- Cancer Type, Stage and Prognosis Assessment from Pathology Reports using LLMs, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2503.01194)]

### Multimodal Generation
- Towards Generalist Biomedical AI, <ins>NEJM AI, 2023</ins> [[Paper](https://ai.nejm.org/doi/abs/10.1056/AIoa2300138)]
- A visual-language foundation model for computational pathology, <ins>Nature Medicine, 2024</ins> [[Paper](https://www.nature.com/articles/s41591-024-02856-4#citeas)]
- PRISM: A MULTI-MODAL GENERATIVE FOUNDATION MODEL FOR SLIDE-LEVEL HISTOPATHOLOGY, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.10254)]
- Large-vocabulary forensic pathological analyses via prototypical cross-modal contrastive learning, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.14904)]
- Towards a generalizable pathology foundation model via unified knowledge distillation, <ins>Nature Biomedical Engineering, 2024</ins> [[Paper](https://arxiv.org/abs/2407.18449)]
- PathoDuet: Foundation models for pathological slide analysis of H&E and IHC stains, <ins>MIA, 2024</ins> [[Paper](https://www.sciencedirect.com/science/article/pii/S1361841524002147)]
- A Multimodal Knowledge-enhanced Whole-slide Pathology Foundation Model, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.15362)]
- A vision–language foundation model for precision oncology, <ins>Nature, 2024</ins> [[Paper](https://doi.org/10.1038/s41586-024-08378-w)]
- Quilt-1M: One Million Image-Text Pairs for Histopathology, <ins>NeurIPS, 2025</ins> [[Paper](https://arxiv.org/abs/2306.11207)]
- PathOrchestra: A Comprehensive Foundation Model for Computational Pathology with Over 100 Diverse Clinical-Grade Tasks, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2503.24345)]

### Other Generation
- Unsupervised Learning for Cell-Level Visual Representation in Histopathology Images With Generative Adversarial Networks, <ins>JBHI, 2019</ins> [[Paper](https://ieeexplore.ieee.org/document/8402089)]
- A visual–language foundation model for pathology image analysis using medical Twitter, <ins>Nature Medicine, 2023</ins> [[Paper](https://www.nature.com/articles/s41591-023-02504-3)]
- Hierarchical Text-to-Vision Self Supervised Alignment for Improved Histopathology Representation Learning, <ins>MICCAI, 2024</ins> [[Paper](https://arxiv.org/abs/2403.14616)]
- Tertiary Lymphoid Structures Generation Through Graph-Based Diffusion, <ins>MICCAI, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-55088-1_4)]
- SynCellFactory: Generative Data Augmentation for Cell Tracking, <ins>MICCAI, 2024</ins> [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_29)]
- Knowledge-enhanced visual-language pretraining for computational pathology, <ins>ECCV, 2024</ins> [[Paper](https://arxiv.org/abs/2404.09942)]
- Transcriptomics-Guided Slide Representation Learning in Computational Pathology, <ins>CVPR, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10656058)]
- PLUTO: Pathology-Universal Transformer, <ins>ICML, 2024</ins> [[Paper](https://openreview.net/forum?id=KGZGqMwPwh)]
- Prompting Vision Foundation Models for Pathology Image Analysis, <ins>CVPR, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10658362)]
- Towards a text-based quantitative and explainable histopathology image analysis, <ins>MICCAI, 2024</ins> [[Paper](https://arxiv.org/abs/2407.07360)]
- Improving 3D deep learning segmentation with biophysically motivated cell synthesis, <ins>Communications Biology, 2024</ins> [[Paper](https://arxiv.org/abs/2408.16471)]
- Spatial Diffusion for Cell Layout Generation, <ins>MICCAI, 2024</ins> [[Paper](https://papers.miccai.org/miccai-2024/717-Paper2613.html)]
- TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2412.06011)]
- Promptable Representation Distribution Learning and Data Augmentation for Gigapixel Histopathology WSI Analysis, <ins>AAAI, 2024</ins> [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32779)]
- AugDiff: Diffusion based feature augmentation for multiple instance learning in whole slide image, <ins>IEEE Transactions on Artificial Intelligence, 2024</ins> [[Paper](https://sider.ai/zh-CN/wisebase/chat/ai-inbox?fid=680b3a91dc8f8d6077fd7ff4&cid=680b3ad7dc8f8d6077fd8380)]
- Dcdiff: Dual-granularity cooperative diffusion models for pathology image analysis, <ins>IEEE TMI, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10577168)]
- Molecular-driven Foundation Model for Oncologic Pathology, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2501.16652)]
- Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images, <ins>ICLR, 2025</ins> [[Paper](https://arxiv.org/abs/2501.15598)]
- MLLM4PUE: Toward Universal Embeddings in Digital Pathology through Multimodal LLMs, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2502.07221)]
- Robust Multimodal Survival Prediction with the Latent Differentiation Conditional Variational AutoEncoder, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2503.09496)]
- MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification, <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2502.07409)]



## ❤️ Community Support
🌱 **Generative Pathology** is a rapidly evolving and highly promising research area. We anticipate that more researchers and practitioners will join in advancing this exciting frontier.

💎  We welcome pull requests from the community! While individual efforts are valuable, collaborative contributions are essential to building a sustainable and comprehensive resource. If you'd like to add papers, improve taxonomy, or fix typos, feel free to fork the repo and [submit a pull request](https://github.com/yuanzhang7/Awesome-Generative-Models-in-Pathology/pulls).  

🌼 Please consider starring, forking, or contributing to the repository on GitHub: [👉 GitHub Link Here](https://github.com/yuanzhang7/Awesome-Generative-Models-in-Pathology)

🙌 Let’s build a stronger community for **Generative Computational Pathology** — together!
