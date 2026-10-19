# Academic & Technical Project Report
## Explainable AI (XAI) for Deepfake Detection Using XceptionNet, Grad-CAM, and LIME

---

### Student & Institutional Details
- **Institution**: Ajeenkya DY Patil School of Engineering, Dr. D. Y. Patil Knowledge City, Charholi Bk., Via Lohegaon, Pune – 411081
- **Department**: Department of Artificial Intelligence and Data Science Engineering
- **Academic Year**: 2026–27
- **Subject**: PCC302AID: Machine Learning
- **Class**: TE-C | **Semester**: V
- **Subject Incharge**: Prof. Ashima Mahendra
- **Student Name**: [Your Name]
- **Roll No**: [Your Roll No]
- **Date of Submission**: October 19, 2026

---

## 1. Title

**"Design and Implementation of an Explainable Deepfake Detection Framework Leveraging Depthwise Separable Convolutions (XceptionNet), Grad-CAM, and LIME Interpretability"**

---

## 2. Real-World Problem

### 2.1 Background
Recent breakthroughs in deep generative modeling—specifically Generative Adversarial Networks (GANs), Autoencoders (AEs), and latent diffusion models—have democratized the ability to manipulate digital facial imagery with photorealistic accuracy. Known colloquially as **Deepfakes**, these synthetic media manipulate identities, expressions, and speech in video sequences. While generative AI offers benign applications in entertainment, visual effects, and gaming, malicious actors routinely exploit this technology to forge non-consensual content, fabricate statements by public leaders, and orchestrate fraud.

### 2.2 Problem Faced
Detecting manipulated digital imagery manually is nearly impossible for human observers because modern synthesis operates at pixel-level micro-textures. Furthermore, early detection approaches that relied on low-level hand-crafted artifacts (e.g., color saturation differences, unnatural blinking, or JPEG block boundaries) quickly become obsolete as generative models improve. 

Contemporary deep neural networks achieve high classification accuracy, but they suffer from the **"Black-Box" dilemma**:
1. They output an uncalibrated scalar score without providing spatial justification.
2. They frequently latch onto **spurious correlations** (e.g., compression noise, background lighting, or microphone watermarks) rather than genuine facial forgery seams.
3. In courtroom forensics, law enforcement, and critical financial KYC (Know Your Customer) systems, an unexplainable verdict cannot be admitted as credible legal evidence.

### 2.3 People and Organizations Affected
1. **Public Figures & Government Institutions**: Vulnerable to geopolitical disinformation, fabricated political statements, and election interference.
2. **Financial Institutions & Fintechs**: Digital onboarding and biometrics-based banking systems face severe vulnerabilities from identity impersonation and presentation attacks.
3. **Journalism & News Media**: Newsrooms struggle to verify user-generated footage before broadcasting, risking their institutional credibility.
4. **General Public**: Citizens, especially women and vulnerable populations, are targeted with non-consensual deepfakes, reputation damage, and social engineering scams.

### 2.4 Need for a Solution
There is an urgent requirement for an **end-to-end detection framework** that not only achieves state-of-the-art accuracy across diverse manipulation techniques (FaceSwap, DeepFakes, Face2Face, NeuralTextures) but also provides **visual, verifiable, and interpretable explanations**. The system must pinpoint the precise biological anomalies, blending boundary seams, and texture warps that explain *why* a particular decision was reached.

---

## 3. Role of Machine Learning

### 3.1 ML Problem Type
- **Supervised Binary Classification**: The task maps an input facial image tensor $x \in \mathbb{R}^{H \times W \times C}$ to a discrete class label $y \in \{0, 1\}$:
  - Class $0$: **Authentic (Real)** face
  - Class $1$: **Manipulated (Fake)** face
- **Probability Estimation**: Calibrated posterior probability estimation $\hat{p} = P(y = 1 \mid x) \in [0, 1]$ via the Sigmoid link function.
- **Model Interpretability / Explainable AI (XAI)**: Post-hoc white-box (Grad-CAM) and model-agnostic black-box (LIME) attribution mapping.

### 3.2 How Machine Learning Addresses the Problem
Machine learning models, particularly deep convolutional architectures, automatically learn hierarchical representations:
- **Low-Level Layers**: Detect local edges, color gradients, and compression artifacts.
- **Mid-Level Layers**: Identify facial landmark spatial configurations, skin pore micro-textures, and boundary blending seams.
- **High-Level Layers**: Synthesize contextual biological consistency across eyes, teeth, mouth, and cheek contours.

By training on thousands of real and synthetic frames, the model learns the subtle mathematical distribution differences between natural biological skin photography and generative neural upsampling artifacts.

### 3.3 Input and Features
- **Raw Input**: High-resolution digital video files (`.mp4`, `.avi`) or static photographic images (`.jpg`, `.png`).
- **Extracted Input Feature Tensor**: $128 \times 128 \times 3$ RGB facial crops normalized to the range $[0.0, 1.0]$.
- **Context Margin Scaling ($1.5\times$)**: Facial bounding boxes are expanded by $50\%$ to encompass outer jawlines, neck transitions, and hairlines where face-swapping blending masks are applied.

### 3.4 Expected Output
1. **Predicted Class**: Binary classification label (`REAL` or `FAKE`).
2. **Confidence Metric**: Confidence score expressed as a percentage ($50.0\% - 100.0\%$).
3. **Calibrated Fake Probability**: $P(\text{Fake}) \in [0.0, 1.0]$ and raw output logit $z \in \mathbb{R}$.
4. **Visual XAI Explanations**:
   - Continuous 2D **Grad-CAM class activation heatmap** overlaid on the face.
   - Segmented **LIME superpixel contribution masks** (green for positive forgery evidence, red for authentic biometric preservation).
5. **Detection Signature**: Specific artifact attribution (e.g., *FaceForensics++ Blending/Compression Artifacts*, *Celeb-DF Generative Synthesis/Facial Warping*, or *Dual Consensus*).

### 3.5 Why Machine Learning is Suitable
Rule-based heuristic programming cannot keep pace with evolving generative models. Generative models operate in non-linear high-dimensional spaces ($128 \times 128 \times 3 = 49,152$ dimensions). Only statistical learning algorithms capable of gradient-based optimization can discover the subtle high-frequency artifacts (such as checkerboard patterns from transpose convolutions and spatial boundary discontinuities) that define synthetic imagery.

---

## 4. Data and Machine Learning Techniques

### 4.1 Data Sources
The model was trained and evaluated on two benchmark forensic datasets:

1. **FaceForensics++ (FF++)**:
   - **Scale**: 1,000 pristine video sequences collected from 977 YouTube videos with frontal, trackable human faces.
   - **4 Automated Manipulation Methods**:
     - *DeepFakes*: Autoencoder-based latent face replacement.
     - *Face2Face*: Real-time facial reenactment transferring source facial expressions.
     - *FaceSwap*: 3D graphics model fitting and texture blending.
     - *NeuralTextures*: Facial modification driven by neural texture maps.
   - **Compression Variants**: Evaluated on high-quality (C23) and compressed (C40) video streams.

2. **Celeb-DF (v2)**:
   - **Scale**: 590 original celebrity interview clips and 5,639 high-quality deepfake video clips.
   - **Characteristics**: Features diverse ages, ethnic groups, genders, and lighting conditions. Designed specifically to eliminate 1st-generation flaws (such as visible color mismatches and erratic blinking), presenting a state-of-the-art forensic detection challenge.

### 4.2 Dataset Comparison Table

| Attribute | FaceForensics++ (FF++) | Celeb-DF (v2) |
| :--- | :--- | :--- |
| **Original Real Videos** | 1,000 | 590 |
| **Manipulated Fake Videos** | 4,000 | 5,639 |
| **Total Videos** | 5,000 | 6,229 |
| **Primary Artifact Types** | Compression seams, boundary blending steps | Subtle micro-texture smoothing, eye reflection asymmetry |
| **Visual Realism** | Moderate to High | Very High (Benchmark standard) |
| **Preprocessing Input Size** | $128 \times 128$ normalized RGB crops | $128 \times 128$ normalized RGB crops |

### 4.3 Important Features & Preprocessing Pipeline
1. **Frame Subsampling**: Video sequences are ingested using OpenCV (`cv2.VideoCapture`), sampling every 10–20 frames to avoid redundant contiguous frames.
2. **Face Localization & 1.5x Margin Expansion**: Extracted using `dlib.get_frontal_face_detector()` and OpenCV Haar Cascades. The bounding box dimensions $(w, h)$ are scaled by a factor of $1.5$ to retain the jawline and cheek boundaries.
3. **Data Transformations**:
   ```python
   transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize((128, 128)),
       transforms.ToTensor()
   ])
   ```
4. **Data Partitioning**:
   - **Training Set**: $80\%$ of extracted facial frames.
   - **Validation Set**: $20\%$ of extracted facial frames.
   - **Independent Evaluation Set**: $469$ verified held-out benchmark facial portraits.

### 4.4 Machine Learning Paradigm
- **Supervised Deep Transfer Learning**: Initialized with ImageNet-1K pretrained weights via `timm` to leverage rich low-level spatial filter representations, followed by fine-tuning on deepfake domain datasets.
- **Explainable AI (XAI)**: Post-hoc attribution using gradient backpropagation (Grad-CAM) and locally weighted surrogate linear modeling (LIME).
- **Dual-Specialist Ensemble Voting**: Heuristic consensus combining Celeb-DF and FaceForensics++ models.

---

### 4.5 Algorithms Implemented & Mathematical Foundations

```mermaid
graph LR
    A[Input Face 128x128] --> B[XceptionNet Backbone]
    B --> C[Depthwise Separable Convolutions]
    C --> D[Adaptive Global Avg Pool: 2048-D]
    D --> E[Linear Projection Layer]
    E --> F[Sigmoid: P(Fake)]
    
    C -.->|Gradients of Target Class| G[Grad-CAM Heatmap]
    A -.->|Perturbations| H[LIME Superpixels]
    H -.->|Ridge Regression| I[Contribution Mask]
```

#### A. XceptionNet (Extreme Inception)
XceptionNet decouples spatial convolutions from cross-channel convolutions by utilizing **Depthwise Separable Convolutions**:
1. **Depthwise Convolution**: Applies an independent spatial filter ($3 \times 3$) to each input channel separately.
2. **Pointwise Convolution**: Applies a $1 \times 1$ convolution across channels to compute linear combinations.

**Computational Efficiency**:
$$\text{Computational Ratio} = \frac{\text{Cost}_{\text{separable}}}{\text{Cost}_{\text{standard}}} = \frac{1}{D_{\text{out}}} + \frac{1}{D_k^2} \approx \frac{1}{9} \approx 11\%$$
This architecture reduces parameter count and FLOPs by nearly $89\%$ compared to standard convolutions while capturing high-frequency edge and texture artifacts through 14 modular residual blocks.

#### B. Loss Function: Binary Cross-Entropy with Logits
The model optimizes `nn.BCEWithLogitsLoss`, which integrates the Sigmoid activation and negative log-likelihood into a numerically stable log-sum-exp formulation:
$$\mathcal{L}(z_i, y_i) = \max(z_i, 0) - z_i \cdot y_i + \log\left(1 + e^{-|z_i|}\right)$$

#### C. Optimization: Adam (Adaptive Moment Estimation)
Parameters $\theta$ are updated using moving averages of past gradients ($m_t$) and squared gradients ($v_t$) with learning rate $\eta = 0.0001$:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

#### D. Grad-CAM (Gradient-Weighted Class Activation Mapping)
Grad-CAM computes the importance weight $\alpha_k^c$ for each feature map $k$ in layer $l$ (specifically `block7` and `conv1` of XceptionNet):
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i, j}^k}$$
The activation map is synthesized using a rectified linear combination:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
Only positive gradients contributing to the "Fake" verdict are preserved and rendered as a JET colormap heatmap.

#### E. LIME (Local Interpretable Model-Agnostic Explanations)
LIME segments the facial image into superpixels via Quickshift, perturbs them, and fits an interpretable linear surrogate model via **weighted Ridge Regression**:
$$\min_{w} \sum_{k=1}^M \pi(z_k) \left( f(z_k) - (w_0 + w^T z'_k) \right)^2 + \lambda \|w\|_2^2$$
- $w_j > 0$: Regional superpixel contributes positively to the fake prediction (rendered **green**).
- $w_j < 0$: Regional superpixel supports authenticity (rendered **red**).

---

### 4.6 Classical Machine Learning Benchmark Comparison

To ground the deep learning architecture in classical machine learning theory, the project includes an automated benchmark tool (`evaluate_classical_ml_baseline.py`) comparing classical algorithms across handcrafted features (HOG + Color Histograms) and deep bottleneck embeddings (XceptionNet 2048-D features):

| Feature Set | Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HOG + Color Histograms** | Logistic Regression (L2) | 100.0%* | 100.0% | 100.0% | 100.0% |
| **HOG + Color Histograms** | Linear Support Vector Machine | 100.0%* | 100.0% | 100.0% | 100.0% |
| **HOG + Color Histograms** | Non-Linear SVM (RBF Kernel) | 100.0%* | 100.0% | 100.0% | 100.0% |
| **HOG + Color Histograms** | Random Forest (100 Trees) | 100.0%* | 100.0% | 100.0% | 100.0% |
| **XceptionNet 2048-D Embeddings** | Logistic Regression (L2) | 100.0%* | 100.0% | 100.0% | 100.0% |
| **XceptionNet 2048-D Embeddings** | Support Vector Machine (RBF) | 100.0%* | 100.0% | 100.0% | 100.0% |
| **End-to-End XceptionNet (Full)** | Deep Convolutional Network | **99.0%** | **100.0%** | **98.9%** | **0.9947** |

*\*Note: Evaluated on curated benchmark test sample distribution.*

---

### 4.7 Relevant Evaluation Measures
Performance is evaluated across standard statistical classification metrics:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Quantitative Results on 469 Held-Out Benchmark Samples:
- **True Positives ($TP$)**: $280$ (Manipulated faces correctly identified)
- **True Negatives ($TN$)**: $186$ (Authentic faces correctly verified)
- **False Positives ($FP$)**: $0$ (Zero legitimate portraits falsely accused!)
- **False Negatives ($FN$)**: $3$ (Subtly blended fakes misclassified)
- **Overall Test Accuracy**: **$99.0\%$**
- **Precision**: **$100.0\%$** ($1.00$)
- **Recall**: **$98.94\%$** ($0.989$)
- **F1-Score**: **$0.9947$**
- **Validation Accuracy (Celeb-DF)**: **$98.2\%$**
- **Validation Accuracy (FaceForensics++)**: **$97.5\%$**

---

## 5. Applications, Benefits, and Limitations

### 5.1 Real-World Applications
1. **Digital Forensic Investigation**: Law enforcement agencies can generate evidentiary dossiers containing Grad-CAM heatmaps showing exact forgery boundaries.
2. **Social Media & Broadcast Moderation**: High-throughput automated screening pipelines for platforms (YouTube, X, Meta) to detect viral synthetic disinformation.
3. **Biometric Security & Digital Banking (KYC)**: Securing mobile facial verification against digital presentation attacks and deepfake identity spoofing.
4. **Fact-Checking & Investigative Journalism**: Empowering news organizations with automated, interpretable verification tools.

### 5.2 Benefits
- **High Forensic Accuracy**: $99.0\%$ accuracy on benchmark datasets.
- **Explainability**: Eliminates the "black box" by generating visual heatmaps and regional contribution segmentations.
- **Ensemble Robustness**: Combines specialist detectors for both high-compression artifacts and advanced generative synthesis.
- **Multi-Modal Deployment**: Accessible via both an interactive **Flask Web Dashboard** (`app.py`) and an automated **CLI tool** (`detect.py`).

### 5.3 Challenges and Limitations
1. **Severe Compression Degradation (C40+)**: Aggressive video compression (H.264 / H.265) destroys high-frequency edge information, reducing recall on heavily compressed viral clips.
2. **Extreme Facial Angles**: Significant head rotation ($> 60^\circ$ yaw) or heavy facial occlusions (sunglasses, masks) can degrade face crop extraction.
3. **Generalization to Unseen Generative Architectures**: Models trained primarily on GAN deepfakes require periodic retraining to maintain robustness against newer latent diffusion architectures.

### 5.4 Ethical, Social, and Environmental Considerations
- **Protection Against Defamation**: Helps safeguard individuals against synthetic extortion and targeted harassment.
- **Mitigating False Accusations**: In legal contexts, falsely labeling authentic media as fake carries severe consequences; our system maintains a conservative threshold ($FP = 0$).
- **Demographic Bias Auditing**: Explainability heatmaps allow researchers to inspect whether detectors treat diverse skin tones and facial geometries equitably.
- **Computational Efficiency**: Depthwise separable convolutions reduce power consumption and carbon footprint during inference, making lightweight CPU/edge deployment feasible.

---

## 6. Conclusion

In this project, an end-to-end Explainable AI deepfake detection framework was successfully designed, trained, and deployed. By leveraging **XceptionNet's Depthwise Separable Convolutions**, the model achieves high computational efficiency alongside strong representational power, attaining **98.2% validation accuracy on Celeb-DF v2**, **97.5% on FaceForensics++**, and **99.0% accuracy on held-out test sets**. 

Crucially, the integration of **Grad-CAM** and **LIME** addresses the black-box nature of deep learning, providing visual proof of synthesis boundaries, warping seams, and unnatural micro-textures. Supported by an interactive web dashboard, CLI pipeline, and classical machine learning baselines, this system provides a robust, transparent, and legally defensible methodology for digital media authentication.

---

## 7. Course & Curriculum Mappings (TE 2024 Pattern)

### Course Outcome (CO) Mapping:
- **CO1**: Formulate and analyze real-world computer vision problems using supervised machine learning paradigms.
- **CO2**: Apply data preprocessing, face detection, bounding box normalization, and tensor transformation pipelines.
- **CO3**: Design and evaluate deep learning architectures (XceptionNet) with custom loss functions and optimizers.
- **CO4**: Implement model interpretability techniques (Explainable AI - Grad-CAM and LIME) to audit machine learning decisions.
- **CO5**: Evaluate model performance using confusion matrices, precision, recall, and F1-scores.

### Program Outcome (PO) Mapping:
- **PO1 (Engineering Knowledge)**: Applied principles of depthwise separable convolutions, matrix calculus, and gradient pooling.
- **PO2 (Problem Analysis)**: Identified digital forensic vulnerabilities in synthetic media and formulated a binary classification solution.
- **PO3 (Design/Development of Solutions)**: Built an interactive web dashboard, CLI tool, and modular inference engine.
- **PO4 (Conduct Investigations of Complex Problems)**: Investigated synthetic manipulation artifacts using Grad-CAM and LIME heatmaps.
- **PO5 (Modern Tool Usage)**: Leveraged PyTorch, timm, OpenCV, scikit-learn, Flask, and Git.
- **PO8 (Ethics)**: Addressed misinformation risks, identity theft, and ethical considerations in biometric forensic analysis.

### Sustainable Development Goal (SDG) Mapping:
- **SDG 9 (Industry, Innovation, and Infrastructure)**: Fostering innovative, trustworthy AI solutions for digital infrastructure security.
- **SDG 16 (Peace, Justice, and Strong Institutions)**: Combating disinformation, identity fraud, and protecting democratic institutions through authentic digital media verification.

---

## 8. References

1. Chollet, François. *"Xception: Deep Learning with Depthwise Separable Convolutions."* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1251–1258, 2017.
2. Selvaraju, Ramprasaath R., et al. *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization."* IEEE International Conference on Computer Vision (ICCV), pp. 618–626, 2017.
3. Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. *"'Why Should I Trust You?': Explaining the Predictions of Any Classifier."* ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pp. 1135–1144, 2016.
4. Rössler, Andreas, et al. *"FaceForensics++: Learning to Detect Facial Forgeries with Facial Manipulation."* IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
5. Li, Yuezun, et al. *"Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics."* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3207–3216, 2020.
6. Kingma, Diederik P., and Jimmy Ba. *"Adam: A Method for Stochastic Optimization."* 3rd International Conference for Learning Representations (ICLR), 2015.
7. Malolan, B., Parekh, A., and Kazi, F. *"Explainable Deep-Fake Detection Using Visual Interpretability Methods."* 3rd International Conference on Information and Computer Technologies (ICICT), pp. 289–293, 2020.
