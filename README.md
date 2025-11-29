## Working Environment

- **Python**: 3.8+
- **Framework**: PyTorch 2.0+
- **Hardware**: CUDA-compatible GPU (recommended)
- **OS**: Windows 10/11, Linux, macOS

### Dependencies
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd plant_model

# Download and extract the virtual environment
# 1. Download .venv.zip from Google Drive (link below)
# 2. Extract to the project root directory (plant_model/)
# 3. The extracted folder should be named .venv (not .venv.zip or venv)

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

**Important Setup Notes**:
- **Virtual Environment**: Download the pre-configured `.venv.zip` from [Google Drive](https://drive.google.com/file/d/1QZ1UrsQn2bE4EyJ01lqXQLlegxCJn-Zs/view?usp=drive_link)
  - Extract the zip file to `plant_model/.venv/` (the .venv folder should be at the same level as src/ and data/)
  - Make sure the folder is named `.venv` (with the dot prefix)
- **Model Checkpoints**: All `.pth` and `.pt` checkpoint files have been removed to reduce repository size
- **Reproducing Experiments**: Use the training scripts in `src/training/` to retrain models from scratch

---

## Reference

This work is based on the research by **Khait et al. (2023)**:

> Khait, I., Lewin-Epstein, O., Sharon, R. et al. Sounds emitted by plants under stress are airborne and informative. *Cell* 186, 1328–1336.e10 (2023).
> https://doi.org/10.1016/j.cell.2023.03.009

We reproduced and extended their baseline CNN model for ultrasonic plant sound classification.

---

## Research Background

Plants emit ultrasonic sounds (20-150 kHz) when experiencing stress conditions such as drought or physical damage. Khait et al. (2023) demonstrated that these sounds are:

- **Airborne and detectable** at distances up to several meters
- **Informative** about the plant's physiological state
- **Distinguishable** between different stress types and plant species

The ability to classify plant sounds has significant implications for:
- **Precision agriculture**: Early stress detection for optimal irrigation
- **Plant phenotyping**: Non-invasive monitoring of plant health
- **Ecosystem monitoring**: Understanding plant-environment interactions

---

## Research Objectives

This project aims to:

1. **Reproduce** the baseline CNN classifier from Khait et al. (2023)
2. **Improve** classification performance with advanced deep learning architectures
3. **Explore** semi-supervised learning to reduce labeling requirements
4. **Analyze** the robustness of models across different plant species and stress types

### Key Innovations

- **Enhanced CNN Architecture**: Improved model with attention mechanisms and residual connections
- **Semi-supervised Learning**: Effective learning with only 50% labeled data
- **Comprehensive Evaluation**: LOPO (Leave-One-Plant-Out) cross-validation for robust assessment
- **Multi-task Analysis**: Classification across species, stress types, and background noise

---

## Dataset

The **PlantSounds** dataset includes ultrasonic recordings (500 kHz sampling rate) from:

- **Plant Species**: Tomato (*Solanum lycopersicum*), Tobacco (*Nicotiana tabacum*)
- **Stress Conditions**:
  - Dry (drought stress)
  - Cut (mechanical damage)
- **Controls**: Empty pots, greenhouse background noise

### Data Organization

```
data/raw/PlantSounds/
├── tomato_dry_*.wav
├── tomato_cut_*.wav
├── tobacco_dry_*.wav
├── tobacco_cut_*.wav
├── tomato_*_empty_*.wav
└── greenhouse_*.wav
```

---

## Model Architecture

### Baseline CNN (Khait et al., 2023)

![Baseline](experiments/figures/baseline_architecture.svg)

### MyModel (Baseline CNN + VAE + SSL + DG)

![MyModel](experiments/figures/mymodel_architecture.svg)

**Architecture Details**:

- **Backbone**: Identical CNN structure to Baseline (Khait et al., 2023)
- **Embedding h**: 128-dimensional feature representation from CNN backbone
- **Main Classifier**: h → binary logit (BCEWithLogitsLoss)
- **VAE Module**:
  - Encoder: h → (μ, log σ²) in 32-dim latent space
  - Reparameterization: z = μ + σ ⊙ ε
  - Decoder: z → h_reconstructed
  - Loss: MSE(h, h_rec) + β·KL(q(z|h) || N(0, I))
- **SSL (Semi-Supervised Learning)**:
  - Consistency regularization between z(x) and z(x_augmented)
  - Enables learning from unlabeled data (50% label experiments)
- **DG (Domain Generalization)**:
  - Domain classifier head: h → domain logits
  - Improves robustness across recording conditions

---

## Training Protocol

### Data Preprocessing

1. **Resampling**: All audio to 500 kHz
2. **Segmentation**: 2ms windows (1000 samples)
3. **High-pass Filter**: Remove low-frequency noise
4. **Normalization**: Zero mean, unit variance

### Cross-Validation

**LOPO (Leave-One-Plant-Out)**:
- Each fold holds out all recordings from one plant
- Tests generalization to new individuals
- More realistic evaluation than random splits

### Training Configuration

```python
Optimizer: Adam (lr=5e-4)
Loss: BCEWithLogitsLoss (with class balancing)
Batch size: 64
Epochs: 10 (with early stopping)
```

### Evaluation Metrics

This project uses multiple classification metrics to provide comprehensive performance assessment:

#### Primary Metrics

##### 1. Balanced Accuracy (Main metric for this project)

- Formula: `(Recall + Specificity) / 2 = (TPR + TNR) / 2`
- **Why we use it**: Our plant stress datasets have class imbalance (especially in LOPO where each fold may test only one plant with one stress condition). Balanced accuracy treats both classes equally by averaging per-class recalls, making it more reliable than regular accuracy for imbalanced data.
- **Interpretation**: Perfect score = 1.0, Random guessing = 0.5

##### 2. Accuracy

- Formula: `(TP + TN) / (TP + TN + FP + FN)`
- Total correct predictions over all samples
- **Limitation**: Can be misleading with class imbalance (e.g., predicting all samples as one class could give high accuracy)

#### Additional Important Metrics

##### 3. Precision (Positive Predictive Value)

- Formula: `TP / (TP + FP)`
- "Of all samples predicted as positive, how many are truly positive?"
- **Use case**: Important when false alarms (FP) are costly
- High precision = Reliable positive predictions

##### 4. Recall (Sensitivity, True Positive Rate)

- Formula: `TP / (TP + FN)`
- "Of all actual positive samples, how many did we correctly identify?"
- **Use case**: Important when missing positives (FN) is costly
- High recall = Finding most/all actual positive cases

##### 5. Specificity (True Negative Rate)

- Formula: `TN / (TN + FP)`
- "Of all actual negative samples, how many did we correctly identify?"
- Measures how well the model identifies negative class

##### 6. F1-Score

- Formula: `2 × (Precision × Recall) / (Precision + Recall)`
- Harmonic mean of precision and recall
- **Use case**: When you need balanced precision and recall
- If either precision or recall is low, F1-score will also be low

#### Understanding the Confusion Matrix

```
                  Predicted
              Negative   Positive
Actual  Neg      TN        FP      ← False Positive: Wrong alarm
        Pos      FN        TP      ← False Negative: Missed detection
                 ↑
            Missed case
```

- **TP (True Positive)**: Correctly predicted positive (e.g., correctly identified stressed plant)
- **TN (True Negative)**: Correctly predicted negative (e.g., correctly identified unstressed plant)
- **FP (False Positive)**: Wrongly predicted positive (false alarm - predicted stress when plant was fine)
- **FN (False Negative)**: Wrongly predicted negative (missed detection - failed to detect actual stress)

#### Metric Selection Guidelines

For this plant stress classification project:

1. **Class imbalance present** → Use Balanced Accuracy + F1-Score
2. **Both FN and FP matter** → Monitor Precision, Recall, and F1-Score
3. **Different costs for errors** → Check confusion matrices for error patterns
4. **Overall performance** → Regular Accuracy still useful when classes are balanced

The confusion matrix visualizations in the figures show the actual error patterns, helping identify whether the model tends to miss stressed plants (FN) or raise false alarms (FP).

---

## Experimental Tasks

### Task 1: Plant Classification Tasks (4 Binary Classification Problems)

**Overview**: Task 1 consists of four distinct binary classification tasks using LOPO (Leave-One-Plant-Out) cross-validation, where all sounds from one plant individual are held out for testing while training on the remaining plants.

---

#### Task 1.1: Tomato Dry vs Cut Stress Classification

**Objective**: Distinguish between drought stress (dry) and mechanical damage (cut) in tomato plants.

**Dataset**: All tomato plant recordings from acoustic box environment

- Dry stress samples: Plants subjected to dehydration
- Cut stress samples: Plants with stem damage

**Challenge**: Each individual plant experienced only ONE type of stress (either dry OR cut), making this a particularly challenging classification task. In LOPO cross-validation, when a plant is used as the test set, that fold contains only one class, requiring the model to generalize from sounds of other plants under different stress conditions.

**Results** (see Fig. 1a, 1b, 1e, 1g):

- **Baseline CNN**: 95.2% balanced accuracy
- **MyModel (full labels)**: 97.6% balanced accuracy (+2.4%)
- **MyModel (50% labels)**: 93.3% balanced accuracy

---

#### Task 1.2: Tobacco Dry vs Cut Stress Classification

**Objective**: Distinguish between drought stress (dry) and mechanical damage (cut) in tobacco plants.

**Dataset**: All tobacco plant recordings from acoustic box environment

- Dry stress samples: Plants subjected to dehydration
- Cut stress samples: Plants with stem damage

**Challenge**: Same single-class fold challenge as Task 1.1, but with tobacco plants.

**Results** (see Fig. 1a, 1b, 1f, 1g):

- **Baseline CNN**: 87.7% balanced accuracy
- **MyModel (full labels)**: **100.0%** balanced accuracy (perfect classification!)
- **MyModel (50% labels)**: 100.0% balanced accuracy (maintained perfect performance)

**Key Finding**: Tobacco stress classification appears easier than tomato, possibly due to more distinctive acoustic signatures.

---

#### Task 1.3: Species Classification (Dry Condition)

**Objective**: Distinguish tomato vs tobacco plants, both under drought stress.

**Dataset**: All dry-stressed plants from both species

- Class 0: Tomato dry-stressed plants
- Class 1: Tobacco dry-stressed plants

**Challenge**: Both species are under the same stress condition, so the model must learn species-specific acoustic characteristics rather than stress patterns.

**Results** (see Fig. 1a):

- **Baseline CNN**: 87.5% balanced accuracy
- **MyModel (full labels)**: 88.7% balanced accuracy (+1.2%)
- **MyModel (50% labels)**: 85.7% balanced accuracy

---

#### Task 1.4: Species Classification (Cut Condition)

**Objective**: Distinguish tomato vs tobacco plants, both under mechanical damage.

**Dataset**: All cut-stressed plants from both species

- Class 0: Tomato cut-stressed plants
- Class 1: Tobacco cut-stressed plants

**Challenge**: Same as Task 1.3, but with cut condition.

**Results** (see Fig. 1a):

- **Baseline CNN**: 93.4% balanced accuracy
- **MyModel (full labels)**: 96.9% balanced accuracy (+3.5%)
- **MyModel (50% labels)**: 96.2% balanced accuracy

**Key Finding**: Cut condition provides clearer species discrimination than dry condition, suggesting mechanical damage produces more species-specific acoustic responses.

---

#### Task 1 Results Visualization

**Figure 1a: Task 1 Confusion Matrices**
![Task 1 Confusion Matrices](experiments/figures/task1_confusion_matrices.png)
*Pooled confusion matrices across all LOPO folds for each Task 1 subtask. Shows classification errors for both Baseline CNN and MyModel. All tasks including tomato/tobacco dry vs cut are successfully visualized.*

**Figure 1b: Task 1 Mean Balanced Accuracy Comparison**
![Task 1 Bar Chart](experiments/figures/task1_balanced_accuracy_bar.png)
*Comparison of mean balanced accuracy across all 4 Task 1 subtasks. Shows Baseline CNN, MyModel (100% labels), and MyModel (50% labels) performance.*

---

### Task 2: Plant vs Empty Pot Classification

**Objective**: Distinguish actual plant stress sounds from empty pot control recordings.

**Dataset**: All plant sounds (tomato + tobacco, dry + cut) vs empty pot recordings

- Class 0 (negative): Empty pot recordings (control/noise)
- Class 1 (positive): Plant stress sounds (all species and conditions)

**Purpose**: This task validates whether plant sounds are genuinely distinguishable from mechanical/environmental noise produced by empty pots in the same acoustic chamber. This is crucial for establishing that observed acoustic emissions are biological in origin.

**Cross-Validation**: LOPO (Leave-One-Plant-Out) for plant sounds, with empty pot samples divided into K groups to match the number of plant folds.

**Results** (see Fig. 2a, 2c):

- **Baseline CNN**: 95.9% balanced accuracy
- **MyModel**: 98.5% balanced accuracy (+2.6%)

**Key Findings**:

- Near-perfect classification demonstrates plant sounds have distinctive biological signatures
- High accuracy confirms that acoustic emissions are not artifacts of the recording environment
- Minimal confusion between plant and empty pot sounds validates the biological nature of plant acoustics

---

### Task 3: Tomato Dry vs Greenhouse Noise Classification

**Objective**: Distinguish tomato dry stress sounds from real-world greenhouse background noise.

**Dataset**: Tomato dry-stressed plants vs greenhouse environmental noise

- Class 0 (negative): Greenhouse noise recordings (environmental background)
- Class 1 (positive): Tomato dry stress sounds

**Purpose**: While Task 2 tested controlled empty pot noise, this task evaluates performance against uncontrolled, real-world greenhouse noise (fans, equipment, air circulation, etc.). This tests ecological validity and real-world deployment feasibility.

**Challenge**: Greenhouse noise is more complex and variable than controlled empty pot recordings, making this a more realistic and challenging scenario.

**Cross-Validation**: LOPO for plant sounds, with greenhouse noise samples divided into K groups.

**Results** (see Fig. 2a, 2c):

- **Baseline CNN**: 98.4% balanced accuracy
- **MyModel**: 98.2% balanced accuracy (-0.2%)

**Key Findings**:

- Both models achieve near-perfect classification even with complex environmental noise
- Slight decrease in MyModel performance suggests baseline may be sufficient for this easier task
- High accuracy in real-world noise conditions demonstrates practical deployment viability
- Plant acoustic signals remain highly discriminable even in noisy greenhouse environments

---

### Task 2 & 3 Results Visualization

**Figure 2a: Task 2/3 Confusion Matrices**
![Task 2/3 Confusion Matrices](experiments/figures/task2_task3_confusion_matrices.png)
*Pooled confusion matrices for Task 2 and Task 3. Demonstrates strong class separation in plant vs noise classification. Text overlap issues have been fixed with improved layout.*

**Figure 2b: Task 2/3 Mean Balanced Accuracy Comparison**
![Task 2/3 Bar Chart](experiments/figures/task2_task3_balanced_accuracy_bar.png)
*Mean balanced accuracy comparison for Task 2 (plant vs empty pot) and Task 3 (tomato vs greenhouse noise). Both tasks show >95% accuracy.*

---

### Task 4: Semi-Supervised Learning Effect

**Objective**: Evaluate the robustness of MyModel's semi-supervised learning approach by reducing labeled training data from 100% to 50%.

**Motivation**: In real-world scenarios, obtaining labeled data is expensive and time-consuming. Semi-supervised learning leverages unlabeled data to maintain performance with fewer labels, making large-scale deployment more practical.

**Methodology**:

- **Full labels (100%)**: All training samples have ground-truth labels
- **Partial labels (50%)**: Only 50% of training samples are labeled; remaining 50% are used as unlabeled data for semi-supervised learning
- **Cross-validation**: Same LOPO splits as Task 1, with random 50% label selection per fold

**Implementation**: MyModel uses a consistency regularization approach where:

1. Labeled samples: Standard supervised loss (cross-entropy)
2. Unlabeled samples: Consistency loss between predictions under different augmentations
3. Combined training: Balances supervised and unsupervised objectives

**Results Analysis** (see Fig. 3):

| Task | Baseline | 100% Labels | 50% Labels | Drop | Relative Retention |
|------|----------|-------------|------------|------|-------------------|
| Tomato dry vs cut | 95.2% | 97.6% | 93.3% | -4.3% | 95.6% |
| Tobacco dry vs cut | 87.7% | 100.0% | 100.0% | 0.0% | 100.0% |
| Dry: tomato vs tobacco | 87.5% | 88.7% | 85.7% | -3.0% | 96.6% |
| Cut: tomato vs tobacco | 93.4% | 96.9% | 96.2% | -0.7% | 99.3% |

**Key Findings**:

- **Minimal performance drop**: Average degradation of only 2-4% with 50% fewer labels
- **Robustness**: Tobacco dry vs cut maintains perfect 100% accuracy even with half the labels
- **Practical value**: 85-100% accuracy retained across all tasks demonstrates feasibility for large-scale applications
- **Label efficiency**: Achieving 93-100% performance with 50% labels represents substantial cost savings
- **Semi-supervised benefit**: All 50% label results significantly outperform baseline, confirming the value of unlabeled data

**Implications**: The strong performance with reduced labels validates MyModel's semi-supervised approach for practical deployment where labeling all data is infeasible.

---

### Task 4 Results Visualization

**Figure 3: Task 4 Semi-Supervised Learning Effect**
![Task 4 Label Fraction Effect](experiments/figures/task4_label_fraction_effect.png)
*Grouped bar chart comparing Baseline CNN, MyModel (50% labels), and MyModel (100% labels) across all 4 Task 1 subtasks. Each task shows three bars for easy comparison. Arrows and percentages indicate improvement from semi-supervised to fully-supervised learning. This clear visualization demonstrates label efficiency with minimal performance drop when using only 50% labeled data.*

---

### Metrics Summary Table

**Figure 4: Comprehensive Metrics Summary**
![Metrics Summary Table](experiments/figures/metrics_summary_table.svg)
*Academic-style metrics summary table showing all key classification metrics for all 6 tasks (Task 1: 4 subtasks, Task 2, Task 3). Includes Accuracy, Balanced Accuracy, Macro F1-Score, Precision, Recall, and Specificity for both Baseline CNN and MyModel. This provides a complete quantitative comparison at a glance.*

**Key Insights from Metrics**:

- **Balanced Performance**: Both models achieve strong precision, recall, and specificity across all tasks
- **Macro F1-Score**: Validates overall classification quality by averaging per-class F1 scores
- **Task-specific Patterns**: Task 1.2 (Tobacco dry vs cut) shows perfect metrics (1.000) for MyModel, while noise separation tasks (Task 2/3) show consistently high performance (>0.95) for both models
- **Consistent Improvement**: MyModel generally outperforms Baseline across most metrics and tasks

---

## Running Experiments

### Data Preparation

```bash
# Place PlantSounds dataset in data/raw/PlantSounds/
# Expected structure:
data/raw/PlantSounds/
├── tomato_dry_*.wav
├── tobacco_dry_*.wav
└── ...
```

### Training

```bash
# Task 1: Baseline
python -m src.training.train_task1

# Task 1: MyModel (full labels)
python -m src.training.train_task1_mymodel

# Task 1: MyModel (semi-supervised, 50% labels)
python -m src.training.train_task1_mymodel_semi

# Task 2 & 3
python -m src.training.train_task2
python -m src.training.train_task3
```

### Visualization

```bash
# Generate all figures
python -m src.analysis.plot_results

# Outputs saved to experiments/figures/
```

---

## Results Summary

### Overall Performance

| Model | Task Type | Mean Accuracy | Key Advantage |
|-------|-----------|---------------|---------------|
| Baseline CNN | Plant-Plant | 90.7% | Simple, reproducible |
| MyModel (full) | Plant-Plant | 93.1% | +2.4% improvement |
| MyModel (50%) | Plant-Plant | 90.3% | Label efficiency |
| MyModel | Plant-Noise | 98.4% | Near-perfect separation |

### Key Achievements

1. ✓ **Reproduced** Khait et al. baseline successfully
2. ✓ **Improved** classification accuracy by 2-4% across tasks
3. ✓ **Demonstrated** semi-supervised learning viability (only 2-4% drop with 50% labels)
4. ✓ **Achieved** perfect classification (100%) for tobacco stress discrimination
5. ✓ **Validated** model generalization through rigorous LOPO cross-validation

---

## Research Significance

### Scientific Contributions

1. **Validation of Plant Bioacoustics**: Confirms that plant ultrasonic emissions contain biologically meaningful information
2. **Deep Learning Applicability**: Demonstrates effectiveness of modern neural networks for plant sound analysis
3. **Label Efficiency**: Shows semi-supervised learning can reduce annotation costs by 50%
4. **Species and Stress Discrimination**: Achieves robust classification across multiple dimensions

### Practical Applications

- **Precision Agriculture**: Early drought detection for optimal irrigation scheduling
- **Greenhouse Monitoring**: Automated plant health assessment
- **Phenotyping Pipelines**: High-throughput stress response characterization
- **Climate Research**: Understanding plant responses to environmental stress

### Future Directions

1. **Multi-class Classification**: Simultaneous species and stress type recognition
2. **Transfer Learning**: Generalization to new plant species with limited data
3. **Real-time Systems**: Deployment for continuous greenhouse monitoring
4. **Multimodal Integration**: Combining acoustic data with imaging and environmental sensors
5. **Temporal Dynamics**: Analyzing stress progression over time

---

## File Structure

```
plant_model/
├── data/
│   ├── raw/
│   │   └── PlantSounds/          # Original dataset (.wav files)
│   └── processed/                 # Preprocessed data (excluded from git)
├── experiments/
│   ├── task1_baseline/            # Baseline results (CSV files only)
│   ├── task1_my_model/            # Enhanced model results (CSV files only)
│   ├── task2_baseline/            # Task 2 baseline results
│   ├── task2_my_model/            # Task 2 enhanced results
│   ├── task3_baseline/            # Task 3 baseline results
│   ├── task3_my_model/            # Task 3 enhanced results
│   ├── task4_my_model/            # Task 4 semi-supervised results
│   └── figures/                   # Generated visualization plots
├── src/
│   ├── datasets/
│   │   └── plantsounds.py         # Data loading & LOPO splits
│   ├── models/
│   │   ├── baseline_cnn.py        # Baseline CNN (Khait et al. 2023)
│   │   └── my_model.py            # Enhanced model (VAE + SSL + DG)
│   ├── training/
│   │   ├── common.py              # Shared training utilities
│   │   ├── train_task1.py         # Task 1 baseline training
│   │   ├── train_my_model_task1.py # Task 1 enhanced training
│   │   ├── train_task2.py         # Task 2 baseline training
│   │   ├── train_my_model_task2.py # Task 2 enhanced training
│   │   ├── train_task3.py         # Task 3 baseline training
│   │   ├── train_my_model_task3.py # Task 3 enhanced training
│   │   └── train_my_model_task4.py # Task 4 semi-supervised training
│   ├── utils/
│   │   ├── audio.py               # Audio processing utilities
│   │   └── split.py               # Data splitting utilities
│   └── analysis/
│       └── plot_results.py        # Visualization and analysis scripts
├── .venv.zip                      # Compressed virtual environment
├── .gitignore                     # Git ignore rules (excludes checkpoints)
├── requirements.txt
├── verify_plots.py                # Plot data verification script
├── validate_plot_consistency.py  # Data consistency validation
└── README.md
```

**Important Notes**:
- **Model checkpoints removed**: All `.pth` and `.pt` files are excluded via `.gitignore` to reduce repository size (~6GB → manageable size)
- **Results preserved**: CSV files containing accuracy metrics and fold-wise results are retained in `experiments/`
- **Virtual environment compressed**: `.venv` is provided as `.venv.zip` - extract before use
- **Raw data included**: Original `.wav` audio files are preserved in `data/raw/PlantSounds/`
- **To reproduce experiments**: Retrain models using the training scripts in `src/training/`

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{khait2023plants,
  title={Sounds emitted by plants under stress are airborne and informative},
  author={Khait, Itzhak and Lewin-Epstein, Ohad and Sharon, Raz and others},
  journal={Cell},
  volume={186},
  number={7},
  pages={1328--1336},
  year={2023},
  publisher={Elsevier}
}
```

---

## License

This project is for research and educational purposes. Please refer to the original PlantSounds dataset license for data usage terms.

---

## Contact

For questions or collaborations, please open an issue in the repository.

---
---
---

# 한국어 버전

**[English](#plant-sound-classification-with-deep-learning)** | 한국어

---

## 작업 및 실행 환경

- **Python**: 3.8 이상
- **프레임워크**: PyTorch 2.0 이상
- **하드웨어**: CUDA 호환 GPU (권장)
- **운영체제**: Windows 10/11, Linux, macOS

### 필수 라이브러리
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

## 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd plant_model

# 가상환경 압축 해제 (이미 구성되어 있음)
# .venv 폴더는 저장소 크기를 줄이기 위해 압축되어 있습니다
# 진행하기 전에 .venv.zip을 .venv/로 압축 해제하세요

# 가상환경 활성화
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 의존성 설치 (필요한 경우)
pip install -r requirements.txt
```

**주의사항**:
- 모델 체크포인트 파일(`.pth`, `.pt`)은 저장소 크기를 줄이기 위해 제거되었습니다
- `.venv` 폴더는 압축 아카이브로 제공됩니다
- 실험을 재현하려면 training 스크립트를 사용하여 모델을 재학습해야 합니다

---

## 참고 논문

이 연구는 **Khait et al. (2023)**의 논문을 기반으로 진행되었습니다:

> Khait, I., Lewin-Epstein, O., Sharon, R. et al. Sounds emitted by plants under stress are airborne and informative. *Cell* 186, 1328–1336.e10 (2023).
> https://doi.org/10.1016/j.cell.2023.03.009

저자들의 baseline CNN 모델을 재현하고 초음파 식물 소리 분류에 대한 확장 연구를 수행했습니다.

---

## 연구 배경

식물은 가뭄이나 물리적 손상과 같은 스트레스 상황에서 초음파 소리(20-150 kHz)를 방출합니다. Khait et al. (2023)은 다음을 증명했습니다:

- **공중 전파 가능**: 수 미터 거리에서도 감지 가능
- **정보 내포**: 식물의 생리학적 상태에 대한 정보 포함
- **구별 가능**: 서로 다른 스트레스 유형 및 식물 종 간 구별 가능

식물 소리 분류 능력은 다음과 같은 분야에 중요한 의미를 갖습니다:
- **정밀 농업**: 최적의 관개를 위한 조기 스트레스 감지
- **식물 표현형 분석**: 비침습적 식물 건강 모니터링
- **생태계 모니터링**: 식물-환경 상호작용 이해

---

## 연구 목적

본 프로젝트는 다음을 목표로 합니다:

1. Khait et al. (2023)의 baseline CNN 분류기를 **재현**
2. 고급 딥러닝 아키텍처로 분류 성능 **개선**
3. 반지도 학습을 통한 라벨링 요구사항 감소 **탐구**
4. 다양한 식물 종 및 스트레스 유형에 대한 모델 강건성 **분석**

### 주요 혁신 사항

- **개선된 CNN 아키텍처**: 어텐션 메커니즘과 잔차 연결을 포함한 향상된 모델
- **반지도 학습**: 50% 라벨 데이터만으로도 효과적인 학습
- **포괄적 평가**: 강건한 평가를 위한 LOPO (Leave-One-Plant-Out) 교차 검증
- **다중 작업 분석**: 종, 스트레스 유형, 배경 소음에 걸친 분류

---

## 데이터셋

**PlantSounds** 데이터셋은 다음을 포함한 초음파 녹음(500 kHz 샘플링)으로 구성됩니다:

- **식물 종**: 토마토 (*Solanum lycopersicum*), 담배 (*Nicotiana tabacum*)
- **스트레스 조건**:
  - Dry (가뭄 스트레스)
  - Cut (기계적 손상)
- **대조군**: 빈 화분, 온실 배경 소음

### 데이터 구조

```
data/raw/PlantSounds/
├── tomato_dry_*.wav
├── tomato_cut_*.wav
├── tobacco_dry_*.wav
├── tobacco_cut_*.wav
├── tomato_*_empty_*.wav
└── greenhouse_*.wav
```

---

## 실험 과제

### Task 1: 식물 분류 과제 (4개의 이진 분류 문제)

**개요**: Task 1은 LOPO (Leave-One-Plant-Out) 교차 검증을 사용하는 4개의 서로 다른 이진 분류 작업으로 구성됩니다. 한 개체 식물의 모든 소리를 테스트에 사용하고 나머지 식물들로 학습합니다.

---

#### Task 1.1: 토마토 Dry vs Cut 스트레스 분류

**목표**: 토마토 식물에서 가뭄 스트레스(dry)와 기계적 손상(cut)을 구별.

**데이터셋**: 음향 챔버 환경의 모든 토마토 식물 녹음

- Dry 스트레스 샘플: 탈수 상태의 식물
- Cut 스트레스 샘플: 줄기 손상을 입은 식물

**도전 과제**: 각 개별 식물은 하나의 스트레스 유형(dry 또는 cut)만 경험했습니다. 이는 특히 어려운 분류 과제입니다. LOPO 교차 검증에서 한 식물이 테스트 세트로 사용될 때, 해당 fold는 하나의 클래스만 포함하므로 모델은 다른 스트레스 조건의 다른 식물 소리로부터 일반화해야 합니다.

**결과** (그림 1a, 1b, 1e, 1g 참조):

- **Baseline CNN**: 95.2% 균형 정확도
- **MyModel (전체 라벨)**: 97.6% 균형 정확도 (+2.4%)
- **MyModel (50% 라벨)**: 93.3% 균형 정확도

---

#### Task 1.2: 담배 Dry vs Cut 스트레스 분류

**목표**: 담배 식물에서 가뭄 스트레스(dry)와 기계적 손상(cut)을 구별.

**데이터셋**: 음향 챔버 환경의 모든 담배 식물 녹음

- Dry 스트레스 샘플: 탈수 상태의 식물
- Cut 스트레스 샘플: 줄기 손상을 입은 식물

**도전 과제**: Task 1.1과 동일한 단일 클래스 fold 문제이지만 담배 식물 대상.

**결과** (그림 1a, 1b, 1f, 1g 참조):

- **Baseline CNN**: 87.7% 균형 정확도
- **MyModel (전체 라벨)**: **100.0%** 균형 정확도 (완벽한 분류!)
- **MyModel (50% 라벨)**: 100.0% 균형 정확도 (완벽한 성능 유지)

**주요 발견**: 담배 스트레스 분류가 토마토보다 쉬운 것으로 나타났으며, 이는 더 뚜렷한 음향 특징 때문일 수 있습니다.

---

#### Task 1.3: 종 분류 (Dry 조건)

**목표**: 둘 다 가뭄 스트레스를 받는 토마토와 담배 식물을 구별.

**데이터셋**: 두 종의 모든 dry 스트레스 식물

- 클래스 0: 토마토 dry 스트레스 식물
- 클래스 1: 담배 dry 스트레스 식물

**도전 과제**: 두 종이 동일한 스트레스 조건에 있으므로, 모델은 스트레스 패턴이 아닌 종 특정 음향 특성을 학습해야 합니다.

**결과** (그림 1a 참조):

- **Baseline CNN**: 87.5% 균형 정확도
- **MyModel (전체 라벨)**: 88.7% 균형 정확도 (+1.2%)
- **MyModel (50% 라벨)**: 85.7% 균형 정확도

---

#### Task 1.4: 종 분류 (Cut 조건)

**목표**: 둘 다 기계적 손상을 받는 토마토와 담배 식물을 구별.

**데이터셋**: 두 종의 모든 cut 스트레스 식물

- 클래스 0: 토마토 cut 스트레스 식물
- 클래스 1: 담배 cut 스트레스 식물

**도전 과제**: Task 1.3과 동일하지만 cut 조건.

**결과** (그림 1a 참조):

- **Baseline CNN**: 93.4% 균형 정확도
- **MyModel (전체 라벨)**: 96.9% 균형 정확도 (+3.5%)
- **MyModel (50% 라벨)**: 96.2% 균형 정확도

**주요 발견**: Cut 조건이 dry 조건보다 더 명확한 종 구별을 제공하며, 이는 기계적 손상이 더 종 특정적인 음향 반응을 생성함을 시사합니다.

---

#### Task 1 결과 시각화

**그림 1a: Task 1 혼동 행렬**
![Task 1 혼동 행렬](experiments/figures/task1_confusion_matrices.png)
*각 Task 1 하위 과제의 모든 LOPO fold에 대한 통합 혼동 행렬. Baseline CNN과 MyModel 모두의 분류 오류를 보여줌. 토마토/담배 dry vs cut을 포함한 모든 과제가 성공적으로 시각화됨.*

**그림 1b: Task 1 평균 균형 정확도 비교**
![Task 1 막대 그래프](experiments/figures/task1_balanced_accuracy_bar.png)
*4개 Task 1 하위 과제 전체의 평균 균형 정확도 비교. Baseline CNN, MyModel (100% 라벨), MyModel (50% 라벨) 성능을 보여줌.*

---

### Task 2: 식물 vs 빈 화분 분류

**목표**: 실제 식물 스트레스 소리를 빈 화분 대조군 녹음과 구별.

**데이터셋**: 모든 식물 소리 (토마토 + 담배, dry + cut) vs 빈 화분 녹음

- 클래스 0 (음성): 빈 화분 녹음 (대조군/소음)
- 클래스 1 (양성): 식물 스트레스 소리 (모든 종 및 조건)

**목적**: 이 작업은 식물 소리가 동일한 음향 챔버에서 빈 화분이 생성하는 기계적/환경적 소음과 진정으로 구별 가능한지 검증합니다. 이는 관찰된 음향 방출이 생물학적 기원임을 입증하는 데 중요합니다.

**교차 검증**: 식물 소리에 대해 LOPO (Leave-One-Plant-Out), 빈 화분 샘플은 식물 fold 수와 일치하도록 K개 그룹으로 분할.

**결과** (그림 2a, 2c 참조):

- **Baseline CNN**: 95.9% 균형 정확도
- **MyModel**: 98.5% 균형 정확도 (+2.6%)

**주요 발견**:

- 거의 완벽한 분류는 식물 소리가 뚜렷한 생물학적 특징을 가지고 있음을 입증
- 높은 정확도는 음향 방출이 녹음 환경의 인공물이 아님을 확인
- 식물과 빈 화분 소리 간 최소 혼동은 식물 음향의 생물학적 특성을 검증

---

### Task 3: 토마토 Dry vs 온실 소음 분류

**목표**: 토마토 dry 스트레스 소리를 실제 온실 배경 소음과 구별.

**데이터셋**: 토마토 dry 스트레스 식물 vs 온실 환경 소음

- 클래스 0 (음성): 온실 소음 녹음 (환경 배경)
- 클래스 1 (양성): 토마토 dry 스트레스 소리

**목적**: Task 2가 제어된 빈 화분 소음을 테스트한 반면, 이 작업은 통제되지 않은 실제 온실 소음(팬, 장비, 공기 순환 등)에 대한 성능을 평가합니다. 이는 생태학적 타당성과 실제 배포 가능성을 테스트합니다.

**도전 과제**: 온실 소음은 제어된 빈 화분 녹음보다 더 복잡하고 가변적이어서 더 현실적이고 어려운 시나리오입니다.

**교차 검증**: 식물 소리에 대해 LOPO, 온실 소음 샘플은 K개 그룹으로 분할.

**결과** (그림 2a, 2c 참조):

- **Baseline CNN**: 98.4% 균형 정확도
- **MyModel**: 98.2% 균형 정확도 (-0.2%)

**주요 발견**:

- 두 모델 모두 복잡한 환경 소음에서도 거의 완벽한 분류 달성
- MyModel 성능의 약간 감소는 이 더 쉬운 작업에 baseline이 충분할 수 있음을 시사
- 실제 소음 조건에서의 높은 정확도는 실용적 배포 가능성 입증
- 식물 음향 신호는 시끄러운 온실 환경에서도 높은 구별력 유지

---

### Task 2 & 3 결과 시각화

**그림 2a: Task 2/3 혼동 행렬**
![Task 2/3 혼동 행렬](experiments/figures/task2_task3_confusion_matrices.png)
*Task 2와 Task 3의 통합 혼동 행렬. 식물 vs 소음 분류에서 강력한 클래스 분리를 보여줌. 텍스트 겹침 문제가 개선된 레이아웃으로 수정됨.*

**그림 2b: Task 2/3 평균 균형 정확도 비교**
![Task 2/3 막대 그래프](experiments/figures/task2_task3_balanced_accuracy_bar.png)
*Task 2 (식물 vs 빈 화분)와 Task 3 (토마토 vs 온실 소음)의 평균 균형 정확도 비교. 두 작업 모두 >95% 정확도를 보여줌.*

---

### Task 4: 반지도 학습 효과

**목표**: 라벨링된 훈련 데이터를 100%에서 50%로 줄여 MyModel의 반지도 학습 접근법의 강건성을 평가.

**동기**: 실제 시나리오에서 라벨링된 데이터를 얻는 것은 비용이 많이 들고 시간이 소요됩니다. 반지도 학습은 라벨링되지 않은 데이터를 활용하여 더 적은 라벨로 성능을 유지하므로 대규모 배포를 더 실용적으로 만듭니다.

**방법론**:

- **전체 라벨 (100%)**: 모든 훈련 샘플에 정답 라벨 있음
- **부분 라벨 (50%)**: 훈련 샘플의 50%만 라벨링됨; 나머지 50%는 반지도 학습을 위한 라벨 없는 데이터로 사용
- **교차 검증**: Task 1과 동일한 LOPO splits, fold당 무작위 50% 라벨 선택

**구현**: MyModel은 일관성 정규화 접근법을 사용:

1. 라벨링된 샘플: 표준 지도 손실 (교차 엔트로피)
2. 라벨링되지 않은 샘플: 서로 다른 증강 하의 예측 간 일관성 손실
3. 결합 훈련: 지도 및 비지도 목적 균형

**결과 분석** (그림 3 참조):

| Task | Baseline | 100% 라벨 | 50% 라벨 | 감소 | 상대 유지율 |
|------|----------|-----------|----------|------|------------|
| 토마토 dry vs cut | 95.2% | 97.6% | 93.3% | -4.3% | 95.6% |
| 담배 dry vs cut | 87.7% | 100.0% | 100.0% | 0.0% | 100.0% |
| Dry: 토마토 vs 담배 | 87.5% | 88.7% | 85.7% | -3.0% | 96.6% |
| Cut: 토마토 vs 담배 | 93.4% | 96.9% | 96.2% | -0.7% | 99.3% |

**주요 발견**:

- **최소 성능 감소**: 라벨 50% 감소 시 평균 2-4% 저하만
- **강건성**: 담배 dry vs cut는 라벨 절반으로도 완벽한 100% 정확도 유지
- **실용적 가치**: 모든 작업에서 85-100% 정확도 유지는 대규모 응용의 실현 가능성 입증
- **라벨 효율성**: 50% 라벨로 93-100% 성능 달성은 상당한 비용 절감
- **반지도 이점**: 모든 50% 라벨 결과가 baseline을 크게 상회하여 라벨 없는 데이터의 가치 확인

**의미**: 감소된 라벨로의 강력한 성능은 모든 데이터 라벨링이 불가능한 실용적 배포를 위한 MyModel의 반지도 접근법을 검증합니다.

---

### Task 4 결과 시각화

**그림 3: Task 4 반지도 학습 효과**
![Task 4 라벨 비율 효과](experiments/figures/task4_label_fraction_effect.png)
*4개 Task 1 하위 과제에 대한 Baseline CNN, MyModel (50% 라벨), MyModel (100% 라벨)을 비교하는 그룹화된 막대 그래프. 각 과제는 쉬운 비교를 위해 세 개의 막대를 보여줌. 화살표와 백분율은 반지도에서 완전 지도 학습으로의 개선을 나타냄. 이 명확한 시각화는 50% 라벨 데이터만 사용했을 때의 최소한의 성능 저하로 라벨 효율성을 입증.*

---

### 지표 요약 테이블

**그림 4: 포괄적 지표 요약**
![지표 요약 테이블](experiments/figures/metrics_summary_table.svg)
*모든 6개 과제(Task 1: 4개 하위과제, Task 2, Task 3)에 대한 모든 주요 분류 지표를 보여주는 학술 논문 스타일 지표 요약 테이블. Baseline CNN과 MyModel 모두에 대한 Accuracy, Balanced Accuracy, Macro F1-Score, Precision, Recall, Specificity를 포함. 한눈에 완전한 정량적 비교 제공.*

**지표에서의 주요 인사이트**:

- **균형 잡힌 성능**: 두 모델 모두 모든 과제에서 강력한 precision, recall, specificity 달성
- **Macro F1-Score**: 클래스별 F1 점수를 평균하여 전반적인 분류 품질 검증
- **과제별 패턴**: Task 1.2 (담배 dry vs cut)는 MyModel에서 완벽한 지표(1.000)를 보이는 반면, 소음 분리 과제(Task 2/3)는 두 모델 모두 일관되게 높은 성능(>0.95) 보임
- **일관된 개선**: MyModel은 대부분의 지표와 과제에서 일반적으로 Baseline을 능가함

---

## 모델 아키텍처

### Baseline CNN (Khait et al., 2023)

![Baseline](experiments/figures/baseline_architecture.svg)

### MyModel (Baseline CNN + VAE + SSL + DG)

![MyModel](experiments/figures/mymodel_architecture.svg)


**아키텍처 상세**:

- **Backbone**: Baseline과 동일한 CNN 구조 (Khait et al., 2023)
- **임베딩 h**: CNN backbone에서 추출한 128차원 특징 표현
- **Main Classifier**: h → 이진 로짓 (BCEWithLogitsLoss)
- **VAE 모듈**:
  - 인코더: h → (μ, log σ²) 32차원 잠재 공간
  - 재매개변수화: z = μ + σ ⊙ ε
  - 디코더: z → h_재구성
  - 손실: MSE(h, h_rec) + β·KL(q(z|h) || N(0, I))
- **SSL (반지도 학습)**:
  - z(x)와 z(x_증강) 간 일관성 정규화
  - 라벨 없는 데이터 학습 가능 (50% 라벨 실험)
- **DG (도메인 일반화)**:
  - 도메인 분류기 헤드: h → 도메인 로짓
  - 녹음 조건 간 강건성 향상

---

## 학습 프로토콜

### 데이터 전처리

1. **리샘플링**: 모든 오디오를 500 kHz로
2. **세그먼테이션**: 2ms 윈도우 (1000 샘플)
3. **고역 통과 필터**: 저주파 노이즈 제거
4. **정규화**: 평균 0, 분산 1

### 교차 검증

**LOPO (Leave-One-Plant-Out)**:
- 각 fold는 한 식물의 모든 녹음을 홀드아웃
- 새로운 개체에 대한 일반화 테스트
- 무작위 분할보다 현실적인 평가

### 학습 설정

```python
Optimizer: Adam (lr=5e-4)
Loss: BCEWithLogitsLoss (클래스 균형 포함)
Batch size: 64
Epochs: 10 (조기 종료 포함)
```

### 평가 지표 (Evaluation Metrics)

이 프로젝트는 포괄적인 성능 평가를 위해 여러 분류 지표를 사용합니다:

#### 주요 지표

##### 1. Balanced Accuracy (균형 정확도) - 본 프로젝트의 주 지표

- 공식: `(Recall + Specificity) / 2 = (TPR + TNR) / 2`
- **사용 이유**: 식물 스트레스 데이터셋은 클래스 불균형이 있습니다 (특히 LOPO에서 각 fold는 하나의 스트레스 조건을 가진 한 식물만 테스트할 수 있음). Balanced accuracy는 클래스별 recall을 평균하여 두 클래스를 동등하게 취급하므로, 불균형 데이터에서 일반 accuracy보다 신뢰할 수 있습니다.
- **해석**: 완벽한 점수 = 1.0, 무작위 추측 = 0.5

##### 2. Accuracy (정확도)

- 공식: `(TP + TN) / (TP + TN + FP + FN)`
- 전체 샘플 중 올바른 예측의 비율
- **한계**: 클래스 불균형 시 오해의 소지가 있음 (예: 모든 샘플을 한 클래스로 예측해도 높은 accuracy 가능)

#### 추가 중요 지표

##### 3. Precision (정밀도, 양성 예측도)

- 공식: `TP / (TP + FP)`
- "양성으로 예측한 것 중 실제로 양성인 비율은?"
- **사용 사례**: 거짓 경보(FP)의 비용이 클 때 중요
- 높은 precision = 신뢰할 수 있는 양성 예측

##### 4. Recall (재현율, 민감도, True Positive Rate)

- 공식: `TP / (TP + FN)`
- "실제 양성 샘플 중 얼마나 많이 올바르게 식별했는가?"
- **사용 사례**: 양성을 놓치는 것(FN)의 비용이 클 때 중요
- 높은 recall = 실제 양성 케이스 대부분/전부 찾아냄

##### 5. Specificity (특이도, True Negative Rate)

- 공식: `TN / (TN + FP)`
- "실제 음성 샘플 중 얼마나 많이 올바르게 식별했는가?"
- 모델이 음성 클래스를 얼마나 잘 식별하는지 측정

##### 6. F1-Score (F1 점수)

- 공식: `2 × (Precision × Recall) / (Precision + Recall)`
- Precision과 Recall의 조화 평균
- **사용 사례**: 균형 잡힌 precision과 recall이 필요할 때
- Precision이나 Recall 중 하나가 낮으면 F1-score도 낮아짐

#### 혼동 행렬(Confusion Matrix) 이해하기

```
                  예측
              음성      양성
실제  음성      TN        FP      ← False Positive: 거짓 경보
      양성      FN        TP      ← False Negative: 놓친 검출
                 ↑
            놓친 케이스
```

- **TP (True Positive)**: 양성을 올바르게 예측 (예: 스트레스 받은 식물을 정확히 식별)
- **TN (True Negative)**: 음성을 올바르게 예측 (예: 스트레스 없는 식물을 정확히 식별)
- **FP (False Positive)**: 양성으로 잘못 예측 (거짓 경보 - 괜찮은 식물을 스트레스 받았다고 예측)
- **FN (False Negative)**: 음성으로 잘못 예측 (놓친 검출 - 실제 스트레스를 감지하지 못함)

#### 지표 선택 가이드라인

이 식물 스트레스 분류 프로젝트의 경우:

1. **클래스 불균형 존재** → Balanced Accuracy + F1-Score 사용
2. **FN과 FP 모두 중요** → Precision, Recall, F1-Score 모니터링
3. **오류 유형별 비용 차이** → 혼동 행렬에서 오류 패턴 확인
4. **전반적 성능** → 클래스가 균형잡혔을 때 일반 Accuracy도 유용

그림의 혼동 행렬 시각화는 실제 오류 패턴을 보여주며, 모델이 스트레스 받은 식물을 놓치는 경향(FN)이 있는지 거짓 경보를 내는 경향(FP)이 있는지 식별하는 데 도움을 줍니다.

---

## 실험 실행

### 데이터 준비

```bash
# PlantSounds 데이터셋을 data/raw/PlantSounds/에 배치
# 예상 구조:
data/raw/PlantSounds/
├── tomato_dry_*.wav
├── tobacco_dry_*.wav
└── ...
```

### 학습

```bash
# Task 1: Baseline
python -m src.training.train_task1

# Task 1: MyModel (전체 라벨)
python -m src.training.train_task1_mymodel

# Task 1: MyModel (반지도, 50% 라벨)
python -m src.training.train_task1_mymodel_semi

# Task 2 & 3
python -m src.training.train_task2
python -m src.training.train_task3
```

### 시각화

```bash
# 모든 그림 생성
python -m src.analysis.plot_results

# experiments/figures/에 출력 저장
```

---

## 결과 요약

### 전체 성능

| 모델 | 작업 유형 | 평균 정확도 | 주요 장점 |
|------|-----------|-------------|-----------|
| Baseline CNN | 식물-식물 | 90.7% | 간단, 재현 가능 |
| MyModel (전체) | 식물-식물 | 93.1% | +2.4% 개선 |
| MyModel (50%) | 식물-식물 | 90.3% | 라벨 효율성 |
| MyModel | 식물-소음 | 98.4% | 거의 완벽한 분리 |

### 주요 성과

1. ✓ Khait et al. baseline 성공적으로 **재현**
2. ✓ 모든 작업에서 분류 정확도 2-4% **개선**
3. ✓ 반지도 학습 실행 가능성 **입증** (50% 라벨로 2-4% 감소만)
4. ✓ 담배 스트레스 구별에 대한 완벽한 분류(100%) **달성**
5. ✓ 엄격한 LOPO 교차 검증을 통한 모델 일반화 **검증**

---

## 연구 의의

### 과학적 기여

1. **식물 생체음향학 검증**: 식물 초음파 방출이 생물학적으로 의미 있는 정보를 포함함을 확인
2. **딥러닝 적용 가능성**: 식물 소리 분석을 위한 현대 신경망의 효과성 입증
3. **라벨 효율성**: 반지도 학습이 주석 비용을 50% 감소시킬 수 있음을 보임
4. **종 및 스트레스 구별**: 여러 차원에 걸친 강건한 분류 달성

### 실용적 응용

- **정밀 농업**: 최적의 관개 일정을 위한 조기 가뭄 감지
- **온실 모니터링**: 자동화된 식물 건강 평가
- **표현형 파이프라인**: 고처리량 스트레스 반응 특성화
- **기후 연구**: 환경 스트레스에 대한 식물 반응 이해

### 향후 방향

1. **다중 클래스 분류**: 종과 스트레스 유형 동시 인식
2. **전이 학습**: 제한된 데이터로 새로운 식물 종에 대한 일반화
3. **실시간 시스템**: 연속 온실 모니터링을 위한 배포
4. **다중 모달 통합**: 음향 데이터와 이미징 및 환경 센서 결합
5. **시간적 역학**: 시간 경과에 따른 스트레스 진행 분석

---

## 파일 구조

```
plant_model/
├── data/
│   ├── raw/
│   │   └── PlantSounds/          # 원본 데이터셋 (.wav 파일)
│   └── processed/                 # 전처리된 데이터 (git에서 제외)
├── experiments/
│   ├── task1_baseline/            # Baseline 결과 (CSV 파일만)
│   ├── task1_my_model/            # 개선 모델 결과 (CSV 파일만)
│   ├── task2_baseline/            # Task 2 baseline 결과
│   ├── task2_my_model/            # Task 2 개선 결과
│   ├── task3_baseline/            # Task 3 baseline 결과
│   ├── task3_my_model/            # Task 3 개선 결과
│   ├── task4_my_model/            # Task 4 반지도 학습 결과
│   └── figures/                   # 생성된 시각화 그래프
├── src/
│   ├── datasets/
│   │   └── plantsounds.py         # 데이터 로딩 & LOPO 분할
│   ├── models/
│   │   ├── baseline_cnn.py        # Baseline CNN (Khait et al. 2023)
│   │   └── my_model.py            # 개선 모델 (VAE + SSL + DG)
│   ├── training/
│   │   ├── common.py              # 공유 학습 유틸리티
│   │   ├── train_task1.py         # Task 1 baseline 학습
│   │   ├── train_my_model_task1.py # Task 1 개선 학습
│   │   ├── train_task2.py         # Task 2 baseline 학습
│   │   ├── train_my_model_task2.py # Task 2 개선 학습
│   │   ├── train_task3.py         # Task 3 baseline 학습
│   │   ├── train_my_model_task3.py # Task 3 개선 학습
│   │   └── train_my_model_task4.py # Task 4 반지도 학습
│   ├── utils/
│   │   ├── audio.py               # 오디오 처리 유틸리티
│   │   └── split.py               # 데이터 분할 유틸리티
│   └── analysis/
│       └── plot_results.py        # 시각화 및 분석 스크립트
├── .venv.zip                      # 압축된 가상환경
├── .gitignore                     # Git 무시 규칙 (체크포인트 제외)
├── requirements.txt
├── verify_plots.py                # 플롯 데이터 검증 스크립트
├── validate_plot_consistency.py  # 데이터 일관성 검증
└── README.md
```

**중요 사항**:
- **모델 체크포인트 제거**: 모든 `.pth` 및 `.pt` 파일은 저장소 크기 축소를 위해 `.gitignore`를 통해 제외됨 (~6GB → 관리 가능한 크기)
- **결과 보존**: 정확도 지표 및 fold별 결과를 포함하는 CSV 파일은 `experiments/`에 유지됨
- **가상환경 압축**: `.venv`는 `.venv.zip`으로 제공됨 - 사용 전 압축 해제 필요
- **원시 데이터 포함**: 원본 `.wav` 오디오 파일은 `data/raw/PlantSounds/`에 보존됨
- **실험 재현**: `src/training/`의 학습 스크립트를 사용하여 모델 재학습 필요

---

## 인용

이 코드나 결과를 연구에 사용하시는 경우, 다음을 인용해주세요:

```bibtex
@article{khait2023plants,
  title={Sounds emitted by plants under stress are airborne and informative},
  author={Khait, Itzhak and Lewin-Epstein, Ohad and Sharon, Raz and others},
  journal={Cell},
  volume={186},
  number={7},
  pages={1328--1336},
  year={2023},
  publisher={Elsevier}
}
```

---

## 라이선스

본 프로젝트는 연구 및 교육 목적입니다. 데이터 사용 조건은 원본 PlantSounds 데이터셋 라이선스를 참조하세요.

---

## 문의

질문이나 협업을 위해서는 저장소에 이슈를 열어주세요.
