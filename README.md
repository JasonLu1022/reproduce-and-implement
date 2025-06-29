# LLM Memorization and Reasoning Detection Framework

A unified implementation framework for detecting memorization vs. reasoning capabilities in Large Language Models, based on five foundational research papers.

## 📋 Overview

This project provides standardized implementations of state-of-the-art methods for distinguishing between memorization and reasoning in LLMs. It consolidates five key research approaches into a modular, extensible framework for cross-paper and cross-dataset evaluations.

## 🎯 Target Papers

| Paper | Method | Status | Code Availability |
|-------|--------|--------|-------------------|
| **Xie et al. (2024)** | Knights and Knaves Logic Perturbation | ✅ Implemented | [GitHub](https://github.com/AlphaPav/mem-kk-logic) |
| **Salido et al. (2025)** | None of the Others (NOTO) | ✅ Implemented | No official repo |
| **Wu et al. (2023)** | Counterfactual Evaluation | ✅ Implemented | [GitHub](https://github.com/ZhaofengWu/counterfactual-evaluation) |
| **Jin et al. (2024)** | Memory-Reasoning Disentanglement | ✅ Implemented | [GitHub](https://github.com/MingyuJ666/Disentangling-Memory-and-Reasoning) |
| **Hong et al. (2025)** | Linear Reasoning Features (LiReF) | 🔧 Environment Ready | [GitHub](https://github.com/yihuaihong/Linear_Reasoning_Features) |

## 🏗️ Project Structure

```
llm-memorization-detection/
├── methods/                    # Core method implementations
│   ├── xie_2024/              # Knights and Knaves logic perturbation
│   ├── salido_2025/            # NOTO transformation method
│   ├── wu_2023/               # Counterfactual evaluation
│   ├── jin_2024/              # Memory-reasoning disentanglement
│   └── hong_2025/             # Linear reasoning features
├── datasets/                   # Standardized datasets
│   ├── knights_knaves/        # Logic puzzle datasets
│   ├── counterfactual/        # Counterfactual arithmetic tasks
│   ├── memory_reasoning/      # Dual-task datasets
│   └── reasoning_features/    # Activation analysis datasets
├── evaluation/                 # Unified evaluation framework
│   ├── metrics.py             # Common evaluation metrics
│   ├── evaluator.py           # Main evaluation engine
│   └── comparison.py          # Cross-method comparison
├── models/                     # Model interfaces
│   ├── api_models.py          # OpenAI, Anthropic, etc.
│   ├── local_models.py        # Local LLM interfaces
│   └── base_model.py          # Abstract base class
├── utils/                      # Shared utilities
│   ├── data_processing.py     # Data preprocessing utilities
│   ├── visualization.py       # Result visualization
│   └── config.py              # Configuration management
├── experiments/                # Experiment scripts
│   ├── run_single_method.py   # Single method evaluation
│   ├── run_comparison.py      # Cross-method comparison
│   └── run_analysis.py        # Result analysis
├── docs/                       # Documentation
│   ├── implementation_guide.md
│   ├── method_comparison.md
│   └── reproduction_notes.md
└── results/                    # Experiment results
    ├── individual_results/     # Per-method results
    ├── comparisons/           # Cross-method comparisons
    └── analysis/              # Analysis reports
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-memorization-detection

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (for API models)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Basic Usage

```python
from evaluation.evaluator import UnifiedEvaluator
from models.api_models import OpenAIModel

# Initialize evaluator
evaluator = UnifiedEvaluator()

# Run single method evaluation
results = evaluator.evaluate_method(
    method="xie_2024",
    model="gpt-4",
    dataset="knights_knaves"
)

# Run cross-method comparison
comparison = evaluator.compare_methods(
    methods=["xie_2024", "salido_2025", "wu_2023"],
    model="gpt-4"
)
```

### Command Line Interface

```bash
# Evaluate a single method
python experiments/run_single_method.py --method xie_2024 --model gpt-4

# Run cross-method comparison
python experiments/run_comparison.py --models gpt-4 claude-3 --methods all

# Generate analysis report
python experiments/run_analysis.py --input results/ --output analysis/
```

## 📊 Available Methods

### 1. Xie et al. (2024) - Knights and Knaves Logic Perturbation
- **Core Idea**: Tests model robustness through logic puzzle perturbations
- **Perturbation Types**: Role reversal, uncommon names, statement reordering
- **Key Metric**: Perturbation sensitivity ranking

### 2. Salido et al. (2025) - NOTO Transformation
- **Core Idea**: Replaces correct answers with "None of the other answers"
- **Applicability**: General MCQ datasets
- **Key Metric**: Performance drop on transformed questions

### 3. Wu et al. (2023) - Counterfactual Evaluation
- **Core Idea**: Tests on out-of-distribution variants (e.g., base-11 arithmetic)
- **Datasets**: Arithmetic tasks in different bases
- **Key Metric**: Performance degradation on counterfactual tasks

### 4. Jin et al. (2024) - Memory-Reasoning Disentanglement
- **Core Idea**: Dual-task design with step type prediction
- **Special Tokens**: `<memory_i>`, `<reason_i>`, `<answer_i>`
- **Key Metric**: Memory vs reasoning step ratio alignment

### 5. Hong et al. (2025) - Linear Reasoning Features
- **Core Idea**: Identifies linear direction in activation space
- **Method**: Activation space analysis and intervention
- **Key Metric**: Reasoning-memorization separability

## 🔧 Configuration

### Model Configuration

```yaml
# config/models.yaml
models:
  openai:
    gpt-4:
      api_key: ${OPENAI_API_KEY}
      temperature: 0.0
      max_tokens: 1000
  anthropic:
    claude-3-sonnet:
      api_key: ${ANTHROPIC_API_KEY}
      temperature: 0.0
  local:
    llama-7b:
      model_path: "meta-llama/Llama-2-7b-chat-hf"
      device: "cuda"
```

### Experiment Configuration

```yaml
# config/experiments.yaml
experiments:
  default:
    batch_size: 32
    max_samples: 1000
    random_seed: 42
  comparison:
    methods: ["xie_2024", "salido_2025", "wu_2023"]
    models: ["gpt-4", "claude-3-sonnet"]
    datasets: ["knights_knaves", "arithmetic", "memory_reasoning"]
```

## 📈 Results and Analysis

### Individual Method Results

Each method produces standardized output including:
- Accuracy metrics (original vs. perturbed/transformed)
- Performance degradation analysis
- Statistical significance tests
- Visualization plots

### Cross-Method Comparison

The framework enables:
- Direct comparison of different approaches
- Correlation analysis between methods
- Identification of complementary strengths
- Recommendations for method selection

## 🧪 Reproducibility

### Environment Setup

```bash
# Create conda environment
conda create -n llm-detection python=3.10
conda activate llm-detection

# Install exact versions
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Reproducing Results

```bash
# Reproduce all experiments
python experiments/reproduce_all.py

# Verify specific method
python experiments/verify_method.py --method xie_2024
```

## 📚 Documentation

- **[Implementation Guide](docs/implementation_guide.md)**: Detailed setup and usage instructions
- **[Method Comparison](docs/method_comparison.md)**: Comparative analysis of all methods
- **[Reproduction Notes](docs/reproduction_notes.md)**: Implementation challenges and solutions
- **[API Reference](docs/api_reference.md)**: Complete API documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-method`)
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper authors for their foundational research
- Open source community for tools and libraries
- Contributors to this unified framework

## 📞 Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Join our discussion forum

---

**Status**: 🚀 Production Ready | **Version**: 1.0.0 | **Last Updated**: January 2025 