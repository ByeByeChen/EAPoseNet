# EAPoseNet:Efficient Animal Pose Network in Limited Computing Power Scenarios
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

This repository contains the implementation of a lightweight animal pose estimation network based on the mmpose framework. Our goal is to provide an efficient and accurate solution for animal pose estimation while maintaining a lightweight model suitable for deployment on resource-constrained devices.

## Table of Contents

- [Abstract](#Abstract)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Results](#results)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Abstract

Accurate and efficient estimation of animal poses is crucial across various fields, such as animal behavior research, pharmaceutical studies, and biomimetic robotics. However, pose estimation typically relies on complex models to achieve higher detection precision. When computational resources are limited, achieving real-time detection with high parametric calculations becomes challenging. Therefore, this paper proposes a lightweight animal pose estimation model (EAPoseNet) to address the need for real-time detection in resource constrained scenarios.

## Installation

To install the necessary dependencies, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/lightweight-animal-pose-estimation.git
    cd lightweight-animal-pose-estimation
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Install mmpose:
    ```sh
    git clone https://github.com/open-mmlab/mmpose.git
    cd mmpose
    pip install -v -e .
    cd ..
    ```

## Usage

### Training

To train the model, use the training script:

```sh
python tools/train.py configs/your_config_file.py
```

### Inference

To perform inference using a trained model, use the inference script:

```sh
python tools/test.py configs/your_config_file.py checkpoints/your_checkpoint.pth
```

## Results

We have evaluated our model on various animal datasets and achieved competitive results. Below are some qualitative examples:

![AP10K](ap10k.png)

For detailed quantitative results, please refer to our [paper](link_to_your_paper).

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your-paper,
  title={Lightweight Animal Pose Estimation Network},
  author={Your Name and Others},
  journal={Journal/Conference},
  year={2023}
}
```

## Acknowledgements

We would like to thank the developers of mmpose for their powerful and flexible framework, and all contributors to open source animal pose estimation research.
