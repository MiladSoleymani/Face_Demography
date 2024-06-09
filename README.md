# Face_Demography

This repository contains a project for demographic analysis using facial recognition and machine learning techniques.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
Face_Demography aims to analyze demographic attributes such as age, gender, and ethnicity from facial images. The project leverages deep learning models to perform these predictions accurately.

## Requirements
- Python 3.x
- Required packages listed in `requirements.txt`

## Installation
To set up the project, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/MiladSoleymani/Face_Demography.git
    cd Face_Demography
    ```
2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage
To run the demographic analysis, use the following command:
```bash
python script/predict.py --image_path path/to/image.jpg
```
For more detailed usage, refer to the scripts provided in the `script` directory.

## Repository Structure
- `dataset`: Contains data preprocessing scripts and datasets.
- `models`: Includes the trained models and scripts for training.
- `script`: Contains scripts for running predictions and evaluations.
- `utils`: Utility functions and helper scripts.
- `.gitignore`: Specifies files to ignore in the repository.
- `requirements.txt`: Lists the dependencies required for the project.
- `README.md`: Project documentation.

## Contributing
We welcome contributions to improve Face_Demography. Please fork the repository, create a new branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
