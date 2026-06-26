Project Architecture and Status Overview:

Directory Structure and Core Components
Model Architectures (codebase/): This directory houses the model pipeline classes, which are designed to support modular model implementations, including architectures like iTransformer.
Each pipeline class standardizes workflows by exposing four primary methods:

preprocess_splits (for training)
preprocess_inference
train
inference

Data Management (data/): Contains the underlying datasets used for training and evaluation.

preprocess/: Contains utility and helper functions utilized across the pipeline classes to ensure consistent data transformation.

Development Environment
Runtime: Python 3.12.3
Environment: Virtual environment (venv) deployed on a Linux infrastructure.
Execution and Verification Status
Testing Workflow: Functional verification of the pipeline classes is currently conducted via execution scripts in main.py.
Testing Coverage: While the training and preprocessing pipelines are functional, the inference capabilities have not yet undergone rigorou testing or validation.
