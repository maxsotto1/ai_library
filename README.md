Project Architecture and Status Overview:
\n\n
Directory Structure and Core Components\n
Model Architectures (codebase/): This directory houses the model pipeline classes, which are designed to support modular model implementations, including architectures like iTransformer.\n
Each pipeline class standardizes workflows by exposing four primary methods:\n
\n\n
preprocess_splits (for training),\n
preprocess_inference,\n
train,\n
inference\n
\n\n
Data Management (data/): Contains the underlying datasets used for training and evaluation.
\n\n
preprocess/: Contains utility and helper functions utilized across the pipeline classes to ensure consistent data transformation.
\n\n
Development Environment\n
Runtime: Python 3.12.3\n\n
Environment: Virtual environment (venv) deployed on a Linux infrastructure.\n\n
Execution and Verification Status\n\n
Testing Workflow: Functional verification of the pipeline classes is currently conducted via execution scripts in main.py.\n
Testing Coverage: While the training and preprocessing pipelines are functional, the inference capabilities have not yet undergone rigorou testing or validation.\n
