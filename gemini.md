This project aims to evaluate and fine-tune Gemma 3 models for fire and smoke detection using the D-Fire dataset.

### Key Components:

*   **Evaluation Framework (`main.py`, `src/`):**
    *   A Python-based framework for evaluating Gemma models against the D-Fire dataset.
    *   It supports multiple models (e.g., `gemma-3-4b-it`, `gemma-3n-e4b-it`) and prompt variations.
    *   The evaluation process involves:
        1.  Loading the D-Fire dataset and categorizing images into `fire`, `smoke`, `both`, or `none`.
        2.  Sending images and prompts to the Gemma models via the Gemini API.
        3.  Parsing the model's response to determine the predicted category.
        4.  Analyzing the results to calculate metrics like accuracy, F1-score, and generating reports with confusion matrices.

*   **Fine-tuning Notebooks:**
    *   `Gemma3N_4B_DFire.ipynb`: This notebook is for fine-tuning the `gemma-3n-e4b-it` model on the D-Fire dataset. It uses the `unsloth` library for efficient fine-tuning. The process includes:
        *   Loading and preprocessing the D-Fire dataset.
        *   Creating vision-text pairs for training.
        *   Fine-tuning the model with LoRA.
    *   `Gemma3N_4B_DFire_Inference.ipynb`: This notebook is for running inference with the fine-tuned `gemma-3n-e4b-it` model. It evaluates the model's performance on the D-Fire test set.
    *   `Gemma3_(4B)_Vision.ipynb`: This notebook appears to be for fine-tuning a non-3n version of the Gemma 3 vision model, likely `gemma-3-4b-it`, on a different dataset (LaTeX OCR).

### Next Steps:

Your immediate goal is to debug the fine-tuning process for the `gemma-3n-e4b-it` model using the `Gemma3N_4B_DFire.ipynb` notebook. After successful fine-tuning, you'll use the `Gemma3N_4B_DFire_Inference.ipynb` notebook to evaluate its performance.
