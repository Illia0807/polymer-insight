"Polymer Insight" Project"

# Polymer Insight: A Data-Driven Approach to Polymer Science

## Project Overview

**Polymer Insight** is an interactive web application designed to predict and explain the physical properties of polymers. The project addresses a common challenge in materials science: the high cost and time required for experimental data collection. By leveraging machine learning and a knowledge-based system, this application provides a powerful tool for researchers, students, and engineers to quickly and accurately estimate key polymer properties from their chemical structure.

## Key Features

- **Property Prediction**: Predicts four fundamental polymer properties (Fractional Free Volume, Density, Crystallization Temperature, and Radius of Gyration) from a simple SMILES string.
- **Explainable AI (XAI) via RAG**: The core of the application is a Retrieval-Augmented Generation (RAG) system that provides context-aware explanations for each prediction. This helps users understand the "why" behind the results.
- **Interactive Web Interface**: A user-friendly interface built with Streamlit allows for easy input and real-time visualization of results.

## Technologies Used

The project is built on a robust and modern data science stack:

- **Web Interface**: **Streamlit**
- **Machine Learning**: **LightGBM** (for its speed and high accuracy)
- **Molecular Descriptors**: **RDKit** (for generating features from chemical structures)
- **Vector Database**: **ChromaDB** (for efficient knowledge retrieval in the RAG system)
- **Core Libraries**: pandas, numpy, scikit-learn, joblib

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Project Setup

Clone the repository and navigate to the project directory.

```bash
git clone <your-repo-link>
cd polymer-insight

2. Install Dependencies

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install all required packages
pip install -r requirements.txt


3. Prepare Data and Models
# Train the models and save them to the 'models' directory
python train_and_save.py

# Create the ChromaDB database with polymer facts
python create_db.py

4. Run the Application
Start the Streamlit web server to launch the app.
streamlit run app.py

Demonstration
Test the Prediction Model
Enter the SMILES string for a common polymer like Polyethylene (C(C)C):

*CC*

The app will display the predicted properties.

Test the RAG System
Use the chat interface to ask questions about the predictions or project.

"What is FFV?"

"How does a high Tc affect a polymer?"

"What are the ethical concerns of this project?

Future Development
Integrate an LLM: Enhance the RAG system by using a local or API-based LLM (e.g., Llama 3) to synthesize more coherent and natural-sounding explanations from the retrieved facts.

Expand the Knowledge Base: Add more facts from scientific articles to make the RAG system more comprehensive.

Explore New Models: Test advanced models like GNNs for predicting complex properties like Tc, which is currently a limitation of the current model.

Ethical Considerations
This project is a tool for estimation, not a replacement for experimental data. The models, while accurate, have inherent limitations and were trained on a finite dataset. The predictions should be verified in a lab setting, and users should be aware of potential data biases. The RAG system's explainability feature aims to address these concerns by promoting transparency and informed decision-making.