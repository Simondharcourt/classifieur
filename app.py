import os
import gradio as gr
import pandas as pd
import numpy as np
from litellm import OpenAI
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import torch
import traceback
import logging

# Import local modules
from classifiers import TFIDFClassifier, LLMClassifier
from utils import load_data, export_data, visualize_results, validate_results
from prompts import (
    CATEGORY_SUGGESTION_PROMPT,
    ADDITIONAL_CATEGORY_PROMPT,
    VALIDATION_ANALYSIS_PROMPT,
    CATEGORY_IMPROVEMENT_PROMPT
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Only initialize client if API key is available
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        logging.info("OpenAI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {str(e)}")

def update_api_key(api_key):
    """Update the OpenAI API key"""
    global OPENAI_API_KEY, client
    
    if not api_key:
        return "API Key cannot be empty"
    
    OPENAI_API_KEY = api_key
    
    try:
        client = OpenAI(api_key=api_key)
        # Test the connection with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return f"API Key updated and verified successfully"
    except Exception as e:
        error_msg = str(e)
        logging.error(f"API key update failed: {error_msg}")
        return f"Failed to update API Key: {error_msg}"

def process_file(file, text_columns, categories, classifier_type, show_explanations):
    """Process the uploaded file and classify text data"""
    # Initialize result_df and validation_report
    result_df = None
    validation_report = None
    
    try:
        # Load data from file
        if isinstance(file, str):
            df = load_data(file)
        else:
            df = load_data(file.name)
        
        if not text_columns:
            return None, "Please select at least one text column"
        
        # Check if all selected columns exist
        missing_columns = [col for col in text_columns if col not in df.columns]
        if missing_columns:
            return None, f"Columns not found in the file: {', '.join(missing_columns)}. Available columns: {', '.join(df.columns)}"
        
        # Combine text from selected columns
        texts = []
        for _, row in df.iterrows():
            combined_text = " ".join(str(row[col]) for col in text_columns)
            texts.append(combined_text)
        
        # Parse categories if provided
        category_list = []
        if categories:
            category_list = [cat.strip() for cat in categories.split(",")]
        
        # Select classifier based on data size and user choice
        num_texts = len(texts)
        
        # If no specific model is chosen, select the most appropriate one
        if classifier_type == "auto":
            if num_texts <= 500:
                classifier_type = "gpt4"
            elif num_texts <= 1000:
                classifier_type = "gpt35"
            elif num_texts <= 5000:
                classifier_type = "hybrid"
            else:
                classifier_type = "tfidf"
        
        # Initialize appropriate classifier
        if classifier_type == "tfidf":
            classifier = TFIDFClassifier()
            results = classifier.classify(texts, category_list)
        elif classifier_type in ["gpt35", "gpt4"]:
            if client is None:
                return None, "Erreur : Le client API n'est pas initialisé. Veuillez configurer une clé API valide dans l'onglet 'Setup'."
            model = "gpt-3.5-turbo" if classifier_type == "gpt35" else "gpt-4"
            classifier = LLMClassifier(client=client, model=model)
            results = classifier.classify(texts, category_list)
        else:  # hybrid
            if client is None:
                return None, "Erreur : Le client API n'est pas initialisé. Veuillez configurer une clé API valide dans l'onglet 'Setup'."
            # First pass with TF-IDF
            tfidf_classifier = TFIDFClassifier()
            tfidf_results = tfidf_classifier.classify(texts, category_list)
            
            # Second pass with LLM for low confidence results
            llm_classifier = LLMClassifier(client=client, model="gpt-3.5-turbo")
            results = []
            low_confidence_texts = []
            low_confidence_indices = []
            
            for i, (text, tfidf_result) in enumerate(zip(texts, tfidf_results)):
                if tfidf_result["confidence"] < 70:  # If confidence is below 70%
                    low_confidence_texts.append(text)
                    low_confidence_indices.append(i)
                    results.append(None)  # Placeholder
                else:
                    results.append(tfidf_result)
            
            if low_confidence_texts:
                llm_results = llm_classifier.classify(low_confidence_texts, category_list)
                for idx, llm_result in zip(low_confidence_indices, llm_results):
                    results[idx] = llm_result
        
        # Create results dataframe
        result_df = df.copy()
        result_df["Category"] = [r["category"] for r in results]
        result_df["Confidence"] = [r["confidence"] for r in results]
        
        if show_explanations:
            result_df["Explanation"] = [r["explanation"] for r in results]
        
        # Validate results using LLM
        validation_report = validate_results(result_df, text_columns, client)
        
        return result_df, validation_report
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        return None, f"Error: {str(e)}\n{error_traceback}"

def export_results(df, format_type):
    """Export results to a file and return the file path for download"""
    if df is None:
        return None
    
    # Create a temporary file
    import tempfile
    import os
    
    # Create a temporary directory if it doesn't exist
    temp_dir = "temp_exports"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"classification_results_{timestamp}"
    
    if format_type == "excel":
        file_path = os.path.join(temp_dir, f"{filename}.xlsx")
        df.to_excel(file_path, index=False)
    else:
        file_path = os.path.join(temp_dir, f"{filename}.csv")
        df.to_csv(file_path, index=False)
    
    return file_path

# Create Gradio interface
with gr.Blocks(title="Text Classification System") as demo:
    gr.Markdown("# Text Classification System")
    gr.Markdown("Upload your data file (Excel/CSV) and classify text using AI")
    
    with gr.Tab("Setup"):
        api_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter your API key here",
            type="password",
            value=OPENAI_API_KEY
        )
        api_key_button = gr.Button("Update API Key")
        api_key_message = gr.Textbox(label="Status", interactive=False)
        
        # Display current API status
        api_status = "API Key is set" if OPENAI_API_KEY else "No API Key found. Please set one."
        gr.Markdown(f"**Current API Status**: {api_status}")
        
        api_key_button.click(update_api_key, inputs=[api_key_input], outputs=[api_key_message])
    
    with gr.Tab("Classify Data"):
        with gr.Column():
            file_input = gr.File(label="Upload Excel/CSV File")
            
            # Variable to store available columns
            available_columns = gr.State([])
            
            # Button to load file and suggest categories
            load_categories_button = gr.Button("Load File")
            
            # Display original dataframe
            original_df = gr.Dataframe(
                label="Original Data",
                interactive=False,
                visible=False
            )

            with gr.Row():
                with gr.Column():
                    suggested_categories = gr.CheckboxGroup(
                        label="Suggested Categories",
                        choices=[],
                        value=[],
                        interactive=True,
                        visible=False
                    )

                    new_category = gr.Textbox(
                        label="Add New Category",
                        placeholder="Enter a new category name",
                        visible=False
                    )
                    with gr.Row():
                        add_category_button = gr.Button("Add Category", visible=False)
                        suggest_category_button = gr.Button("Suggest Category", visible=False)
                

                    # Original categories input (hidden)
                    categories = gr.Textbox(
                        visible=False
                    )
                
                
                with gr.Column():
                    text_column = gr.CheckboxGroup(
                        label="Select Text Columns", 
                        choices=[], 
                        interactive=True,
                        visible=False
                    )

                    classifier_type = gr.Dropdown(
                        choices=[
                            ("TF-IDF (Rapide, <1000 lignes)", "tfidf"),
                            ("LLM GPT-3.5 (Fiable, <1000 lignes)", "gpt35"),
                            ("LLM GPT-4 (Très fiable, <500 lignes)", "gpt4"),
                            ("TF-IDF + LLM (Hybride, >1000 lignes)", "hybrid")
                        ],
                        label="Modèle de classification",
                        value="tfidf",
                        visible=False
                    )
                    show_explanations = gr.Checkbox(label="Show Explanations", value=True, visible=False)
                
                    process_button = gr.Button("Process and Classify", visible=False)

        results_df = gr.Dataframe(interactive=True, visible=False)
        
        # Create containers for visualization and validation report
        with gr.Row(visible=False) as results_row:
            with gr.Column():
                visualization = gr.Plot(label="Classification Distribution")
                with gr.Row():
                    csv_download = gr.File(label="Download CSV", visible=False)
                    excel_download = gr.File(label="Download Excel", visible=False)
            with gr.Column():
                validation_output = gr.Textbox(label="Validation Report", interactive=False)
                improve_button = gr.Button("Improve Classification with Report", visible=False)

        # Function to load file and suggest categories
        def load_file_and_suggest_categories(file):
            if not file:
                return [], gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[], visible=False), gr.Textbox(visible=False), gr.Button(visible=False), gr.Button(visible=False), gr.CheckboxGroup(choices=[], visible=False), gr.Dropdown(visible=False), gr.Checkbox(visible=False), gr.Button(visible=False), gr.Dataframe(visible=False)
            try:
                df = load_data(file.name)
                columns = list(df.columns)
                
                # Analyze columns to suggest text columns
                suggested_text_columns = []
                for col in columns:
                    # Check if column contains text data
                    if df[col].dtype == 'object':  # String type
                        # Check if column contains mostly text (not just numbers or dates)
                        sample = df[col].head(100).dropna()
                        if len(sample) > 0:
                            # Check if most values contain spaces (indicating text)
                            text_ratio = sum(' ' in str(val) for val in sample) / len(sample)
                            if text_ratio > 0.3:  # If more than 30% of values contain spaces
                                suggested_text_columns.append(col)
                
                # If no columns were suggested, use all object columns
                if not suggested_text_columns:
                    suggested_text_columns = [col for col in columns if df[col].dtype == 'object']
                
                # Get a sample of text for category suggestion
                sample_texts = []
                for col in suggested_text_columns:
                    sample_texts.extend(df[col].head(5).tolist())
                
                # Use LLM to suggest categories
                if client:
                    prompt = CATEGORY_SUGGESTION_PROMPT.format("\n---\n".join(sample_texts[:5]))
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=100
                        )
                        suggested_cats = [cat.strip() for cat in response.choices[0].message.content.strip().split(",")]
                    except:
                        suggested_cats = ["Positive", "Negative", "Neutral", "Mixed", "Other"]
                else:
                    suggested_cats = ["Positive", "Negative", "Neutral", "Mixed", "Other"]
                
                return (
                    columns, 
                    gr.CheckboxGroup(choices=columns, value=suggested_text_columns), 
                    gr.CheckboxGroup(choices=suggested_cats, value=suggested_cats, visible=True),
                    gr.Textbox(visible=True),
                    gr.Button(visible=True),
                    gr.Button(visible=True),
                    gr.CheckboxGroup(choices=columns, value=suggested_text_columns, visible=True),
                    gr.Dropdown(visible=True),
                    gr.Checkbox(visible=True),
                    gr.Button(visible=True),
                    gr.Dataframe(value=df, visible=True)
                )
            except Exception as e:
                return [], gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[], visible=False), gr.Textbox(visible=False), gr.Button(visible=False), gr.Button(visible=False), gr.CheckboxGroup(choices=[], visible=False), gr.Dropdown(visible=False), gr.Checkbox(visible=False), gr.Button(visible=False), gr.Dataframe(visible=False)
        
        # Function to add a new category
        def add_new_category(current_categories, new_category):
            if not new_category or new_category.strip() == "":
                return current_categories
            new_categories = current_categories + [new_category.strip()]
            return gr.CheckboxGroup(choices=new_categories, value=new_categories)
        
        # Function to update categories textbox
        def update_categories_textbox(selected_categories):
            return ", ".join(selected_categories)
        
        # Function to show results after processing
        def show_results(df, validation_report):
            """Show the results after processing"""
            if df is None:
                return gr.Row(visible=False), gr.File(visible=False), gr.File(visible=False), gr.Dataframe(visible=False)
            
            # Export to both formats
            csv_path = export_results(df, "csv")
            excel_path = export_results(df, "excel")
            
            return gr.Row(visible=True), gr.File(value=csv_path, visible=True), gr.File(value=excel_path, visible=True), gr.Dataframe(value=df, visible=True)
        
        # Function to suggest a new category
        def suggest_new_category(file, current_categories, text_columns):
            if not file or not text_columns:
                return gr.CheckboxGroup(choices=current_categories, value=current_categories)
            
            try:
                df = load_data(file.name)
                
                # Get sample texts from selected columns
                sample_texts = []
                for col in text_columns:
                    sample_texts.extend(df[col].head(5).tolist())
                
                if client:
                    prompt = ADDITIONAL_CATEGORY_PROMPT.format(
                        existing_categories=", ".join(current_categories),
                        sample_texts="\n---\n".join(sample_texts[:5])
                    )
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=50
                        )
                        new_cat = response.choices[0].message.content.strip()
                        if new_cat and new_cat not in current_categories:
                            current_categories.append(new_cat)
                    except:
                        pass
                
                return gr.CheckboxGroup(choices=current_categories, value=current_categories)
            except Exception as e:
                return gr.CheckboxGroup(choices=current_categories, value=current_categories)
        
        # Function to handle export and show download button
        def handle_export(df, format_type):
            if df is None:
                return gr.File(visible=False)
            file_path = export_results(df, format_type)
            return gr.File(value=file_path, visible=True)
        
        # Function to improve classification based on validation report
        def improve_classification(df, validation_report, text_columns, categories, classifier_type, show_explanations, file):
            """Improve classification based on validation report"""
            if df is None or not validation_report:
                return df, validation_report, gr.Button(visible=False), gr.CheckboxGroup(choices=[], value=[])
            
            try:
                # Extract insights from validation report
                if client:
                    prompt = VALIDATION_ANALYSIS_PROMPT.format(
                        validation_report=validation_report,
                        current_categories=categories
                    )
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=300
                        )
                        improvements = json.loads(response.choices[0].message.content.strip())
                        
                        # Get current categories
                        current_categories = [cat.strip() for cat in categories.split(",")]
                        
                        # If new categories are needed, suggest them based on the data
                        if improvements.get("new_categories_needed", False):
                            # Get sample texts for category suggestion
                            sample_texts = []
                            for col in text_columns:
                                if isinstance(file, str):
                                    temp_df = load_data(file)
                                else:
                                    temp_df = load_data(file.name)
                                sample_texts.extend(temp_df[col].head(5).tolist())
                            
                            category_prompt = CATEGORY_IMPROVEMENT_PROMPT.format(
                                current_categories=", ".join(current_categories),
                                analysis=improvements.get('analysis', ''),
                                sample_texts="\n---\n".join(sample_texts[:5])
                            )
                            
                            category_response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": category_prompt}],
                                temperature=0.2,
                                max_tokens=100
                            )
                            
                            new_categories = [cat.strip() for cat in category_response.choices[0].message.content.strip().split(",")]
                            # Combine current and new categories
                            all_categories = current_categories + new_categories
                            categories = ",".join(all_categories)
                        
                        # Process with improved parameters
                        improved_df, new_validation = process_file(
                            file,
                            text_columns,
                            categories,
                            classifier_type,
                            show_explanations
                        )
                        
                        return improved_df, new_validation, gr.Button(visible=True), gr.CheckboxGroup(choices=all_categories, value=all_categories)
                    except Exception as e:
                        print(f"Error in improvement process: {str(e)}")
                        return df, validation_report, gr.Button(visible=True), gr.CheckboxGroup(choices=current_categories, value=current_categories)
                else:
                    return df, validation_report, gr.Button(visible=True), gr.CheckboxGroup(choices=current_categories, value=current_categories)
            except Exception as e:
                print(f"Error in improvement process: {str(e)}")
                return df, validation_report, gr.Button(visible=True), gr.CheckboxGroup(choices=current_categories, value=current_categories)
        
        # Connect functions
        load_categories_button.click(
            load_file_and_suggest_categories,
            inputs=[file_input],
            outputs=[
                available_columns, 
                text_column, 
                suggested_categories,
                new_category,
                add_category_button,
                suggest_category_button,
                text_column,
                classifier_type,
                show_explanations,
                process_button,
                original_df
            ]
        )
        
        add_category_button.click(
            add_new_category,
            inputs=[suggested_categories, new_category],
            outputs=[suggested_categories]
        )
        
        suggested_categories.change(
            update_categories_textbox,
            inputs=[suggested_categories],
            outputs=[categories]
        )
        
        suggest_category_button.click(
            suggest_new_category,
            inputs=[file_input, suggested_categories, text_column],
            outputs=[suggested_categories]
        )
        
        process_button.click(
            lambda: gr.Dataframe(visible=True),
            inputs=[],
            outputs=[results_df]
        ).then(
            process_file,
            inputs=[file_input, text_column, categories, classifier_type, show_explanations],
            outputs=[results_df, validation_output]
        ).then(
            show_results,
            inputs=[results_df, validation_output],
            outputs=[results_row, csv_download, excel_download, results_df]
        ).then(
            visualize_results,
            inputs=[results_df, text_column],
            outputs=[visualization]
        ).then(
            lambda x: gr.Button(visible=True),
            inputs=[],
            outputs=[improve_button]
        )
        
        improve_button.click(
            improve_classification,
            inputs=[results_df, validation_output, text_column, categories, classifier_type, show_explanations, file_input],
            outputs=[results_df, validation_output, improve_button, suggested_categories]
        ).then(
            show_results,
            inputs=[results_df, validation_output],
            outputs=[results_row, csv_download, excel_download, results_df]
        ).then(
            visualize_results,
            inputs=[results_df, text_column],
            outputs=[visualization]
        )

def create_example_data():
    """Create example data for demonstration"""
    from utils import create_example_file
    example_path = create_example_file()
    return f"Example file created at: {example_path}"

if __name__ == "__main__":
    # Create examples directory and sample file if it doesn't exist
    if not os.path.exists("examples"):
        create_example_data()
        
    # Launch the Gradio app
    demo.launch()
