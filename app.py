import os
import gradio as gr
import asyncio

import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import logging
from dotenv import load_dotenv
from process import update_api_key, process_file_async, export_results, improve_classification
from client import get_client, initialize_client
from utils import load_data, visualize_results, analyze_text_columns, get_sample_texts
from classifiers.llm import LLMClassifier

# Load environment variables from .env file
load_dotenv()

# Import local modules
from prompts import (
    CATEGORY_SUGGESTION_PROMPT,
    ADDITIONAL_CATEGORY_PROMPT,
    VALIDATION_ANALYSIS_PROMPT,
    CATEGORY_IMPROVEMENT_PROMPT,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Initialize client if API key is available
if OPENAI_API_KEY:
    success, message = initialize_client(OPENAI_API_KEY)
    if success:
        logging.info("OpenAI client initialized successfully")
    else:
        logging.error(f"Failed to initialize OpenAI client: {message}")

# Create Gradio interface
with gr.Blocks(title="Text Classification System") as demo:
    gr.Markdown("# Text Classification System")
    gr.Markdown("Upload your data file (Excel/CSV) and classify text using AI")

    with gr.Tab("Setup"):
        api_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter your API key here",
            type="password",
            value=OPENAI_API_KEY,
        )
        api_key_button = gr.Button("Update API Key")
        api_key_message = gr.Textbox(label="Status", interactive=False)

        # Display current API status
        client = get_client()
        api_status = "API Key is set" if client else "No API Key found. Please set one."
        gr.Markdown(f"**Current API Status**: {api_status}")

        api_key_button.click(
            update_api_key, inputs=[api_key_input], outputs=[api_key_message]
        )

    with gr.Tab("Classify Data"):
        with gr.Column():
            file_input = gr.File(label="Upload Excel/CSV File")

            # Variable to store available columns
            available_columns = gr.State([])

            # Button to load file and suggest categories
            load_categories_button = gr.Button("Load File")

            # Display original dataframe
            original_df = gr.Dataframe(
                label="Original Data", interactive=False, visible=False
            )

            with gr.Row():
                with gr.Column():
                    suggested_categories = gr.CheckboxGroup(
                        label="Suggested Categories",
                        choices=[],
                        value=[],
                        interactive=True,
                        visible=False,
                    )

                    new_category = gr.Textbox(
                        label="Add New Category",
                        placeholder="Enter a new category name",
                        visible=False,
                    )
                    with gr.Row():
                        add_category_button = gr.Button("Add Category", visible=False)
                        suggest_category_button = gr.Button(
                            "Suggest Category", visible=False
                        )

                    # Original categories input (hidden)
                    categories = gr.Textbox(visible=False)

                with gr.Column():
                    text_column = gr.CheckboxGroup(
                        label="Select Text Columns",
                        choices=[],
                        interactive=True,
                        visible=False,
                    )

                    classifier_type = gr.Dropdown(
                        choices=[
                            ("TF-IDF (Rapide, <1000 lignes)", "tfidf"),
                            ("LLM GPT-3.5 (Fiable, <1000 lignes)", "gpt35"),
                            ("LLM GPT-4 (Très fiable, <500 lignes)", "gpt4"),
                            ("TF-IDF + LLM (Hybride, >1000 lignes)", "hybrid"),
                        ],
                        label="Modèle de classification",
                        value="gpt35",
                        visible=False,
                    )
                    show_explanations = gr.Checkbox(
                        label="Show Explanations", value=True, visible=False
                    )

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
                validation_output = gr.Textbox(
                    label="Validation Report", interactive=True,
                    lines=15
                )
                improve_button = gr.Button(
                    "Improve Classification with Report", visible=False
                )

        # Function to load file and suggest categories
        async def load_file_and_suggest_categories(file):
            if not file:
                return (
                    [],
                    gr.CheckboxGroup(choices=[]),
                    gr.CheckboxGroup(choices=[], visible=False),
                    gr.Textbox(visible=False),
                    gr.Button(visible=False),
                    gr.Button(visible=False),
                    gr.CheckboxGroup(choices=[], visible=False),
                    gr.Dropdown(visible=False),
                    gr.Checkbox(visible=False),
                    gr.Button(visible=False),
                    gr.Dataframe(visible=False),
                )
            try:
                df = load_data(file.name)
                columns = list(df.columns)

                # Analyze columns to suggest text columns
                suggested_text_columns = analyze_text_columns(df)

                # Get sample texts for category suggestion
                sample_texts = get_sample_texts(df, suggested_text_columns)

                # Use LLM to suggest categories
                if client:
                    classifier = LLMClassifier(client=client)
                    suggested_cats = await classifier.suggest_categories_from_texts(sample_texts)
                else:
                    suggested_cats = ["Positive", "Negative", "Neutral", "Mixed", "Other"]

                return (
                    columns,
                    gr.CheckboxGroup(choices=columns, value=suggested_text_columns),
                    gr.CheckboxGroup(
                        choices=suggested_cats, value=suggested_cats, visible=True
                    ),
                    gr.Textbox(visible=True),
                    gr.Button(visible=True),
                    gr.Button(visible=True),
                    gr.CheckboxGroup(
                        choices=columns, value=suggested_text_columns, visible=True
                    ),
                    gr.Dropdown(visible=True),
                    gr.Checkbox(visible=True),
                    gr.Button(visible=True),
                    gr.Dataframe(value=df, visible=True),
                )
            except Exception as e:
                return (
                    [],
                    gr.CheckboxGroup(choices=[]),
                    gr.CheckboxGroup(choices=[], visible=False),
                    gr.Textbox(visible=False),
                    gr.Button(visible=False),
                    gr.Button(visible=False),
                    gr.CheckboxGroup(choices=[], visible=False),
                    gr.Dropdown(visible=False),
                    gr.Checkbox(visible=False),
                    gr.Button(visible=False),
                    gr.Dataframe(visible=False),
                )

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
                return (
                    gr.Row(visible=False),
                    gr.File(visible=False),
                    gr.File(visible=False),
                    gr.Dataframe(visible=False),
                )

            # Export to both formats
            csv_path = export_results(df, "csv")
            excel_path = export_results(df, "excel")

            return (
                gr.Row(visible=True),
                gr.File(value=csv_path, visible=True),
                gr.File(value=excel_path, visible=True),
                gr.Dataframe(value=df, visible=True),
            )

        # Function to suggest a new category
        async def suggest_new_category(file, current_categories, text_columns):
            if not file or not text_columns:
                return gr.CheckboxGroup(
                    choices=current_categories, value=current_categories
                )

            try:
                df = load_data(file.name)
                sample_texts = get_sample_texts(df, text_columns)

                if client:
                    classifier = LLMClassifier(client=client)
                    new_categories = await classifier.suggest_categories_from_texts(
                        sample_texts, current_categories
                    )
                    return gr.CheckboxGroup(
                        choices=new_categories, value=new_categories
                    )

                return gr.CheckboxGroup(
                    choices=current_categories, value=current_categories
                )
            except Exception as e:
                return gr.CheckboxGroup(
                    choices=current_categories, value=current_categories
                )

        # Function to handle export and show download button
        def handle_export(df, format_type):
            if df is None:
                return gr.File(visible=False)
            file_path = export_results(df, format_type)
            return gr.File(value=file_path, visible=True)

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
                original_df,
            ],
        )

        add_category_button.click(
            add_new_category,
            inputs=[suggested_categories, new_category],
            outputs=[suggested_categories],
        )

        suggested_categories.change(
            update_categories_textbox,
            inputs=[suggested_categories],
            outputs=[categories],
        )

        suggest_category_button.click(
            suggest_new_category,
            inputs=[file_input, suggested_categories, text_column],
            outputs=[suggested_categories],
        )

        process_button.click(
            lambda: gr.Dataframe(visible=True), inputs=[], outputs=[results_df]
        ).then(
            process_file_async,
            inputs=[
                file_input,
                text_column,
                categories,
                classifier_type,
                show_explanations,
            ],
            outputs=[results_df, validation_output],
        ).then(
            show_results,
            inputs=[results_df, validation_output],
            outputs=[results_row, csv_download, excel_download, results_df],
        ).then(
            visualize_results, inputs=[results_df, text_column], outputs=[visualization]
        ).then(
            lambda x: gr.Button(visible=True), inputs=[], outputs=[improve_button]
        )

        improve_button.click(
            improve_classification,
            inputs=[
                results_df,
                validation_output,
                text_column,
                categories,
                classifier_type,
                show_explanations,
                file_input,
            ],
            outputs=[
                results_df,
                validation_output,
                improve_button,
                suggested_categories,
            ],
        ).then(
            show_results,
            inputs=[results_df, validation_output],
            outputs=[results_row, csv_download, excel_download, results_df],
        ).then(
            visualize_results, inputs=[results_df, text_column], outputs=[visualization]
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
