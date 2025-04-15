import os
import gradio as gr
import asyncio

import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import logging
from dotenv import load_dotenv
from process import update_api_key, process_file_async, export_results
from client import get_client, initialize_client

# Load environment variables from .env file
load_dotenv()

# Import local modules
from utils import load_data, visualize_results
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
                    label="Validation Report", interactive=False
                )
                improve_button = gr.Button(
                    "Improve Classification with Report", visible=False
                )

        # Function to load file and suggest categories
        def load_file_and_suggest_categories(file):
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
                suggested_text_columns = []
                for col in columns:
                    # Check if column contains text data
                    if df[col].dtype == "object":  # String type
                        # Check if column contains mostly text (not just numbers or dates)
                        sample = df[col].head(100).dropna()
                        if len(sample) > 0:
                            # Check if most values contain spaces (indicating text)
                            text_ratio = sum(" " in str(val) for val in sample) / len(
                                sample
                            )
                            if (
                                text_ratio > 0.3
                            ):  # If more than 30% of values contain spaces
                                suggested_text_columns.append(col)

                # If no columns were suggested, use all object columns
                if not suggested_text_columns:
                    suggested_text_columns = [
                        col for col in columns if df[col].dtype == "object"
                    ]

                # Get a sample of text for category suggestion
                sample_texts = []
                for col in suggested_text_columns:
                    sample_texts.extend(df[col].head(5).tolist())

                # Use LLM to suggest categories
                if client:
                    prompt = CATEGORY_SUGGESTION_PROMPT.format(
                        "\n---\n".join(sample_texts[:5])
                    )
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=100,
                        )
                        suggested_cats = [
                            cat.strip()
                            for cat in response.choices[0]
                            .message.content.strip()
                            .split(",")
                        ]
                    except:
                        suggested_cats = [
                            "Positive",
                            "Negative",
                            "Neutral",
                            "Mixed",
                            "Other",
                        ]
                else:
                    suggested_cats = [
                        "Positive",
                        "Negative",
                        "Neutral",
                        "Mixed",
                        "Other",
                    ]

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
        def suggest_new_category(file, current_categories, text_columns):
            if not file or not text_columns:
                return gr.CheckboxGroup(
                    choices=current_categories, value=current_categories
                )

            try:
                df = load_data(file.name)

                # Get sample texts from selected columns
                sample_texts = []
                for col in text_columns:
                    sample_texts.extend(df[col].head(5).tolist())

                if client:
                    prompt = ADDITIONAL_CATEGORY_PROMPT.format(
                        existing_categories=", ".join(current_categories),
                        sample_texts="\n---\n".join(sample_texts[:10]),
                    )
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=50,
                        )
                        new_cat = response.choices[0].message.content.strip()
                        if new_cat and new_cat not in current_categories:
                            current_categories.append(new_cat)
                    except:
                        pass

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

        # Function to improve classification based on validation report
        async def improve_classification_async(
            df,
            validation_report,
            text_columns,
            categories,
            classifier_type,
            show_explanations,
            file,
        ):
            """Async version of improve_classification"""
            if df is None or not validation_report:
                return (
                    df,
                    validation_report,
                    gr.Button(visible=False),
                    gr.CheckboxGroup(choices=[], value=[]),
                )

            try:
                # Extract insights from validation report
                if client:
                    prompt = VALIDATION_ANALYSIS_PROMPT.format(
                        validation_report=validation_report,
                        current_categories=categories,
                    )
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=300,
                        )
                        improvements = json.loads(
                            response.choices[0].message.content.strip()
                        )

                        # Get current categories
                        current_categories = [
                            cat.strip() for cat in categories.split(",")
                        ]

                        # If new categories are needed, suggest them based on the data
                        if improvements.get("new_categories_needed", False):
                            # Get sample texts for category suggestion
                            sample_texts = []
                            for col in text_columns:
                                if isinstance(file, str):
                                    temp_df = load_data(file)
                                else:
                                    temp_df = load_data(file.name)
                                sample_texts.extend(temp_df[col].head(10).tolist())

                            category_prompt = CATEGORY_IMPROVEMENT_PROMPT.format(
                                current_categories=", ".join(current_categories),
                                analysis=improvements.get("analysis", ""),
                                sample_texts="\n---\n".join(sample_texts[:10]),
                            )

                            category_response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": category_prompt}],
                                temperature=0,
                                max_tokens=100,
                            )

                            new_categories = [
                                cat.strip()
                                for cat in category_response.choices[0]
                                .message.content.strip()
                                .split(",")
                            ]
                            # Combine current and new categories
                            all_categories = current_categories + new_categories
                            categories = ",".join(all_categories)

                        # Process with improved parameters
                        improved_df, new_validation = await process_file_async(
                            file,
                            text_columns,
                            categories,
                            classifier_type,
                            show_explanations,
                        )

                        return (
                            improved_df,
                            new_validation,
                            gr.Button(visible=True),
                            gr.CheckboxGroup(
                                choices=all_categories, value=all_categories
                            ),
                        )
                    except Exception as e:
                        print(f"Error in improvement process: {str(e)}")
                        return (
                            df,
                            validation_report,
                            gr.Button(visible=True),
                            gr.CheckboxGroup(
                                choices=current_categories, value=current_categories
                            ),
                        )
                else:
                    return (
                        df,
                        validation_report,
                        gr.Button(visible=True),
                        gr.CheckboxGroup(
                            choices=current_categories, value=current_categories
                        ),
                    )
            except Exception as e:
                print(f"Error in improvement process: {str(e)}")
                return (
                    df,
                    validation_report,
                    gr.Button(visible=True),
                    gr.CheckboxGroup(
                        choices=current_categories, value=current_categories
                    ),
                )

        def improve_classification(
            df,
            validation_report,
            text_columns,
            categories,
            classifier_type,
            show_explanations,
            file,
        ):
            """Synchronous wrapper for improve_classification_async"""
            return asyncio.run(
                improve_classification_async(
                    df,
                    validation_report,
                    text_columns,
                    categories,
                    classifier_type,
                    show_explanations,
                    file,
                )
            )

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
