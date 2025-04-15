import logging
import time
import traceback
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, List, Dict, Any, Tuple, Union
import pandas as pd
from pathlib import Path
import json

from classifiers import TFIDFClassifier, LLMClassifier
from utils import load_data, validate_results, get_sample_texts
from client import get_client
from prompts import VALIDATION_ANALYSIS_PROMPT, CATEGORY_IMPROVEMENT_PROMPT


def update_api_key(api_key: str) -> Tuple[bool, str]:
    """Update the OpenAI API key"""
    from client import initialize_client
    return initialize_client(api_key)


async def process_file_async(
    file: Union[str, Path],
    text_columns: List[str],
    categories: Optional[str],
    classifier_type: str,
    show_explanations: bool
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Async version of process_file"""
    # Initialize result_df and validation_report
    result_df: Optional[pd.DataFrame] = None
    validation_report: Optional[str] = None

    try:
        # Load data from file
        if isinstance(file, str):
            df: pd.DataFrame = load_data(file)
        else:
            df: pd.DataFrame = load_data(file.name)

        if not text_columns:
            return None, "Please select at least one text column"

        # Check if all selected columns exist
        missing_columns: List[str] = [col for col in text_columns if col not in df.columns]
        if missing_columns:
            return (
                None,
                f"Columns not found in the file: {', '.join(missing_columns)}. Available columns: {', '.join(df.columns)}",
            )

        # Combine text from selected columns
        texts: List[str] = []
        for _, row in df.iterrows():
            combined_text: str = " ".join(str(row[col]) for col in text_columns)
            texts.append(combined_text)

        # Parse categories if provided
        category_list: List[str] = []
        if categories:
            category_list = [cat.strip() for cat in categories.split(",")]

        # Select classifier based on data size and user choice
        num_texts: int = len(texts)

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

        # Get the client instance
        client = get_client()

        # Initialize appropriate classifier
        if classifier_type == "tfidf":
            classifier: TFIDFClassifier = TFIDFClassifier()
            results: List[Dict[str, Any]] = classifier.classify(texts, category_list)
        elif classifier_type in ["gpt35", "gpt4"]:
            if client is None:
                return (
                    None,
                    "Erreur : Le client API n'est pas initialisé. Veuillez configurer une clé API valide dans l'onglet 'Setup'.",
                )
            model: str = "gpt-3.5-turbo" if classifier_type == "gpt35" else "gpt-4"
            classifier: LLMClassifier = LLMClassifier(client=client, model=model)
            results: List[Dict[str, Any]] = await classifier.classify_async(texts, category_list)
        else:  # hybrid
            if client is None:
                return (
                    None,
                    "Erreur : Le client API n'est pas initialisé. Veuillez configurer une clé API valide dans l'onglet 'Setup'.",
                )
            # First pass with TF-IDF
            tfidf_classifier: TFIDFClassifier = TFIDFClassifier()
            tfidf_results: List[Dict[str, Any]] = tfidf_classifier.classify(texts, category_list)

            # Second pass with LLM for low confidence results
            llm_classifier: LLMClassifier = LLMClassifier(client=client, model="gpt-3.5-turbo")
            results: List[Optional[Dict[str, Any]]] = []
            low_confidence_texts: List[str] = []
            low_confidence_indices: List[int] = []

            for i, (text, tfidf_result) in enumerate(zip(texts, tfidf_results)):
                if tfidf_result["confidence"] < 70:  # If confidence is below 70%
                    low_confidence_texts.append(text)
                    low_confidence_indices.append(i)
                    results.append(None)  # Placeholder
                else:
                    results.append(tfidf_result)

            if low_confidence_texts:
                llm_results: List[Dict[str, Any]] = await llm_classifier.classify_async(
                    low_confidence_texts, category_list
                )
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
        error_traceback: str = traceback.format_exc()
        return None, f"Error: {str(e)}\n{error_traceback}"


def process_file(
    file: Union[str, Path],
    text_columns: List[str],
    categories: Optional[str],
    classifier_type: str,
    show_explanations: bool
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Synchronous wrapper for process_file_async"""
    return asyncio.run(process_file_async(file, text_columns, categories, classifier_type, show_explanations))


def export_results(df: pd.DataFrame, format_type: str) -> Optional[str]:
    """Export results to a file and return the file path for download"""
    if df is None:
        return None

    # Create a temporary file
    import tempfile
    import os

    # Create a temporary directory if it doesn't exist
    temp_dir: str = "temp_exports"
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a unique filename
    timestamp: str = time.strftime("%Y%m%d-%H%M%S")
    filename: str = f"classification_results_{timestamp}"

    if format_type == "excel":
        file_path: str = os.path.join(temp_dir, f"{filename}.xlsx")
        df.to_excel(file_path, index=False)
    else:
        file_path: str = os.path.join(temp_dir, f"{filename}.csv")
        df.to_csv(file_path, index=False)

    return file_path


async def improve_classification(
    df: pd.DataFrame,
    validation_report: str,
    text_columns: List[str],
    categories: str,
    classifier_type: str,
    show_explanations: bool,
    file: Union[str, Path]
) -> Tuple[Optional[pd.DataFrame], Optional[str], bool, List[str]]:
    """
    Improve classification based on validation report

    Args:
        df (pd.DataFrame): Current classification results
        validation_report (str): Validation report from previous classification
        text_columns (List[str]): List of text column names
        categories (str): Comma-separated list of categories
        classifier_type (str): Type of classifier to use
        show_explanations (bool): Whether to show explanations
        file (Union[str, Path]): Path to the input file

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str], bool, List[str]]: 
            - Improved dataframe
            - New validation report
            - Whether improvement was successful
            - Updated categories
    """
    if df is None or not validation_report:
        return None, validation_report, False, []

    try:
        client = get_client()
        if not client:
            return None, "Error: API client not initialized", False, []

        # Extract insights from validation report
        prompt = VALIDATION_ANALYSIS_PROMPT.format(
            validation_report=validation_report,
            current_categories=categories,
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
        )

        improvements = json.loads(response.choices[0].message.content.strip())
        current_categories = [cat.strip() for cat in categories.split(",")]

        # If new categories are needed, suggest them based on the data
        if improvements.get("new_categories_needed", False):
            # Get sample texts for category suggestion
            sample_texts = get_sample_texts(df, text_columns, sample_size=10)

            category_prompt = CATEGORY_IMPROVEMENT_PROMPT.format(
                current_categories=", ".join(current_categories),
                analysis=improvements.get("analysis", ""),
                sample_texts="\n---\n".join(sample_texts)
            )

            category_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": category_prompt}],
                    temperature=0,
                    max_tokens=100,
                )
            )

            new_categories = [
                cat.strip()
                for cat in category_response.choices[0].message.content.strip().split(",")
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

        return improved_df, new_validation, True, all_categories if improvements.get("new_categories_needed", False) else current_categories

    except Exception as e:
        print(f"Error in improvement process: {str(e)}")
        return df, validation_report, False, current_categories
