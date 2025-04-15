import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile
from prompts import VALIDATION_PROMPT
from typing import List, Optional, Any, Union
from pathlib import Path
from matplotlib.figure import Figure


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from an Excel or CSV file

    Args:
        file_path (str): Path to the file

    Returns:
        pd.DataFrame: Loaded data
    """
    file_ext: str = os.path.splitext(file_path)[1].lower()

    if file_ext == ".xlsx" or file_ext == ".xls":
        return pd.read_excel(file_path)
    elif file_ext == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Please upload an Excel or CSV file."
        )


def export_data(df: pd.DataFrame, file_name: str, format_type: str = "excel") -> str:
    """
    Export dataframe to file

    Args:
        df (pd.DataFrame): Dataframe to export
        file_name (str): Name of the output file
        format_type (str): "excel" or "csv"

    Returns:
        str: Path to the exported file
    """
    # Create export directory if it doesn't exist
    export_dir: str = "exports"
    os.makedirs(export_dir, exist_ok=True)

    # Full path for the export file
    export_path: str = os.path.join(export_dir, file_name)

    # Export based on format type
    if format_type == "excel":
        df.to_excel(export_path, index=False)
    else:
        df.to_csv(export_path, index=False)

    return export_path


def visualize_results(df: pd.DataFrame, text_column: str, category_column: str = "Category") -> Figure:
    """
    Create visualization of classification results

    Args:
        df (pd.DataFrame): Dataframe with classification results
        text_column (str): Name of the column containing text data
        category_column (str): Name of the column containing categories

    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    # Check if category column exists
    if category_column not in df.columns:
        # Create a simple figure with a message
        fig: Figure
        ax: Any
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5, "No categories to display", ha="center", va="center", fontsize=12
        )
        ax.set_title("No Classification Results Available")
        plt.tight_layout()
        return fig

    # Get categories and their counts
    category_counts: pd.Series = df[category_column].value_counts()

    # Create a new figure
    fig: Figure
    ax: Any
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram
    bars: Any = ax.bar(category_counts.index, category_counts.values)

    # Add value labels on top of each bar
    for bar in bars:
        height: float = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # Customize the plot
    ax.set_xlabel("Categories")
    ax.set_ylabel("Number of Texts")
    ax.set_title("Distribution of Classified Texts")

    # Rotate x-axis labels if they're too long
    plt.xticks(rotation=45, ha="right")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    return fig


def validate_results(df: pd.DataFrame, text_columns: List[str], client: Any) -> str:
    """
    Use LLM to validate the classification results

    Args:
        df (pd.DataFrame): Dataframe with classification results
        text_columns (list): List of column names containing text data
        client: LiteLLM client

    Returns:
        str: Validation report
    """
    try:
        # Sample a few rows for validation
        sample_size: int = min(5, len(df))
        sample_df: pd.DataFrame = df.sample(n=sample_size, random_state=42)

        # Build validation prompts
        validation_prompts: List[str] = []
        for _, row in sample_df.iterrows():
            # Combine text from all selected columns
            text: str = " ".join(str(row[col]) for col in text_columns)
            assigned_category: str = row["Category"]
            confidence: float = row["Confidence"]

            validation_prompts.append(
                f"Text: {text}\nAssigned Category: {assigned_category}\nConfidence: {confidence}\n"
            )

        # Use the prompt from prompts.py
        prompt: str = VALIDATION_PROMPT.format("\n---\n".join(validation_prompts))

        # Call LLM API
        response: Any = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )

        validation_report: str = response.choices[0].message.content.strip()
        return validation_report

    except Exception as e:
        return f"Validation failed: {str(e)}"
