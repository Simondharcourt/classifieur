"""Prompts used in the text classification system"""

# Category suggestion prompt
CATEGORY_SUGGESTION_PROMPT = """
Based on these example texts, suggest between 3 and 5 appropriate categories for classification:

{}

Return your answer as a comma-separated list of category names only.
"""

# Text classification prompt
TEXT_CLASSIFICATION_PROMPT = """
Classify the following text into one of these categories: {categories}

Text: {text}

Return your answer in JSON format with these fields:
- category: the chosen category from the list
- confidence: a value between 0 and 100 indicating your confidence in this classification (as a percentage)
- explanation: a brief explanation of why this category was chosen (1-2 sentences)

JSON response:
"""

# Additional category suggestion prompt
ADDITIONAL_CATEGORY_PROMPT = """
Based on these example texts and the existing categories ({existing_categories}),
suggest one additional appropriate category for classification.

Example texts:
{}

Return only the suggested category name, nothing else.
"""

# Validation report analysis prompt
VALIDATION_ANALYSIS_PROMPT = """
Based on this validation report, analyze the current classification and suggest improvements:

{validation_report}

Return your answer in JSON format with these fields:
- suggested_categories: list of improved category names (must be different from current categories: {current_categories})
- confidence_threshold: a number between 0 and 100 for minimum confidence
- focus_areas: list of specific aspects to focus on during classification
- analysis: a brief analysis of what needs improvement
- new_categories_needed: boolean indicating if new categories should be added

JSON response:
"""

# Category improvement prompt
CATEGORY_IMPROVEMENT_PROMPT = """
Based on these example texts and the current categories ({current_categories}),
suggest new categories that would improve the classification. The validation report indicates:
{analysis}

Example texts:
{}

Return your answer as a comma-separated list of new category names only.
"""
