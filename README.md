# Feedback-Fusion-for-Educators

Feedback Fusion is an AI-powered NLP tool designed to automate the classification of open-ended survey responses, particularly in large-scale MOOCs. By leveraging BERT-based models, it enhances efficiency in identifying student concerns, improving instructor response times, and streamlining feedback analysis.

## Features
- **Automated Classification**: Uses BERT-based models to categorize feedback into predefined concern categories.
- **Category Distribution Analysis**: Visual representation of concern distribution to help instructors identify key issues at a glance.
- **Streamlit UI**: User-friendly web interface for easy accessibility, even for non-technical users.
- **Real-time Insights**: Enables rapid analysis and response to student concerns.
- **Downloadable Reports**: Export categorized responses for further analysis and documentation.
- **Model Comparisons**: Evaluates multiple NLP models (BERT Base, TinyBERT, DistilBERT) based on accuracy, latency, and efficiency.

## Installation
Clone the repository and install dependencies:

```bash
git clone [https://github.com/your-username/feedback-fusion.git](https://github.com/AyushiChakrabarty/Feedback-Fusion-for-Educators.git)
cd feedback-fusion
pip install -r requirements.txt
```

## Usage
Run the main script to classify survey responses:

```bash
streamlit run app.py
```

## Model Performance
| Model      | Accuracy | Latency (ms) |
|------------|---------|--------------|
| BERT Base  | 80.1%   | 8.3          |
| TinyBERT   | 75.1%   | 2.3          |
| DistilBERT | 79.6%   | 4.2          |

## Limitations
- **Class Imbalance**: Underrepresented categories may have lower classification performance.
- **Augmentation Constraints**: Data augmentation replicates patterns from limited samples, impacting variability.
- **Generalization**: Performance may vary across different datasets without domain-specific fine-tuning.

## Future Enhancements
- **Multilingual Support**: Expanding analysis to multiple languages.
- **Retrieval-Augmented Generation (RAG)**: Generating automated instructor response suggestions.
- **Integration with Other Platforms**: Extending support to non-education industries with large-scale feedback processing.

## Contributors
- **Ayushi Chakrabarty (achakrabarty8@gatech.edu)** – Lead Developer
- **Greg Mayer (greg.mayer@gatech.edu)**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by challenges in large-scale MOOC feedback analysis.
- Research backed by existing NLP and educational data mining literature.
---
