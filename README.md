# Project Fixer & Generator

A powerful Streamlit-based web application that allows developers to upload their project codebases (as ZIP files), generate essential DevOps files, automatically fix code quality issues using Google Gemini AI, and perform structured AI-driven analysis of the codebase.

## Features

- **Project Upload**: Upload your entire project as a ZIP file for processing.
- **DevOps File Generation**:
  - Generate `Dockerfile` and `.dockerignore` for containerization.
  - Generate `README.md` with project description, installation, and usage instructions.
  - Generate `requirements.txt` listing project dependencies.
  - Generate `.gitignore` with common Python ignores.
- **Codebase Auto-Fix**: Automatically fix bugs, improve formatting, and enhance code quality across all valid code files using Google Gemini AI.
- **AI-Powered Analysis**: Analyze the codebase to provide:
  - Project summary.
  - List of bugs with suggested fixes.
  - File-level summaries.
  - Improvement suggestions.
- **Download Fixed Project**: After fixing, download the entire corrected project as a ZIP file.
- **Report Generation**: Includes human-readable and JSON reports of fixes applied.

## Use Cases

- **Code Quality Improvement**: Developers can upload messy or buggy codebases and get them automatically cleaned up and fixed.
- **DevOps Setup**: Quickly generate Dockerfiles, ignore files, and dependency lists for new or existing projects.
- **Project Documentation**: Auto-generate README files with descriptions, installation steps, and usage guides.
- **Code Review Assistance**: Use the AI analysis to identify potential issues, bugs, and areas for improvement before manual review.
- **Rapid Prototyping**: Fix and analyze codebases on the fly to accelerate development cycles.
- **Educational Tool**: Learn from AI-suggested fixes and improvements in code structure and best practices.

## Installation

1. Clone or download this repository.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Google API key:
   - Obtain a Google Generative AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
   - Set the environment variable:
     ```
     export GOOGLE_API_KEY="your-api-key-here"
     ```
     Or create a `.env` file in the project root with:
     ```
     GOOGLE_API_KEY=your-api-key-here
     ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Open the provided URL in your browser (usually `http://localhost:8501`).
3. Upload your project as a ZIP file.
4. Choose from the available options:
   - Generate DevOps files (Dockerfile, README, etc.).
   - Auto-fix the entire codebase.
   - Analyze the codebase with AI.
5. Download generated files or the fixed project as needed.

## Requirements

- Python 3.7+
- Google Generative AI API key
- Internet connection for AI processing

## License

This project is open-source. Feel free to use, modify, and distribute as needed.

## Contributing

Contributions are welcome! Please submit issues or pull requests for improvements.

## Disclaimer

This tool uses AI for code analysis and fixes. Always review AI-generated code and suggestions before applying them to production environments.
