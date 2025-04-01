# Event Attendance Analysis Dashboard

An interactive dashboard for analyzing event attendance data from Excel files.

## Features

- Interactive data visualization
- Event attendance analysis
- Name matching and data merging
- Excel file support
- Statistical analysis

## Installation

### Option 1: Install from PyPI
```bash
pip install event-analysis
```

### Option 2: Install from source
```bash
git clone https://github.com/yourusername/event-analysis.git
cd event-analysis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

1. Start the dashboard:
```bash
event-analysis
```

2. Open your browser and navigate to http://localhost:8501

3. Upload your Excel file and start analyzing!

## Deployment

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository and the main file (event_analysis/app.py)
6. Click "Deploy"

### Option 2: Deploy on Heroku

1. Create a Heroku account and install the Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Deploy your app:
```bash
git push heroku main
```

### Option 3: Deploy on Docker

1. Build the Docker image:
```bash
docker build -t event-analysis .
```

2. Run the container:
```bash
docker run -p 8501:8501 event-analysis
```

## Data Format Requirements

Your Excel file should contain:
- Event names or dates
- Attendee information
- Any additional metadata you want to analyze

## Requirements

- Python 3.9 or higher
- See requirements.txt for all dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 