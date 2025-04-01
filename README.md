# HAULYP Event Analysis Application

## Overview
This application analyzes event attendance data from Wild Apricot exports, providing insights into both event and member participation patterns.

## Setup Instructions

1. Install Python requirements:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with your admin password:
```
ADMIN_PASSWORD=your_password_here
```

## Running the Application

1. Start the application:
```bash
streamlit run event_analysis/app.py
```

2. Open your web browser and navigate to:
   - Local URL: http://localhost:8501
   - Or use the Network URL shown in the terminal for access from other devices

## Using the Application

### 1. Data Upload
- Click the "Upload Data" section at the top
- Upload your Wild Apricot Excel export file
- The app accepts `.xls` or `.xlsx` files

### 2. Analysis Types
Choose between two types of analysis:

#### Member Analysis
1. Select "Member Name" from the dropdown
2. Use the search box to find a specific member
   - You can search by first or last name
   - The dropdown will filter as you type
3. Select a member from the dropdown to view their:
   - Total events attended
   - Unique dates attended
   - Most common day of attendance
   - Event timeline
   - Day of week distribution
   - Monthly attendance patterns

#### Event Analysis
1. Select "Event Name" from the dropdown
2. Use the search and sort options to find your event:
   - Search by event name or location
   - Sort by:
     - Name (alphabetical)
     - Date (newest/oldest)
     - Location
3. Select an event to view:
   - Total attendance
   - Average attendance
   - Number of occurrences
   - Attendance trends
   - List of attendees with attendance rates

## Tips
- Use the search functionality to quickly find specific members or events
- Sort options help organize events by date or location
- Hover over graphs for detailed information
- Use the full-screen button on graphs for a better view

## Troubleshooting
- If the file upload fails, ensure it's in the correct Excel format
- Check that date columns are properly formatted in your Excel file
- Ensure all required columns are present (Event Title, Date, Location)
- If you encounter errors, verify your Excel file matches the expected Wild Apricot export format

## Support
For additional support or to report issues, please contact your system administrator.

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