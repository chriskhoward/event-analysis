import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import calendar
import os
from scipy import stats as scipy_stats
from fuzzywuzzy import fuzz
from nameparser import HumanName
import re
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple, Optional

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Event Attendance Analysis",
    page_icon="📊",
    layout="wide"
)

def calculate_event_stats(event_data: pd.DataFrame, event_dates: pd.Series) -> Dict[str, Any]:
    """Calculate comprehensive statistics for an event.
    
    Args:
        event_data (pd.DataFrame): DataFrame containing event data
        event_dates (pd.Series): Series containing event dates
        
    Returns:
        Dict[str, Any]: Dictionary containing event statistics
    """
    stats = {}
    
    try:
        # Ensure we have valid data
        if event_data.empty or event_dates.empty:
            raise ValueError("No data available for analysis")
        
        # Basic counts
        stats['total_attendees'] = len(event_data)
        stats['occurrences'] = len(event_dates.unique())
        stats['avg_attendance'] = stats['total_attendees'] / stats['occurrences'] if stats['occurrences'] > 0 else 0
        
        # Peak attendance
        attendance_by_date = event_data.groupby(event_dates).size()
        if not attendance_by_date.empty:
            stats['peak_attendance'] = attendance_by_date.max()
            stats['peak_date'] = attendance_by_date.idxmax()
        else:
            stats['peak_attendance'] = 0
            stats['peak_date'] = None
        
        # Attendance trend
        if len(attendance_by_date) > 1:
            x = np.arange(len(attendance_by_date))
            slope, _, r_value, _, _ = scipy_stats.linregress(x, attendance_by_date.values)
            stats['attendance_trend'] = 'Increasing' if slope > 0 else 'Decreasing'
            stats['trend_strength'] = abs(r_value)
        else:
            stats['attendance_trend'] = 'No trend'
            stats['trend_strength'] = 0
        
        # Consistency
        stats['attendance_std'] = attendance_by_date.std() if not attendance_by_date.empty else 0
        stats['consistency'] = 'High' if stats['attendance_std'] < stats['avg_attendance'] * 0.3 else 'Medium' if stats['attendance_std'] < stats['avg_attendance'] * 0.6 else 'Low'
        
        return stats
    except Exception as e:
        st.error(f"Error calculating event statistics: {str(e)}")
        # Return default values in case of error
        return {
            'total_attendees': 0,
            'occurrences': 0,
            'avg_attendance': 0,
            'peak_attendance': 0,
            'peak_date': None,
            'attendance_trend': 'No data',
            'trend_strength': 0,
            'attendance_std': 0,
            'consistency': 'No data'
        }

# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.session_state["credentials"]:
            if st.session_state["password"] == st.session_state["credentials"][st.session_state["username"]]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store password
            else:
                st.session_state["password_correct"] = False
        else:
            st.session_state["password_correct"] = False

    # First run or credentials not loaded
    if "credentials" not in st.session_state:
        # Default credentials if environment variables are not set
        admin_username = os.getenv("ADMIN_USERNAME", "admin_haulyp")
        admin_password = os.getenv("ADMIN_PASSWORD", "Haulyp@Admin2024!")
        user_username = os.getenv("USER_USERNAME", "user_haulyp")
        user_password = os.getenv("USER_PASSWORD", "Haulyp@User2024!")
        
        st.session_state["credentials"] = {
            admin_username: admin_password,
            user_username: user_password
        }

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.markdown("### Login Required")
    st.markdown("Please enter your credentials to access the dashboard.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Invalid username or password")
    
    return False

def main():
    # Check authentication
    if not check_password():
        st.stop()

    # Title
    st.title("Event Attendance Analysis Dashboard")

    # Initialize session state for the dataframe and unique names
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'unique_names' not in st.session_state:
        st.session_state.unique_names = []
    if 'first_name_col' not in st.session_state:
        st.session_state.first_name_col = None
    if 'last_name_col' not in st.session_state:
        st.session_state.last_name_col = None
    if 'unique_events' not in st.session_state:
        st.session_state.unique_events = []
    if 'selected_events' not in st.session_state:
        st.session_state.selected_events = []
    if 'merged_df' not in st.session_state:
        st.session_state.merged_df = None
    if 'name_matches' not in st.session_state:
        st.session_state.name_matches = []

    # Name standardization function
    def standardize_name(name):
        if pd.isna(name):
            return ""
        # Convert to string and clean
        name = str(name).strip()
        # Parse the name
        parsed = HumanName(name)
        # Get first name and standardize common variations
        first_name = parsed.first.lower()
        # Common first name variations
        name_variations = {
            'christopher': 'chris',
            'robert': 'bob',
            'william': 'bill',
            'james': 'jim',
            'james': 'jimmy',
            'michael': 'mike',
            'joseph': 'joe',
            'thomas': 'tom',
            'thomas': 'tommy',
            'richard': 'dick',
            'richard': 'rick',
            'charles': 'chuck',
            'charles': 'charlie',
            'daniel': 'dan',
            'daniel': 'danny',
            'edward': 'ed',
            'edward': 'eddie',
            'george': 'georgie',
            'henry': 'hank',
            'john': 'johnny',
            'john': 'jack',
            'lawrence': 'larry',
            'matthew': 'matt',
            'nicholas': 'nick',
            'nicholas': 'nicky',
            'patrick': 'pat',
            'patrick': 'paddy',
            'peter': 'pete',
            'peter': 'petey',
            'ronald': 'ron',
            'ronald': 'ronnie',
            'samuel': 'sam',
            'samuel': 'sammy',
            'stephen': 'steve',
            'stephen': 'stevie',
            'theodore': 'ted',
            'theodore': 'teddy',
            'timothy': 'tim',
            'timothy': 'timmy',
            'walter': 'walt',
            'walter': 'wally'
        }
        # Return standardized first name if it exists in variations, otherwise return original
        return name_variations.get(first_name, first_name)

    # Function to find similar names
    def find_similar_names(df, name_column, email_column=None, threshold=85):
        similar_names = []
        
        # Get unique names
        unique_names = df[name_column].unique()
        
        # Create a dictionary to store standardized names
        name_dict = {name: standardize_name(name) for name in unique_names if pd.notna(name)}
        
        # Compare each name with every other name
        for i, name1 in enumerate(unique_names):
            if pd.isna(name1):
                continue
            
            std_name1 = name_dict[name1]
            
            for name2 in unique_names[i+1:]:
                if pd.isna(name2):
                    continue
                
                std_name2 = name_dict[name2]
                
                # Calculate similarity scores
                ratio = fuzz.ratio(std_name1, std_name2)
                partial_ratio = fuzz.partial_ratio(std_name1, std_name2)
                
                # If email column exists, check if emails match
                email_match = False
                if email_column and email_column in df.columns:
                    email1 = df[df[name_column] == name1][email_column].iloc[0] if not df[df[name_column] == name1].empty else None
                    email2 = df[df[name_column] == name2][email_column].iloc[0] if not df[df[name_column] == name2].empty else None
                    email_match = pd.notna(email1) and pd.notna(email2) and email1 == email2
                
                # If either ratio is above threshold or emails match
                if ratio >= threshold or partial_ratio >= threshold or email_match:
                    similar_names.append({
                        'name1': name1,
                        'name2': name2,
                        'ratio': ratio,
                        'partial_ratio': partial_ratio,
                        'email_match': email_match,
                        'confidence': max(ratio, partial_ratio) if not email_match else 100
                    })
        
        return similar_names

    # Function to merge similar records
    def merge_similar_records(df, matches, name_column, email_column=None):
        merged_df = df.copy()
        
        # Create a mapping of names to their standardized versions
        name_mapping = {}
        for match in matches:
            if match['confidence'] >= 85:  # Only merge high confidence matches
                name_mapping[match['name2']] = match['name1']
        
        # Apply the mapping
        merged_df[name_column] = merged_df[name_column].map(lambda x: name_mapping.get(x, x))
        
        # Group by the standardized name and aggregate
        if email_column and email_column in merged_df.columns:
            merged_df = merged_df.groupby([name_column, email_column]).agg({
                col: lambda x: '; '.join(x.unique()) if x.dtype == 'object' else x.iloc[0]
                for col in merged_df.columns
                if col not in [name_column, email_column]
            }).reset_index()
        else:
            merged_df = merged_df.groupby(name_column).agg({
                col: lambda x: '; '.join(x.unique()) if x.dtype == 'object' else x.iloc[0]
                for col in merged_df.columns
                if col != name_column
            }).reset_index()
        
        return merged_df

    # Load data
    @st.cache_data
    def load_data(file):
        """Load and preprocess the Excel file."""
        try:
            # Read the Excel file
            df = pd.read_excel(file)
            
            # Convert date columns to datetime
            date_columns = df.select_dtypes(include=['object']).columns
            for col in date_columns:
                try:
                    # Try to parse with common date formats
                    df[col] = pd.to_datetime(df[col], format='mixed')
                except:
                    continue
            
            # Ensure we have at least one date column
            if not df.select_dtypes(include=['datetime64']).columns.any():
                st.error("No valid date columns found in the file. Please check the data format.")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    # Function to get unique names from the dataframe
    def get_unique_names(df):
        """Get unique names from the dataframe."""
        try:
            # Look for first name and last name columns
            first_name_cols = [col for col in df.columns if 'first name' in col.lower()]
            last_name_cols = [col for col in df.columns if 'last name' in col.lower()]
            
            if first_name_cols and last_name_cols:
                # Keep first and last names separate
                first_name_col = first_name_cols[0]
                last_name_col = last_name_cols[0]
                
                # Create a list of tuples with (first_name, last_name)
                name_pairs = list(zip(
                    df[first_name_col].astype(str),
                    df[last_name_col].astype(str)
                ))
                # Remove duplicates and sort
                unique_pairs = sorted(set(name_pairs))
                
                return unique_pairs, first_name_col, last_name_col
            else:
                # Fallback to any name column if first/last name columns not found
                name_columns = [col for col in df.columns if any(term in col.lower() for term in ['name', 'attendee', 'participant'])]
                if name_columns:
                    return [(name, None) for name in sorted(df[name_columns[0]].unique().tolist())], name_columns[0], None
                return [], None, None
        except Exception as e:
            st.error(f"Error getting unique names: {str(e)}")
            return [], None, None

    # Function to get unique events from the dataframe
    def get_unique_events(df):
        """Get unique events from the dataframe."""
        try:
            # Look specifically for Event Title column
            event_title_columns = [col for col in df.columns if 'event title' in col.lower()]
            if event_title_columns:
                events = df[event_title_columns[0]].dropna().unique().tolist()
                return sorted([str(event) for event in events if str(event).strip()])
            
            # Fallback to other event columns if Event Title not found
            event_columns = [col for col in df.columns if any(term in col.lower() for term in ['event', 'program', 'activity', 'session'])]
            if event_columns:
                events = df[event_columns[0]].dropna().unique().tolist()
                return sorted([str(event) for event in events if str(event).strip()])
            return []
        except Exception as e:
            st.error(f"Error getting unique events: {str(e)}")
            return []

    # Search section - always visible
    st.sidebar.header("Search")

    # File upload section - now mandatory
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=['xls', 'xlsx'])

    if uploaded_file is not None:
        try:
            st.session_state.df = load_data(uploaded_file)
            names, first_col, last_col = get_unique_names(st.session_state.df)
            st.session_state.unique_names = names
            st.session_state.first_name_col = first_col
            st.session_state.last_name_col = last_col
            st.session_state.unique_events = get_unique_events(st.session_state.df)
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}")
            if st.session_state.df is None:
                st.info("Please make sure the Excel file is in the correct format and contains the expected columns.")
                st.stop()
    else:
        st.info("Please upload an Excel file to begin analysis")
        st.stop()

    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Display data preview and column info
        with st.sidebar.expander("View Data Structure"):
            st.write("Columns in dataset:")
            for col in df.columns:
                st.write(f"- {col} ({df[col].dtype})")
        
        # Add search type selector
        search_type = st.sidebar.radio(
            "Search by:",
            ["Attendee Name", "Event Name"]
        )
        
        # Initialize search variables
        search_name = ""
        search_event = ""
        
        if search_type == "Attendee Name":
            if not st.session_state.unique_names:
                st.warning("No name columns found in the data. Please check your Excel file format.")
            else:
                # Create a list of display names for the dropdown
                display_names = [f"{first} {last}" if last else first for first, last in st.session_state.unique_names]
                search_name = st.selectbox("Select a member to analyze:", display_names)
                if search_name:
                    # Get the first and last name columns
                    first_name_col = st.session_state.first_name_col
                    last_name_col = st.session_state.last_name_col
                    
                    if first_name_col and last_name_col:
                        # Split the search name into first and last name
                        first_name, last_name = search_name.split(' ', 1)
                        
                        # Filter data for the selected name
                        attendee_data = st.session_state.df[
                            (st.session_state.df[first_name_col].astype(str) == first_name) &
                            (st.session_state.df[last_name_col].astype(str) == last_name)
                        ].copy()
                        
                        if not attendee_data.empty:
                            display_member_analysis(attendee_data, search_name)
                        else:
                            st.warning(f"No data found for member: {search_name}")
                    else:
                        st.warning("First name and last name columns not found in the data. Please check your Excel file format.")
                else:
                    st.info("Please select a member to analyze")
        elif search_type == "Event Name":
            if not st.session_state.unique_events:
                st.warning("No event columns found in the data. Please check your Excel file format.")
            else:
                # Create a DataFrame of events with their dates and locations
                event_title_columns = [col for col in st.session_state.df.columns if 'event title' in col.lower()]
                if not event_title_columns:
                    st.warning("Event Title column not found in the data. Please check your Excel file format.")
                else:
                    event_title_col = event_title_columns[0]
                    
                    # Get date columns
                    date_columns = st.session_state.df.select_dtypes(include=['datetime64']).columns
                    date_col = date_columns[0] if len(date_columns) > 0 else None
                    
                    # Get location columns
                    location_columns = [col for col in st.session_state.df.columns if any(term in col.lower() for term in ['location', 'venue', 'address', 'place'])]
                    location_col = location_columns[0] if location_columns else None
                    
                    # Create a DataFrame with event information
                    event_info = pd.DataFrame({
                        'Event': st.session_state.df[event_title_col].unique()
                    })
                    
                    # Add date information if available
                    if date_col:
                        event_info['Date'] = event_info['Event'].map(
                            lambda x: st.session_state.df[st.session_state.df[event_title_col] == x][date_col].iloc[0]
                            if not st.session_state.df[st.session_state.df[event_title_col] == x].empty else None
                        )
                    
                    # Add location information if available
                    if location_col:
                        event_info['Location'] = event_info['Event'].map(
                            lambda x: st.session_state.df[st.session_state.df[event_title_col] == x][location_col].iloc[0]
                            if not st.session_state.df[st.session_state.df[event_title_col] == x].empty else "Not specified"
                        )
                    
                    # Add search and sort options
                    col1, col2 = st.columns(2)
                    with col1:
                        search_text = st.text_input("Search events by name or location", "")
                    with col2:
                        sort_by = st.selectbox(
                            "Sort by:",
                            ["Name", "Date (Newest)", "Date (Oldest)", "Location"] if date_col else ["Name", "Location"]
                        )
                    
                    # Filter and sort the events
                    if search_text:
                        event_info = event_info[
                            event_info['Event'].str.contains(search_text, case=False, na=False) |
                            (location_col and event_info['Location'].str.contains(search_text, case=False, na=False))
                        ]
                    
                    if sort_by == "Name":
                        event_info = event_info.sort_values('Event')
                    elif sort_by == "Date (Newest)" and date_col:
                        event_info = event_info.sort_values('Date', ascending=False)
                    elif sort_by == "Date (Oldest)" and date_col:
                        event_info = event_info.sort_values('Date', ascending=True)
                    elif sort_by == "Location" and location_col:
                        event_info = event_info.sort_values('Location')
                    
                    # Event selection
                    search_event = st.selectbox("Select an event to analyze:", event_info['Event'].tolist())
                    
                    if search_event:
                        # Filter data for the selected event
                        event_data = st.session_state.df[st.session_state.df[event_title_col] == search_event].copy()
                        
                        if not event_data.empty:
                            try:
                                display_event_analysis(event_data, search_event)
                            except Exception as e:
                                st.error(f"Error analyzing event: {str(e)}")
                                st.info("Please check the data format and try again.")
                        else:
                            st.warning(f"No data found for event: {search_event}")
        else:
            st.info("Please select or type a name or event to see the analysis")
    else:
        st.info("Please upload an Excel file to begin analysis")

def display_member_analysis(attendee_data, search_name):
    """Display analysis for a selected member."""
    st.header(f"Analysis for {search_name}")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_events = len(attendee_data)
        st.metric("Total Events Attended", total_events)
    
    with col2:
        unique_dates = attendee_data.select_dtypes(include=['datetime64']).nunique().sum()
        st.metric("Unique Dates", unique_dates)
    
    with col3:
        # Calculate most common day of week
        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
        most_common_day = all_dates.dt.day_name().mode().iloc[0]
        st.metric("Most Common Day", most_common_day)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Event Timeline", "Day of Week Distribution", "Monthly Attendance", "Attendance Patterns"])
    
    with tab1:
        # Create timeline of events
        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
        all_dates = all_dates.dropna()
        
        # Create a more detailed timeline
        fig_timeline = go.Figure()
        
        # Add scatter plot for events
        fig_timeline.add_trace(go.Scatter(
            x=all_dates,
            y=[1] * len(all_dates),
            mode='markers',
            marker=dict(
                size=10,
                color='#1f77b4',
                symbol='circle'
            ),
            name='Events'
        ))
        
        # Update layout
        fig_timeline.update_layout(
            title="Event Timeline",
            xaxis_title="Date",
            yaxis_title="",
            showlegend=False,
            height=400,
            yaxis=dict(showticklabels=False),
            hovermode='x unified'
        )
        
        # Add hover text
        fig_timeline.update_traces(
            hovertemplate="Date: %{x}<br>Event<extra></extra>"
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        # Day of week distribution
        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
        all_dates = all_dates.dropna()
        day_dist = all_dates.dt.day_name().value_counts()
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_dist = day_dist.reindex(day_order)
        
        fig_days = px.bar(
            x=day_dist.index,
            y=day_dist.values,
            title="Attendance by Day of Week",
            color=day_dist.values,
            color_continuous_scale='Viridis'
        )
        
        fig_days.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Number of Events",
            height=400
        )
        
        st.plotly_chart(fig_days, use_container_width=True)
    
    with tab3:
        # Monthly attendance
        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
        all_dates = all_dates.dropna()
        monthly_counts = all_dates.dt.to_period('M').value_counts().sort_index()
        
        fig_monthly = px.bar(
            x=monthly_counts.index.astype(str),
            y=monthly_counts.values,
            title="Monthly Attendance",
            color=monthly_counts.values,
            color_continuous_scale='Viridis'
        )
        
        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Events",
            height=400
        )
        
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab4:
        # Create a heatmap of attendance patterns
        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
        all_dates = all_dates.dropna()
        
        # Create a DataFrame with day of week and month
        attendance_patterns = pd.DataFrame({
            'date': all_dates,
            'day_of_week': all_dates.dt.day_name(),
            'month': all_dates.dt.month_name()
        })
        
        # Create pivot table for heatmap
        pivot_table = pd.pivot_table(
            attendance_patterns,
            values='date',
            index='day_of_week',
            columns='month',
            aggfunc='count'
        ).fillna(0)
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(day_order)
        
        # Create heatmap
        fig_heatmap = px.imshow(
            pivot_table,
            title="Attendance Patterns by Day and Month",
            aspect='auto',
            color_continuous_scale='Viridis'
        )
        
        fig_heatmap.update_layout(
            xaxis_title="Month",
            yaxis_title="Day of Week",
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Display raw data with better formatting
    st.subheader("Event Details")
    st.dataframe(
        attendee_data,
        use_container_width=True,
        hide_index=True
    )

def display_event_analysis(event_data, search_event):
    """Display analysis for a selected event."""
    try:
        st.header(f"Analysis for {search_event}")
        
        # Get event dates - find the first datetime column
        date_columns = event_data.select_dtypes(include=['datetime64']).columns
        if len(date_columns) == 0:
            st.error("No date columns found in the event data")
            return
        
        event_dates = event_data[date_columns[0]]
        
        # Find location column
        location_columns = [col for col in event_data.columns if any(term in col.lower() for term in ['location', 'venue', 'address', 'place'])]
        location_info = ""
        if location_columns:
            location_info = event_data[location_columns[0]].iloc[0] if not event_data[location_columns[0]].empty else "Location not specified"
        
        # Display event date and location
        col1, col2 = st.columns(2)
        with col1:
            if not event_dates.empty:
                event_date = event_dates.iloc[0]
                st.write("**Event Date:**", event_date.strftime('%B %d, %Y'))
            else:
                st.write("**Event Date:** Not specified")
        
        with col2:
            st.write("**Location:**", location_info)
        
        # Add a separator
        st.markdown("---")
        
        # Calculate comprehensive statistics
        stats = calculate_event_stats(event_data, event_dates)
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Attendees", stats['total_attendees'])
            st.metric("Average Attendance", f"{stats['avg_attendance']:.1f}")
        
        with col2:
            st.metric("Number of Occurrences", stats['occurrences'])
            st.metric("Peak Attendance", stats['peak_attendance'])
        
        with col3:
            st.metric("Attendance Trend", stats['attendance_trend'])
            st.metric("Consistency", stats['consistency'])
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Attendance Over Time", "Attendee List", "Event Details"])
        
        with tab1:
            # Create attendance timeline
            attendance_counts = event_data.groupby(event_dates).size()
            
            fig_timeline = go.Figure()
            
            # Add line plot
            fig_timeline.add_trace(go.Scatter(
                x=attendance_counts.index,
                y=attendance_counts.values,
                mode='lines+markers',
                name='Attendance'
            ))
            
            # Add trend line
            if len(attendance_counts) > 1:
                x = np.arange(len(attendance_counts))
                z = np.polyfit(x, attendance_counts.values, 1)
                p = np.poly1d(z)
                fig_timeline.add_trace(go.Scatter(
                    x=attendance_counts.index,
                    y=p(x),
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash')
                ))
            
            fig_timeline.update_layout(
                title="Attendance Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Attendees",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Add attendance statistics
            st.subheader("Attendance Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if stats['peak_date']:
                    st.write("Peak Date:", stats['peak_date'].strftime('%Y-%m-%d'))
                st.write("Standard Deviation:", f"{stats['attendance_std']:.1f}")
            
            with col2:
                st.write("Trend Strength:", f"{stats['trend_strength']:.2f}")
                st.write("Consistency Level:", stats['consistency'])
            
            with col3:
                st.write("Growth Rate:", f"{stats['attendance_trend']}")
                st.write("Total Unique Attendees:", len(event_data))
        
        with tab2:
            # Get unique attendees for this event
            first_name_col = None
            last_name_col = None
            
            # Look for first name and last name columns
            for col in event_data.columns:
                if 'first name' in col.lower():
                    first_name_col = col
                elif 'last name' in col.lower():
                    last_name_col = col
            
            if first_name_col and last_name_col:
                st.subheader("Attendees")
                
                # Create a DataFrame with attendee statistics
                # First, get the attendance counts for each name combination
                attendance_counts = event_data.groupby([first_name_col, last_name_col]).size().reset_index(name='count')
                
                # Create the final DataFrame with all required columns
                attendee_stats = pd.DataFrame({
                    'First Name': attendance_counts[first_name_col],
                    'Last Name': attendance_counts[last_name_col],
                    'Times Attended': attendance_counts['count'],
                    'Attendance Rate': (attendance_counts['count'] / stats['occurrences'] * 100).round(1)
                })
                
                # Sort by last name, then first name
                attendee_stats = attendee_stats.sort_values(['Last Name', 'First Name'])
                
                # Display as a table
                st.dataframe(
                    attendee_stats,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # Fallback to single name column if first/last name columns not found
                name_column = None
                for col in event_data.columns:
                    if event_data[col].dtype == 'object' or event_data[col].dtype == 'string':
                        if event_data[col].astype(str).str.replace(' ', '').str.isalpha().all():
                            name_column = col
                            break
                
                if name_column:
                    st.subheader("Attendees")
                    
                    # Add attendance frequency for each attendee
                    attendee_freq = event_data[name_column].value_counts()
                    
                    # Create a DataFrame with attendee statistics
                    attendee_stats = pd.DataFrame({
                        'Name': attendee_freq.index,
                        'Times Attended': attendee_freq.values,
                        'Attendance Rate': (attendee_freq.values / stats['occurrences'] * 100).round(1)
                    })
                    
                    # Sort by name
                    attendee_stats = attendee_stats.sort_values('Name')
                    
                    # Display as a table
                    st.dataframe(
                        attendee_stats,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("Could not identify the attendee column")
        
        with tab3:
            # Display event details
            st.subheader("Event Details")
            st.dataframe(
                event_data,
                use_container_width=True,
                hide_index=True
            )
    except Exception as e:
        st.error(f"Error displaying event analysis: {str(e)}")
        st.info("Please try selecting a different event or check the data format.")

if __name__ == "__main__":
    main() 