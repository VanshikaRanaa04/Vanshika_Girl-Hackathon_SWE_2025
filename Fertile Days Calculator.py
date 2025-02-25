import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="Conception Planner", layout="centered")

st.markdown(
    """
    <style>
        body {
            background-color: #FFF5F5;
        }
        .stApp {
            background-color: #FFF5F5;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            width: 50%;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin: auto;
        }
        .stTextInput, .stNumberInput, .stButton {
            width: 100%;
            margin-bottom: 15px;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF80AB;
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FF4081;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        .results-section {
            text-align: left; /* Align results to the left */
            margin-top: 20px;
        }
        .results-section h3, .results-section p {
            margin-bottom: 5px; /* Add some spacing */
        }
        .results-section p strong {
            font-size: 16px; /* Match input label size */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>Conception Planner</h1>", unsafe_allow_html=True)

# Input fields 
start_date = st.date_input("Enter the first day of your last period (YYYY/MM/DD):")
cycle_length = st.number_input("Enter your average cycle length (in days, e.g., 28):", min_value=20, max_value=40)

# Calculate button
calculate = st.button("Calculate")


# logic
if calculate:
    try:
        # Convert start date to datetime 
        start_date_conversion = datetime.combine(start_date, datetime.min.time())
        
        # Calculate ovulation day and fertile window
        ovulation_day = start_date_conversion + timedelta(days=cycle_length - 14)
        fertile_window_begin = ovulation_day - timedelta(days=4)
        fertile_window_end = ovulation_day + timedelta(days=1)

        # Display results with consistent styling
        st.markdown("<div class='results-section'>", unsafe_allow_html=True) 
        
        # Fertile Window display
        st.markdown("Patient's Fertile Window:", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:16px'>{fertile_window_begin.date()} to {fertile_window_end.date()}</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True) 

    except ValueError:
        st.error("Please enter the date in the correct YYYY/MM/DD format.")
