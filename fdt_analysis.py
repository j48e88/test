import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
# Set page configuration
st.set_page_config(page_title="Crew Estimation", page_icon=":bar_chart", layout="wide")

st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>This app is used for crew estimation only.</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>The crew is calculated according to the aircraft types :</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>For passenger flights, <b><span style='color: blue;'>A320</b> will need <b><span style='color: red;'>7</b> crews ; <b><span style='color: blue;'> A330</b> will need <b><span style='color: red;'>11</b> crews </p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>For cargo flights, both will need <b><span style='color: red;'>4</b> crews.</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>Turnaround =<b><span style='color: red;'>1</b> crew team perform <b><span style='color: red;'>2</b> flights.</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>Layover =<b><span style='color: red;'>1</b> crew team perform <b><span style='color: red;'>1</b> flight.</p>", unsafe_allow_html=True)


st.title("Flight Data Analysis")
# Add a file uploader to allow the user to upload an Excel /CSV file
uploaded_file = st.file_uploader("Upload the Flight Schedule file", type=["xlsx", "csv"])


# Define a function to preprocess the data
def preprocess_data(df):
    # Drop all rows that have a 'CNL' status from the original DataFrame
    df.drop(df[df['Status'] == 'CNL'].index, inplace=True)

    # split the "STD/ETD" column into two columns
    df[['STD', 'ETD']] = df['STD/ETD'].str.split('/', expand=True)
    df[['STA', 'ETA']] = df['STA/ETA'].str.split('/', expand=True)
    # drop the original "STD/ETD" column
    df.drop('STD/ETD', axis=1, inplace=True)
    df.drop('STA/ETA', axis=1, inplace=True)

    # Convert the 'Type' column to string data type
    df['Type'] = df['Type'].astype(str)
    # Create a new column 'ac_type' containing the first two characters of the 'Type' column
    df['ac_type'] = df['Type'].apply(lambda x: x[:2] if len(x) >= 2 else None)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # Convert the 'Date' column to the desired format
    df['Date'] = df['Date'].dt.strftime('%d%b')

    df['STD'] = df['STD'].str.strip()
    stds = pd.to_datetime(df['STD'], format='%H:%M')
    stds = stds.dropna()
    df['STA'] = df['STA'].str.strip()
    stas = pd.to_datetime(df['STA'], format='%H:%M')
    stas = stas.dropna()
    BlockOff = pd.to_datetime(df['STD'], format='%H:%M')
    BlockOn = pd.to_datetime(df['STA'], format='%H:%M')
    BlockOn = BlockOn + pd.DateOffset(days=1)
    # Calculate report time
    report = np.where(df['DepStn'] == "HKG",
                      np.where(df['STC'] == "J",
                               (BlockOff - pd.Timedelta(minutes=75)).dt.strftime('%H:%M').str.split().str[-1],
                               (BlockOff - pd.Timedelta(minutes=65)).dt.strftime('%H:%M').str.split().str[-1]),
                      (BlockOff - pd.Timedelta(minutes=60)).dt.strftime('%H:%M').str.split().str[-1])

    # Calculate postflight time
    postflight = BlockOn + pd.DateOffset(minutes=30)
    postflight = pd.to_datetime(postflight.dt.strftime('%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
    postflight_str = postflight.dt.strftime('%H:%M').str.split().str[-1]

    # Calculate time difference between report and postflight
    diff = postflight - pd.to_datetime(report, format='%H:%M')
    diff_str = diff.astype(str).str.split().str[-1]
    df.insert(15, "Diff", diff_str)

    return df

def parse_date(date_str):
    if isinstance(date_str, str):
        return pd.datetime.strptime(date_str, '%d%b')
    else:
        return date_str

if uploaded_file is not None:
    # Read the uploaded file and preprocess the data
    df = pd.read_excel(uploaded_file, skiprows=3, parse_dates=['Date'], date_parser=parse_date)
    df = df.dropna()
    if 'ac_type' in df.columns:
        df['ac_type'] = df['ac_type'].astype(str).tolist()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    stc = df['STC']
    df = df.rename(columns={'Flight No': 'Flight_No'})
    df = preprocess_data(df)

    # Display the preprocessed data and allow the user to download it as an Excel file
    st.write("The reversed excel data is as follow:")
    st.write(df)

    def calculate_num_layover(count, dep, arr):
        dep, arr = sorted([dep, arr])
        if (count % 2 == 1 and arr != "HKG"):
            return (count - 1) // 2
        else:
            return 0

    def calculate_num_nonregular(groups, date):
        non_regular_count = 0
        for group, count in groups.items():
            if group[0] == date and group[2] == ac_type:
                dep = group[1][0]
                arr = group[1][1]
                if dep == 'HKG' and arr == 'HKG':
                    non_regular_count += count
        return non_regular_count

    groups = {}
    ac_32 = []
    ac_33 = []

    for index, row in df.iterrows():
        dep = row['DepStn']
        arr = row['ArrStn']
        ac_type = row['ac_type']
        group = (datetime.strptime(row['Date'], '%d%b').strftime('%d%b'), tuple(sorted([dep, arr])), row['ac_type'])
        if group not in groups:
            groups[group] = 1
        else:
            groups[group] += 1
        if row['ac_type'] == '32':
            ac_32.append(row['Flight_No'])
        elif row['ac_type'] == '33':
            ac_33.append(row['Flight_No'])

    # Group flights by aircraft and date
    ac_dates = {}
    for group, count in groups.items():
        ac_type = group[2]
        date = group[0]
        if ac_type not in ac_dates:
            ac_dates[ac_type] = {}
        if date not in ac_dates[ac_type]:
            ac_dates[ac_type][date] = []
        ac_dates[ac_type][date].append(count)

    # Calculate number of turnarounds for each group
    num_turnarounds = {} 
    for ac_type, dates in ac_dates.items():
        for date, counts in dates.items():
            num_turnarounds[(date, ac_type)] = sum([count // 2 for count in counts])

    # Calculate number of layovers for each group
    num_layovers = {}
    for group, count in groups.items():
        dep = group[1][0]
        arr = group[1][1]
        ac_type = group[2]
        num_layovers[(group[0], ac_type)] = calculate_num_layover(count, dep, arr)

    # Calculate number of non-regular flights for each date
    num_nonregular = {}
    for group, count in groups.items():
        date = group[0]
        ac_type = group[2]
        num_nonregular[(group[0], ac_type)] = calculate_num_nonregular(groups, date)

    def calculate_crew_num(ac_type, num_turnaround, num_layover, is_nonreg=False, num_nonreg=0):
        if is_nonreg:
            return 4 * num_nonreg + 7 * (num_turnaround + num_layover) if ac_type == '32' else 11 * (num_turnaround + num_layover)
        elif ac_type == '32':
            return 7 * (num_turnaround + num_layover)
        elif ac_type == '33':
            return 11 * (num_turnaround + num_layover)
        else:
            return None

    # Calculate total number of crew
    total_crew_num = 0
    for (date, ac_type), num_turnaround in num_turnarounds.items():
        num_layover = num_layovers[(date, ac_type)]
        num_nonreg = num_nonregular[(date, ac_type)]
        if row['STC'] == 'J':
            if num_nonreg > 0:
                num_crew = calculate_crew_num(ac_type, num_turnaround, num_layover, is_nonreg=True, num_nonreg=num_nonreg)
            else:
                num_crew = calculate_crew_num(ac_type, num_turnaround, num_layover)
            if num_crew is not None:
                total_crew_num += num_crew
        elif row['STC'] != 'J':
            num_crew = 4

    # count the number of unique dates in the DataFrame
    num_dates = len(df['Date'].unique())

    # calculate the average crew number per day
    avg_crew_per_day = round(total_crew_num / num_dates)

    # Calculate number of crew on each day
    daily_crew_nums = {}
    for (date, ac_type), num_turnaround in num_turnarounds.items():
        num_layover = num_layovers[(date, ac_type)]
        num_nonreg = num_nonregular[(date, ac_type)]
        if num_nonreg > 0:
            num_crew = calculate_crew_num(ac_type, num_turnaround, num_layover, is_nonreg=True, num_nonreg=num_nonreg)
        else:
            num_crew = calculate_crew_num(ac_type, num_turnaround, num_layover)
        if num_crew is not None:
            daily_crew_nums[date] = daily_crew_nums.get(date, 0) + num_crew 

    # Print results sorted by date
    # Get the list of flight groups sorted by date
    flight_groups = sorted(groups.keys(), key=lambda x: x[2])

    # Create a list of unique dates from the flight groups
    unique_dates = sorted(list(set([group[0] for group in flight_groups])))

    # Create a list of options for the selectbox
    options = unique_dates

    # Display the selectbox in the sidebar
    st.sidebar.write("----------")
    st.sidebar.markdown("<h1 style='text-align: center; color: black; font-size: 28px;'>---For Flight Analysis---</h1>", unsafe_allow_html=True)

    selected_date = st.sidebar.selectbox('Date (For Flights stations per day(s))', options = options)

    # Filter the flight groups by the selected date
    filtered_groups = [
        (group[1], count)
        for group, count in groups.items()
        if datetime.strptime(group[0], '%d%b').strftime('%d%b') == selected_date]
    
    # Display the filtered flight groups
    with st.container():
        st.write("-----------------------------")
        st.markdown("<h1 style='text-align: left; color: black; font-size: 28px;'>The daliy flight stations are shown:</h1>", unsafe_allow_html=True)
        # Create a placeholder for the content
        content_placeholder = st.empty()
        # Add a button to show or hide the content
        show_content = st.checkbox("Show the flight stations per day")
            # Update the content based on the checkbox value
        if show_content:
            for count in sorted(filtered_groups):
                st.write(count)
        else:
            content_placeholder.empty()  # Hide the content
    st.markdown(f"<p style='font-size: 20px;'>On date: <b><span style='color: blue;'>{selected_date}</b>.</p>", unsafe_allow_html=True)

    # Create a dictionary containing all flight information, keyed by date
    flight_info_by_date = {}
    for (date, ac_type), num_turnaround in sorted(num_turnarounds.items()):
        num_layover = num_layovers[(date, ac_type)]
        num_nonreg = num_nonregular[(date, ac_type)]
        if date not in flight_info_by_date:
            flight_info_by_date[date] = []
        flight_info_by_date[date].append(f"Aircraft Type: **{ac_type}** : **{num_turnaround}** turnarounds, **{num_layover}** layovers, **{num_nonreg}** non-regular flights")        

    # Current page date
    current_page_date = st.sidebar.selectbox('Date (For Types of Flights)', sorted(flight_info_by_date.keys()))

    # Calculate the data range for the current page
    start_index = 0
    end_index = len(flight_info_by_date[current_page_date])
    page_flight_info = flight_info_by_date[current_page_date][start_index:end_index]
    total_turnaround = sum(num_turnarounds.values())
    total_layover = sum(num_layovers.values())
    total_nonreg = sum(num_nonregular.values())

    # Display the current page's content
    st.write("-----------------------------")
    st.markdown("<h1 style='text-align: left; color: black; font-size: 28px;'>Types of Flights</h1>", unsafe_allow_html=True)
    for item in page_flight_info:
        st.write(item)

    # Display pagination information
    st.markdown(f"<p style='font-size: 18px;'> On date: <b><span style='color: blue;'>{current_page_date}</b>.</p>", unsafe_allow_html=True)

    # Create a placeholder for the content
    content_placeholder = st.empty()

    # Add a button to show or hide the content
    show_content = st.checkbox("Show the total Number of flights in this month:")

    # Update the content based on the checkbox value
    if show_content:
        st.markdown(f"<p style='font-size: 20px;'>Turnaround flights: <b><span style='color: red;'>{total_turnaround}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px;'>Layover flights: <b><span style='color: red;'>{total_layover}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px;'>Non-Regular Flights: <b><span style='color: red;'>{total_nonreg}</b></p>", unsafe_allow_html=True)
    else:
        content_placeholder.empty()  # Hide the content


    # Create a list of crew information, sorted by date
    data = []
    for date, crew_num in sorted(daily_crew_nums.items()):
        avg_crew = avg_crew_per_day
        diff = avg_crew - crew_num
        data.append({'date': date, 'Required Crew': crew_num, 'Crew Difference': diff})
    # Create a line chart using plotly
    fig = px.line(df, x='date', y='Crew Difference')
    # find the date and required crew number with the maximum/minimum crew number
    max_crew_num = max(data, key=lambda x: x['Required Crew'])['Required Crew']
    min_crew_num = min(data, key=lambda x: x['Required Crew'])['Required Crew']
    max_crew_dates = [info['date'] for info in data if info['Required Crew'] == max_crew_num]
    min_crew_dates = [info['date'] for info in data if info['Required Crew'] == min_crew_num]

    # create a new DataFrame with the minimum and maximum crew numbers and the corresponding dates
    crew_table = pd.DataFrame({'Date': max_crew_dates + min_crew_dates,
                               'Minimum Required Crew': [min_crew_num if date in min_crew_dates else None for date in max_crew_dates + min_crew_dates],
                               'Maximum Required Crew': [max_crew_num if date in max_crew_dates else None for date in max_crew_dates + min_crew_dates]})


    # calculate the crew differences and add new columns to the crew_table DataFrame
    crew_table['Over manpower'] = avg_crew_per_day - crew_table['Minimum Required Crew']
    crew_table['Lack manpower'] = avg_crew_per_day - crew_table['Maximum Required Crew']

    df_data = pd.DataFrame(data)
    df_data = df_data.set_index('date')

    # Number of rows to display per page
    rows_per_page = 7
    # Total number of pages
    total_pages = len(data) // rows_per_page + 1
    # Current page index
    current_page_index = st.sidebar.number_input("Page  (To review the crews required on each day)", min_value=1, max_value=total_pages, value=1, step=1) - 1

    # Calculate the data range for the current page
    start_index = current_page_index * rows_per_page
    end_index = start_index + rows_per_page
    page_crew_info = data[start_index:end_index]
    page_data = df_data.iloc[start_index:end_index].copy()
    # Reset the index and rename the "Required Crew" column
    page_data = page_data.reset_index()
    page_data = page_data.rename(columns={"crew_num": "Required Crew"})
    # Display the current page's content
    st.write("-----------------------------")
    st.markdown("<h1 style='text-align: left; color: black; font-size: 30px;'>The number of crews requied on each day is:</h1>", 
                unsafe_allow_html=True)
    # Create a placeholder for the content
    content_placeholder1= st.empty()
    content_placeholder2 = st.empty()

    # Add a button to show or hide the content
    show_content1 = st.checkbox("Show the required crew according to the date")


    # Update the content based on the checkbox value
    if show_content1:
        st.dataframe(pd.DataFrame(page_data), height=280)
        st.plotly_chart(fig)
        show_content2 = st.checkbox("Show the minimum or maximum number")  
        if show_content2:
            # print the crew information and the date and required crew number with the maximum crew number
            st.write(crew_table)
        else:
            content_placeholder2.empty()
        # Display pagination information
        st.write(f"Page {current_page_index+1} of {total_pages}")

    else:
        content_placeholder1.empty()  # Hide the content

