import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
# Set page configuration
st.set_page_config(page_title="Crew Estimation", page_icon=":bar_chart", layout="wide")

st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>This app is used for crew estimation.</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>The crew is calculated according to the aircraft types :</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>For passenger flights, <b><span style='color: blue;'>A320</b> will need <b><span style='color: red;'>7</b> crews ; <b><span style='color: blue;'> A330</b> will need <b><span style='color: red;'>11</b> crews </p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>For cargo flights, both will need <b><span style='color: red;'>4</b> crews.</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>Turnaround =<b><span style='color: red;'>1</b> crew team perform <b><span style='color: red;'>2</b> flights.</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey; font-size: 20px;'>Layover =<b><span style='color: red;'>1</b> crew team perform <b><span style='color: red;'>1</b> flight.</p>", unsafe_allow_html=True)


st.title("Flight Data Analysis")
# Add a file uploader to allow the user to upload an Excel file
uploaded_file2 = st.file_uploader("Upload the Flight Schedule file", type=["xlsx"])


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


if uploaded_file2 is not None:
    # Read the uploaded file and preprocess the data
    df = pd.read_excel(uploaded_file2, skiprows=3, parse_dates=['Date'], date_parser=parse_date)
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

    # Number of rows to display per pages
    rows_per_page2 = 10

    # Total number of pages
    total_pages2 = len(flight_info_by_date) // rows_per_page2 + 1

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
    crew_info = []
    for date, crew_num in sorted(daily_crew_nums.items()):
        crew_info.append(f"{date} : Required Crew = {crew_num}")
    data = []
    for info in crew_info:
        date = info.split(" : ")[0]
        crew_num = int(info.split(" = ")[1])
        data.append({'date': date, 'Required Crew': crew_num})

    df_data = pd.DataFrame(data)
    df_data = df_data.set_index('date')

    # Number of rows to display per page
    rows_per_page = 5

    # Total number of pages
    total_pages = len(crew_info) // rows_per_page + 1

    # Current page index
    current_page_index = st.sidebar.number_input("Page  (To review the crews required on each day)", min_value=1, max_value=total_pages, value=1, step=1) - 1

    # Calculate the data range for the current page
    start_index = current_page_index * rows_per_page
    end_index = start_index + rows_per_page
    page_crew_info = crew_info[start_index:end_index]
    page_data = df_data.iloc[start_index:end_index].copy()
    # Reset the index and rename the "Required Crew" column
    page_data = page_data.reset_index()
    page_data = page_data.rename(columns={"crew_num": "Required Crew"})
    # Display the current page's content
    st.write("-----------------------------")
    st.markdown("<h1 style='text-align: left; color: black; font-size: 30px;'>The number of crews requied on each day is:</h1>", unsafe_allow_html=True)
    # Create a placeholder for the content
    content_placeholder = st.empty()

    # Add a button to show or hide the content
    show_content = st.checkbox("Show the required crew according to the date")

    # Update the content based on the checkbox value
    if show_content:
        st.dataframe(pd.DataFrame(page_data), height=210)  

    # Display pagination information
        st.write(f"Page {current_page_index+1} of {total_pages}")
    else:
        content_placeholder.empty()  # Hide the content

    # Update start and end indices based on the current page index
    start_index = current_page_index * rows_per_page
    end_index = start_index + rows_per_page

    # Display the total number of crew needed and aircraft type counts
    if st.button("Show the estimated crew number"):
        st.markdown("<h1 style='text-align: left; color: black; font-size: 25px;'>Total Num of Crew Needed:</h1>", unsafe_allow_html=True)
        st.write(f"{total_crew_num}")
        st.markdown("<h1 style='text-align: left; color: black; font-size: 25px;'>Aircraft Types Counts:</h1>", unsafe_allow_html=True)
        st.write("\nType Series 320: ", len(ac_32))
        st.write("\nType Series 330: ", len(ac_33))
        
    fdp_rules = {
    "0700-0759": {1: 13, 2: 12.25, 3: 11.5, 4: 10.75, 5: 10, 6: 9.25, 7: 9, 8: 9},
    "0800-1259": {1: 14, 2: 13.25, 3: 12.5, 4: 11.75, 5: 11, 6: 10.25, 7: 9.5, 8: 9},
    "1300-1759": {1: 13, 2: 12.25, 3: 11.5, 4: 10.75, 5: 10, 6: 9.25, 7: 9, 8: 9},
    "1800-2159": {1: 12, 2: 11.25, 3: 10.5, 4: 9.75, 5: 9, 6: 9, 7: 9, 8: 9},
    "2200-0659": {1: 11, 2: 10.25, 3: 9.5, 4: 9, 5: 9, 6: 9, 7: 9, 8: 9}
    }

    df = df.dropna()
    departures = df['DepStn'].tolist()
    arrivals = df['ArrStn'].tolist()
    # Remove trailing spaces from the 'STD' column
    df['STD'] = df['STD'].str.strip()
    # Parse the 'STD' column as datetime
    stds = pd.to_datetime(df['STD'], format='%H:%M')
    stds = stds.dropna()

    # Remove trailing spaces from the 'STD' column
    df['STA'] = df['STA'].str.strip()
    # Parse the 'STD' column as datetime
    stas = pd.to_datetime(df['STA'], format='%H:%M')
    stas = stas.dropna()
    aircraft_types = df['Type'].tolist()
    flight_no = df['Flight_No'].tolist()
    reg = df['Reg'].str.upper().astype(str).tolist()
    df['date'] = pd.to_datetime(df['Date'], format='%d%b')
    date = df['date']

    reporting_time = [(std - timedelta(minutes=75)).strftime('%H%M') for std in stds]
    fdt = pd.to_datetime(df['Diff'], format='%H:%M:%S')
    fdt_decimal = fdt.dt.hour + fdt.dt.minute/60 + fdt.dt.second/3600
    df['diff decimal'] = fdt_decimal
    diff_sum = round(df['diff decimal'].sum(), 2)
    n_flights = len(departures)

    valid_flights = set()  # set to store flights with a valid connection
    checked = set()  # set to store checked flights
    valid_count = 0  # counter for valid flights
    invalid_count = 0
    # create variables to store the total valid count and total invalid count
    total_valid_count = 0
    total_invalid_count = 0
    turn = []
    lay = []
    grouped_data = df.groupby('Date')

    if reporting_time:
        time_ranges = []
        for rt in reporting_time:
            rt_obj = datetime.strptime(rt, '%H%M')
            time_0700 = datetime.strptime("07:00", "%H:%M")
            time_0800 = datetime.strptime("08:00", "%H:%M")
            time_1300 = datetime.strptime("13:00", "%H:%M")
            time_1800 = datetime.strptime("18:00", "%H:%M")
            if time_0700 <= rt_obj <= datetime.strptime("07:59", "%H:%M"):
                time_ranges.append("0700-0759")
            elif time_0800 <= rt_obj <= datetime.strptime("12:59", "%H:%M"):
                time_ranges.append("0800-1259")
            elif time_1300 <= rt_obj <= datetime.strptime("17:59", "%H:%M"):
                time_ranges.append("1300-1759")
            elif time_1800 <= rt_obj <= datetime.strptime("21:59", "%H:%M"):
                time_ranges.append("1800-2159")
            else:
                time_ranges.append("2200-0659")

        df['Time_Range'] = time_ranges

    st.write('\n')
    # Get the list of unique ArrStn and DepStn for populating the selectbox
    stations = df[['ArrStn', 'DepStn']].stack().unique()

    # add a selection box to the sidebar to filter the data by date
    selected_date = st.sidebar.selectbox("Select a date (For the flight types details)", grouped_data.groups.keys())

    # Add a selectbox on the sidebar for selecting the station
    selected_station = st.sidebar.selectbox('Select Station (HKG will show all the flight pairs)', stations)


    # Filter the data for the selected station
    filtered_data = df[(df['ArrStn'] == selected_station) | (df['DepStn'] == selected_station)]

    # Show the turnaround / layover flight for the selected stations
    st.write("-----------------------------")
    st.markdown("<h1 style='text-align: left; color: black; font-size: 25px;'>Here are the details regarding the flight types:</h1>", unsafe_allow_html=True)
    # create an empty list to store the data for each date
    data_list = []
    # Create a placeholder for the content
    content_placeholder = st.empty()
    # Add a button to show or hide the content
    show_content = st.checkbox("Show the results")
    # Update the content based on the checkbox value
    if show_content:

        # filter the data by the selected date
        filtered_data = grouped_data.get_group(selected_date)

        # allow the user to show the valid connections by clicking a button
        data = []
        used_flights_on_date = set() # initialize set to track used flight numbers on the same day
        for i in range(len(filtered_data)):
            flight1 = filtered_data.iloc[i]
            valid_connection = False

            # check for valid connections
            for j in range(i+1, len(df)):
                flight2 = df.iloc[j]
                if (flight1['ArrStn'] == flight2['DepStn'] and flight1['DepStn'] == flight2['ArrStn'] and
                    flight1['Flight_No'] != flight2['Flight_No'] and
                    flight1['Date'] == flight2['Date'] and
                    all(flight1['Flight_No'] not in x and flight2['Flight_No'] not in x for x in valid_flights) and
                    flight1['Flight_No'] not in used_flights_on_date and flight2['Flight_No'] not in used_flights_on_date):
                    valid_connection = True
                    sum_fdt = round(flight1['diff decimal'] + flight2['diff decimal'], 2)
                    connection = (flight1['DepStn'], flight1['ArrStn'], flight2['DepStn'], flight2['ArrStn'])
                    valid_flights.add(f"{flight1['Flight_No']} and {flight2['Flight_No']}")
                    valid_count += 1
                    turn.append(flight2['Flight_No'])
                    fdp = fdp_rules[flight1['Time_Range']][2] if flight2['Flight_No'] in turn else fdp_rules[flight1['Time_Range']][1]
                    remaintime = round(fdp - sum_fdt ,2)
                    data.append([flight1['Flight_No'], flight1['DepStn'], flight1['ArrStn'],
                                flight2['Flight_No'], flight2['DepStn'], flight2['ArrStn'],
                                round(sum_fdt, 2), round(fdp, 2), round(remaintime, 2), "Turnaround"])
                    used_flights_on_date.add(flight1['Flight_No']) # add flight numbers to set of used flights on the same day
                    used_flights_on_date.add(flight2['Flight_No'])

            if not valid_connection and all(flight1['Flight_No'] not in x for x in valid_flights):
                sum_fdt = round(flight1['diff decimal'], 2)
                invalid_count += 1
                lay.append(flight1['Flight_No'])
                fdp = fdp_rules[flight1['Time_Range']][1] if flight1['Flight_No'] in lay else fdp_rules[flight1['Time_Range']][2]
                data.append([flight1['Flight_No'], flight1['DepStn'], flight1['ArrStn'],
                            "", "", "",
                            round(sum_fdt, 2), round(fdp, 2), round(fdp-sum_fdt, 2), "Layover"])

        if len(data) > 0:
            # create a DataFrame from the data list
            df_data = pd.DataFrame(data, columns=["Flight 1", "DepStn 1", "ArrStn 1", "Flight 2", "DepStn 2", "ArrStn 2", "Sum FDT", "FDP", "Remaining Time", "Type"])
            # display the DataFrame as a table using st.write
            st.write(df_data)
            # add the data to the list for the current date
            data_list.append(df_data)
        else:
            st.write("No valid connections found.")

        # display the number of valid connections to the user
        total_count_day = len(filtered_data)
        total_count_month = len(df)
        st.markdown(f"<p style='font-size: 20px;'>There are in total <b><span style='color: red;'>{total_count_day}</b> flights on <b><span style='color: blue;'>{selected_date}</b>.</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px;'><b><span style='color: red;'>{valid_count}</b> pairs of flight(s) is/are turnaround.</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px;'><b><span style='color: red;'>{invalid_count}</b> flight(s) is/are layover.</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-family: Arial; font-size: 20px;'>There are in total <b><span style='color: red;'>{total_count_month}</b> flights in this month.</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 20px;'>You have selected date: <b><span style='color: blue;'>{selected_date}</b>.</p>", unsafe_allow_html=True)

    else:
        content_placeholder.empty()  # Hide the content
