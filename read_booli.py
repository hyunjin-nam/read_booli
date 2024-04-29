import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time


def main_page(page):
    request = requests.get(page)
    soup = BeautifulSoup(request.text, 'lxml')
    links = soup.select("li[class*=search-page__module-container]")

    data = []

    for j, row in enumerate(links):
        price_element = row.select_one(".heading-3")  # Select the price element
        area_element = row.select_one(
            "ul > li:nth-of-type(1)")  # Select the first <li> element within <ul> for area info
        address_element = row.select_one("h3 a")  # Select the address element
        date_element = row.select_one('span.flex-none.text-sm.text-bui-color-middle-dark.ml-3')
        region_element = row.select_one('.object-card__preamble')  # Select the region element

        price = None  # Set default value for price
        area = None  # Set default value for area
        address = None  # Set default value for address
        date = None  # Set default value for date
        link = None  # Set default value for link
        region1 = None  # Set default value for region1
        region2 = None  # Set default value for region2

        if price_element:
            price_text = price_element.get_text().strip()  # Get the text content of the price element
            price_text = price_text.replace("\xa0", "")  # Remove non-breaking space characters
            price_match = re.search(r'(\d+)', price_text)  # Adjusted regex pattern to extract digits from the price text
            if price_match:
                price = int(price_match.group(1))  # Convert the price to an integer

        if area_element:
            area_text = area_element.get_text().strip()  # Get the text content of the area element
            area_text = area_text.replace("&nbsp", "")
            area_text = area_text.replace("½", ".5")
            area_match = re.search(r'([0-9.]+)', area_text)  # Extract the square meter using regex
            if area_match:
                area = round(float(area_match.group(1)), 1)  # Extract the square meter value and round to one decimal place

        if address_element:
            address = address_element.get_text().strip()  # Get the text content of the address element

        if date_element:
            date_text = date_element.get_text().strip()  # Get the text content of the date element
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_text)  # Extract the date using regex
            if date_match:
                date = date_match.group(1)  # Extract the date value

        if address_element:
            link = address_element['href']  # Extract the link value
            link = "https://www.booli.se" + link

        if region_element:
            region_text = region_element.get_text().strip()  # Get the text content of the region element
            region_parts = region_text.split(" · ")
            if len(region_parts) >= 2:
                region1 = region_parts[-2]
                region2 = region_parts[-1]

        data.append({"Price": price, "Area": area, "Address": address, "Date": date, "Link": link, "Region1": region1, "Region2": region2})

    df = pd.DataFrame(data)
    df = df.dropna()

    # Calculate price per area and round to the nearest integer
    df['Price per sqm'] = (df['Price'] / df['Area']).round().astype(int)
    df['Price'] = df['Price'].astype(int)

    return df





import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


def sub_page(link):
    try:
        request_apartment = requests.get(link)
        soup_apartment = BeautifulSoup(request_apartment.text, 'html.parser')

        # Find all <li> elements containing the header and value pairs
        pairs = soup_apartment.find_all('li')

        data = {}

        # Iterate through each pair
        for pair in pairs:
            # Find the <span> element for the header
            header_element = pair.find('span', class_='text-sm')
            if header_element:
                header = header_element.get_text(strip=True)

                # Find the <p> element for the value
                value_element = pair.find('p', class_='heading-5')
                if value_element:
                    value = value_element.get_text(strip=True)

                    # Special handling for each header to extract numerical values and convert them to the desired format
                    if "Rum" in header:
                        # Extract numerical value and handle special case for '½'
                        rum_value = re.search(r'(\d+|½)', value)
                        if rum_value:
                            rum_value = 0.5 if rum_value.group(1) == '½' else int(rum_value.group(1))
                            data["Rum"] = rum_value

                    elif "Avgift" in header:
                        # Extracting the numerical value from the string
                        value = re.search(r'(\d[\d\s]*)', value)
                        if value:
                            # Remove whitespace and convert to int
                            data["Avgift"] = int(value.group(1).replace(' ', '').replace('\xa0', ''))
                    
                    elif "Byggår" in header:
                        data["Built year"] = value

        # Create a DataFrame from the data dictionary
        df = pd.DataFrame([data])

        return df

    except Exception as e:
        print("Error:", e)
        return pd.DataFrame()




def overview(page):
    df = main_page(page)
    # Create an empty list to store additional details
    additional_details = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Check if the "Link" column is not None
        if row['Link'] is not None:
            # Construct the complete URL for each property
            property_url = row['Link']
            # Fetch additional details for the property
            details = sub_page(property_url)

            # Append the additional details to the list
            additional_details.append(details)

    # Concatenate the list of DataFrames into one DataFrame
    if additional_details:
        additional_details_df = pd.concat(additional_details, ignore_index=True, sort=False)
        
        # Replace null values in "Avgift" column with 0
        additional_details_df['Avgift'].fillna(0, inplace=True)
        additional_details_df['Avgift'] = additional_details_df['Avgift'].astype(int)


        # Concatenate the additional details DataFrame with the original DataFrame
        result_df = pd.concat([df, additional_details_df], axis=1)

        return result_df
    else:
        print("No additional details found.")
        return df



def extract_page(page):
    request = requests.get(page)
    soup = BeautifulSoup(request.text, 'lxml')
    html = str(soup)  # Convert BeautifulSoup object to string
    pattern = r'Visar sida <!-- -->\d+<!-- --> av <!-- -->(\d+)'

    # Search for the pattern in the HTML
    match = re.search(pattern, html)

    if match:
        # Extract the number of pages from the matched group
        page_count = int(match.group(1))
        return page_count
    else:
        return None



def multi_page_overview(page):
    start_time = time.time()  # Record the start time
    maximum_page = extract_page(page)
    df = overview(page)
    # Create an empty list to store additional details
    additional_details = []

    # Iterate over each page
    for page_number in range(2, maximum_page + 1):
    # for page_number in range(2, 3):
        # Record the start time for each page
        page_start_time = time.time()
        
        # Construct the complete URL for the page
        page_url = page + '&page=' + str(page_number)
        # Fetch additional details for the page
        page_details = overview(page_url)
        
        # Append the additional details to the list
        additional_details.append(page_details)
        
        # Calculate the percentage completion
        percentage = (page_number / maximum_page) * 100
        
        # Calculate the time taken for the current page
        page_elapsed_time = time.time() - page_start_time
        
        # Print progress
        print("Processed page", page_number, "out of", maximum_page, "(", round(percentage, 2), "%)", "- Time:", round(page_elapsed_time, 2), "seconds per page")
    
    # Concatenate the list of DataFrames into one DataFrame
    if additional_details:
        additional_details_df = pd.concat(additional_details, ignore_index=True, sort=False)
        # Concatenate the original DataFrame with the additional details DataFrame
        result_df = pd.concat([df, additional_details_df], ignore_index=True, sort=False)
        
        # Calculate the overall time taken
        overall_elapsed_time = time.time() - start_time
        print("Overall time:", round(overall_elapsed_time, 2), "seconds")
        
        return result_df
    else:
        return df


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def predict_house_price(train_data, new_data, model_setting):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(train_data)

    # Preprocess the data (similar to what you did before)
    df.drop('Price per sqm', axis=1, inplace=True)
    df[['Year', 'Month', 'Day']] = df['Date'].str.split('-', expand=True).astype(int)
    df_address = df[['Address']].copy()
    df_address[['Street Name', 'Street Number']] = df_address['Address'].str.extract(r'(.+)\s(\d+)\w*$')
    df_address = df_address.drop_duplicates(subset='Street Name')
    df_address = df_address.pivot(index='Address', columns='Street Name', values='Street Number')

    
    df = df.merge(df_address, how='left', on='Address')
    df = df.fillna(0)
    df.drop(['Date', 'Address', 'Link', 'Valid from','Valid to'], axis=1, inplace=True)
    df.columns = df.columns.astype(str)

    # Split data into features and target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    # model = RandomForestRegressor(n_estimators=20, random_state=42, max_depth=8, min_samples_split=10, min_samples_leaf=7)
    model = model_setting
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate RMSE (Root Mean Squared Error)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Process new data
    new_df = pd.DataFrame([new_data])
    new_df[['Year', 'Month', 'Day']] = new_df['Date'].str.split('-', expand=True).astype(int)
    new_df_address = new_df[['Address']].copy()
    new_df_address[['Street Name', 'Street Number']] = new_df_address['Address'].str.extract(r'(.+)\s(\d+)\w*$')
    new_df_address = new_df_address.pivot(index='Address', columns='Street Name', values='Street Number')
    new_df = new_df.merge(new_df_address, how='left', on='Address')

    column_names_training = X_train.columns

    for column in column_names_training:
        if column not in new_df.columns:
            new_df[column] = 0
    new_df = new_df[X_train.columns]

    # Predict the price for the new data point
    predicted_price = model.predict(new_df)[0]
    upper_bound = predicted_price + rmse_test
    lower_bound = predicted_price - rmse_test

    # Return train RMSE, test RMSE, predicted price, and its range
    return rmse_train, rmse_test, predicted_price, lower_bound, upper_bound, model, X_train, X_test, y_train, y_test



from sklearn.ensemble import RandomForestRegressor

def print_feature_importance(model, feature_names):
    """
    Print feature importance based on a trained Random Forest model.

    Parameters:
    - model: Trained Random Forest model
    - feature_names: List of feature names used in the model
    """
    if not isinstance(model, RandomForestRegressor):
        print("Error: The model provided is not a RandomForestRegressor.")
        return

    # Get feature importances from the model
    importances = model.feature_importances_

    # Create a DataFrame to display feature importances
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

    # Sort features by importance
    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Print the feature importance
    print(importance_df)

import pandas as pd

def merge_and_filter(df):
    # Provided data
    interest_data = {
        'Valid to': ['3/4/2024', '7/2/2024', '29/11/2023', '27/9/2023', '5/7/2023', '3/5/2023', '15/02/2023', '30/11/2022', '21/9/2022', '6/7/2022', '8/6/2022', '4/5/2022', '16/2/2022', '1/12/2021', '22/9/2021', '7/7/2021', '28/4/2021'],
        'Policy rate': [4.00, 4.00, 4.00, 4.00, 3.75, 3.50, 3.00, 2.50, 1.75, 0.75, 0.25, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00],
        'Deposit rate': [3.90, 3.90, 3.90, 3.90, 3.65, 3.40, 2.90, 2.40, 1.65, 0.65, 0.15, 0.15, -0.10, -0.10, -0.10, -0.10, -0.10],
        'Lending rate': [4.10, 4.10, 4.10, 4.10, 3.85, 3.60, 3.10, 2.60, 1.85, 0.85, 0.35, 0.35, 0.10, 0.10, 0.10, 0.10, 0.10]
    }

    # Create DataFrame
    interest_data = pd.DataFrame(interest_data)
    interest_data['Valid to'] = pd.to_datetime(interest_data['Valid to'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    interest_data['Valid to'] = pd.to_datetime(interest_data['Valid to'], errors='coerce')
    interest_data = interest_data.sort_values(by='Valid to', ascending=False)
    interest_data['Valid from'] = pd.to_datetime(interest_data['Valid to']).shift(-1) - pd.Timedelta(days=-1)
    interest_data['Valid from'] = interest_data['Valid from'].fillna('2000-01-01')
    interest_data.loc[interest_data['Valid to'] == '2024-04-03', 'Valid to'] = '2024-12-31'

    # Assuming df2 is your DataFrame
    df2 = interest_data

    # Perform left join
    merged_df = df.merge(df2, how='cross')

    # Filter rows where df.date is between df2.valid_from and df2.valid_to
    result = merged_df[(merged_df['Date'] >= merged_df['Valid from']) & (merged_df['Date'] <= merged_df['Valid to'])]
    return result.sort_values(by='Date', ascending=True)
