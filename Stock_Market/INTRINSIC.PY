import requests
from bs4 import BeautifulSoup

def fetch_eps_pe_ratio(stock_symbol):
    try:
        # Construct the URL for the Yahoo Finance summary page
        url = f'https://finance.yahoo.com/quote/{stock_symbol}?p={stock_symbol}'

        # Send an HTTP request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract EPS and P/E ratio using appropriate HTML tags and classes
            eps = None

            # Try to find EPS using different attributes
            possible_eps_attributes = ['data-reactid', 'data-test']
            for attribute in possible_eps_attributes:
                eps_element = soup.find('td', {attribute: 'OPEN-value'})
                if eps_element:
                    eps = eps_element.text.strip()
                    break

            pe_ratio = soup.find('td', {'data-test': 'PE_RATIO-value'}).text.strip()

            return eps, pe_ratio
        else:
            print(f"Error: Failed to fetch data. Status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def calculate_intrinsic_value(eps, pe_ratio, required_rate_of_return):
    try:
        eps = float(eps.replace(',', ''))  # Convert EPS to float (handle commas in large numbers)
        pe_ratio = float(pe_ratio.replace(',', ''))  # Convert P/E ratio to float

        # Calculate intrinsic value using the formula: Intrinsic value = EPS * (1 + r) * P/E ratio
        intrinsic_value = eps * (1 + required_rate_of_return / 100) * pe_ratio
        return intrinsic_value
    except ValueError:
        print("Error: Invalid EPS or P/E ratio format.")
        return None

try:
    stock_symbol = input("Enter the stock symbol (e.g., AAPL): ")
    required_rate_of_return = float(input("Enter the required rate of return (%): "))

    eps, pe_ratio = fetch_eps_pe_ratio(stock_symbol)

    if eps is not None and pe_ratio is not None:
        print(f"Earnings per share (EPS): {eps}")
        print(f"Price-to-earnings (P/E) ratio: {pe_ratio}")

        intrinsic_value = calculate_intrinsic_value(eps, pe_ratio, required_rate_of_return)

        if intrinsic_value is not None:
            print(f"Intrinsic value: {intrinsic_value}")
        else:
            print("Error calculating intrinsic value.")
    else:
        print(f"Error: EPS or P/E ratio information not available for {stock_symbol} on Yahoo Finance.")
except ValueError:
    print("Error: Invalid input. Please enter valid numeric values.")
except KeyboardInterrupt:
    print("\nOperation aborted.")
except Exception as e:
    print(f"Error: {e}")
