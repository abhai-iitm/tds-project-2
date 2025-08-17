import requests
from bs4 import BeautifulSoup
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
from sklearn.linear_model import LinearRegression

def scrape_data(url):
    """Scrapes the data from the Wikipedia page."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        if not table:
            return None
        
        headers = [th.text.strip() for th in table.find_all('th')]
        
        # Handle variations in table structure
        if "Worldwide gross" not in headers:
            table = soup.find('table', {'class': 'wikitable sortable'})
            if not table:
                return None
            headers = [th.text.strip() for th in table.find_all('th')]

        data = []
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = [td.text.strip() for td in row.find_all('td')]
            if len(cells) >= len(headers) and headers:
                row_data = {}
                for i, header in enumerate(headers):
                    if i < len(cells):
                        row_data[header] = cells[i]
                data.append(row_data)

        return data
    except requests.exceptions.RequestException as e:
        print(f"Error during data retrieval: {e}")
        return None
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None

def clean_gross(gross_str):
    """Cleans and converts gross revenue to a float."""
    if not isinstance(gross_str, str):
        return None
    
    gross_str = gross_str.replace('$', '').replace(',', '').replace('M', '').replace('B', '')
    
    try:
        return float(gross_str) * (1000000000 if 'B' in gross_str.upper() else 1000000)
    except ValueError:
        return None

def question_1(data, target_year, threshold_2bn):
    """Answers question 1: How many $2 bn movies were released before 2020?"""
    count = 0
    if not data:
        return json.dumps(["0"])
    for row in data:
        try:
            year = int(row.get('Year', '').split('[')[0])  # Handle potential footnotes
            gross_str = row.get('Worldwide gross', '')
            gross = clean_gross(gross_str)
            if year < target_year and gross is not None and gross >= threshold_2bn * 1000000000:
                count += 1
        except (ValueError, TypeError):
            continue
    return json.dumps([str(count)])


def question_2(data, threshold_1_5bn):
    """Answers question 2: Which is the earliest film that grossed over $1.5 bn?"""
    eligible_films = []
    if not data:
        return json.dumps([])

    for row in data:
        try:
            year = int(row.get('Year', '').split('[')[0])  # Handle potential footnotes
            gross_str = row.get('Worldwide gross', '')
            gross = clean_gross(gross_str)
            if gross is not None and gross >= threshold_1_5bn * 1000000000:
                eligible_films.append((year, row.get('Title', '')))
        except (ValueError, TypeError):
            continue
    
    if not eligible_films:
        return json.dumps([])
    
    eligible_films.sort(key=lambda x: x[0])  # Sort by year
    earliest_film = eligible_films[0][1]
    return json.dumps([earliest_film])


def question_3(data):
    """Answers question 3: What's the correlation between the Rank and Peak?"""
    ranks = []
    peaks = []

    if not data:
        return json.dumps(["0.00"])
    
    for row in data:
        try:
            rank_str = row.get('Rank', '')
            rank = int(re.sub(r"[^0-9]", "", rank_str))  # Extract rank as integer
            peak_str = row.get('Peak', '')
            peak = int(re.sub(r"[^0-9]", "", peak_str)) if peak_str else None # Extract peak if present

            if peak is None:
                gross_str = row.get('Worldwide gross', '')
                gross = clean_gross(gross_str)
                if gross is not None:
                    peak = int(gross / 1000000000)
                else:
                    continue
            
            ranks.append(rank)
            peaks.append(peak)

        except (ValueError, TypeError):
            continue

    if not ranks or not peaks:
         return json.dumps(["0.00"])

    try:
        correlation = np.corrcoef(ranks, peaks)[0, 1]
        return json.dumps([f"{correlation:.2f}"])
    except:
        return json.dumps(["0.00"])
        


def question_4(data):
    """Answers question 4: Draw a scatterplot of Rank and Peak with regression line."""
    ranks = []
    peaks = []

    if not data:
        return "data:image/png;base64,"
    
    for row in data:
        try:
            rank_str = row.get('Rank', '')
            rank = int(re.sub(r"[^0-9]", "", rank_str))  # Extract rank as integer
            peak_str = row.get('Peak', '')
            peak = int(re.sub(r"[^0-9]", "", peak_str)) if peak_str else None
            
            if peak is None:
                gross_str = row.get('Worldwide gross', '')
                gross = clean_gross(gross_str)
                if gross is not None:
                    peak = int(gross / 1000000000)
                else:
                    continue

            ranks.append(rank)
            peaks.append(peak)

        except (ValueError, TypeError):
            continue

    if not ranks or not peaks:
        return "data:image/png;base64,"

    # Perform linear regression
    try:
        ranks_np = np.array(ranks).reshape(-1, 1)
        peaks_np = np.array(peaks)
        model = LinearRegression()
        model.fit(ranks_np, peaks_np)
        y_pred = model.predict(ranks_np)

        # Create scatterplot with regression line
        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, peaks)
        plt.plot(ranks, y_pred, color='red', linestyle=':')
        plt.xlabel("Rank")
        plt.ylabel("Peak")
        plt.title("Rank vs. Peak with Regression Line")
        plt.gca().invert_xaxis()  # Invert x-axis for rank (lower rank = higher)
        plt.tight_layout()

        # Save plot to in-memory PNG
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode('utf-8')
        plt.close()
        return f"data:image/png;base64,{plot_data}"
    except Exception as e:
        print(f"Error generating plot: {e}")
        return "data:image/png;base64,"

# Main execution
url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
target_year = 2020
threshold_2bn = 2
threshold_1_5bn = 1.5

data = scrape_data(url)

if data is not None:
    answer_1 = question_1(data, target_year, threshold_2bn)
    answer_2 = question_2(data, threshold_1_5bn)
    answer_3 = question_3(data)
    answer_4 = question_4(data)

    final_json = {
        "answer_1": json.loads(answer_1),
        "answer_2": json.loads(answer_2),
        "answer_3": json.loads(answer_3),
        "answer_4": answer_4
    }
    print(json.dumps(final_json))
else:
    print(json.dumps({
        "answer_1": ["0"],
        "answer_2": [],
        "answer_3": ["0.00"],
        "answer_4": "data:image/png;base64,"
    }))
