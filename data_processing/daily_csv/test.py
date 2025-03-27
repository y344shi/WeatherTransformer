import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import time

# ------------------------------
# CONFIG
# ------------------------------
BASE_SEARCH_URL = "https://climat.meteo.gc.ca/historical_data/search_historic_data_stations_e.html"
PROVINCE = "AB"            # Province code
START_YEAR = 2007          # For CSV downloads (daily data)
END_YEAR = 2024
ROWS_PER_PAGE = 25         # 25 stations per page
SLEEP_SECONDS = 1          # Polite pause between page fetches
DOWNLOAD = True            # Set to False to skip downloading CSV
SAVE_DIR = "daily_csv"     # Folder where CSV will be saved

# ------------------------------
# FUNCTIONS
# ------------------------------

def get_station_ids_for_page(start_row=0, row_per_page=25):
    """
    Fetch a single "page" of stations for the province, starting at `start_row`.
    Returns (station_ids, station_count):
      - station_ids is a list of IDs found on this page
      - station_count is how many <form> items (stations) were found
    """
    params = {
        "searchType": "stnProv",
        "timeframe": "1",
        "lstProvince": PROVINCE,
        "optLimit": "yearRange",
        "StartYear": "2006",
        "EndYear": "2024",
        "Year": "2025",
        "Month": "3",
        "Day": "19",
        "selRowPerPage": str(row_per_page),
        # The key param for pagination in this version:
        "startRow": str(start_row),
    }
    
    # Build the URL for this page
    url = f"{BASE_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    resp = requests.get(url)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    station_forms = soup.select('form[id^="stnRequest"]')
    
    station_ids = []
    for form in station_forms:
        stn_id_input = form.find("input", {"name": "StationID"})
        if stn_id_input and stn_id_input.has_attr("value"):
            station_ids.append(stn_id_input["value"])
    
    return station_ids, len(station_forms)


def fetch_all_station_ids():
    """
    Paginate through ALL pages for the given province by incrementing startRow in steps of 25,
    collecting station IDs until no more are found.
    """
    all_ids = []
    start_row = 0
    
    while True:
        page_ids, form_count = get_station_ids_for_page(start_row, ROWS_PER_PAGE)
        if form_count == 0:
            # No more stations on this page => done
            break
        all_ids.extend(page_ids)
        start_row += ROWS_PER_PAGE
        print(f"Fetching stations at startRow={start_row}...[page_ids]")
        time.sleep(SLEEP_SECONDS)  # small pause so we don't bombard the server
    
    return all_ids


def build_bulk_data_url(station_id, year, timeframe=2, month=1, day=14, use_utc=False):
    """
    Build the bulk_data_e.html URL for CSV downloads.
    timeframe=2 => daily data by default.
    You can set timeframe=1 for hourly, 3 for monthly, etc.
    Setting use_utc=True will add &time=utc to the query string.
    """
    base = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    query = {
        "format": "csv",
        "stationID": station_id,
        "Year": year,
        "Month": month,
        "Day": day,
        "timeframe": timeframe,        # 2 => daily
        "submit": " Download+Data"
    }
    if use_utc:
        query["time"] = "utc"         # Request UTC data instead of local time
    
    return f"{base}?{urllib.parse.urlencode(query)}"


def download_csv(url, out_path):
    """
    Download CSV from `url`, saving to `out_path`.
    """
    resp = requests.get(url, timeout=30)
    # If station/year not found, sometimes the site returns a 404 or custom page
    resp.raise_for_status()
    
    with open(out_path, "wb") as f:
        f.write(resp.content)


def main():
    # 1) Get all station IDs across all pages
    station_ids = fetch_all_station_ids()
    
    # Remove duplicates if any
    station_ids = list(set(station_ids))
    print(f"\nCollected {len(station_ids)} unique station IDs for {PROVINCE}.\n")
    
    if DOWNLOAD:
        os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2) For each station ID, build daily CSV URLs for 2007–2024, check them, optionally download
    for stn_id in station_ids:
        print(f"== Station ID {stn_id} ==")
        
        for year in range(START_YEAR, END_YEAR + 1):
            # By default timeframe=2 => daily
            url = build_bulk_data_url(stn_id, year, timeframe=2)
            
            # Quick HEAD check to see if the resource is valid
            try:
                head_resp = requests.head(url, timeout=10)
                status = head_resp.status_code
            except Exception as e:
                print(f"  Year {year}: HEAD Error: {e}")
                continue
            
            if status != 200:
                print(f"  Year {year}: Not found (HTTP {status}) => skip.")
                continue
            
            print(f"  Year {year}: OK => {url}")
            
            # Optional: Actually download CSV
            if DOWNLOAD:
                filename = f"station_{stn_id}_{year}.csv"
                out_path = os.path.join(SAVE_DIR, filename)
                
                try:
                    download_csv(url, out_path)
                    print(f"    => Downloaded to {out_path}")
                except Exception as e:
                    print(f"    !! Download error => {e}")
            
            # Polite sleep to avoid server overload
            time.sleep(1)

if __name__ == "__main__":
    main()
