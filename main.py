
from depop_scraper import scrape_depop_items

if __name__ == "__main__":
    # Example usage
    results = scrape_depop_items("t shirt", max_items=10)
    for item in results:
        print(item)
