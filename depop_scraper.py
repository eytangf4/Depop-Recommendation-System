
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_depop_items(keyword, gender=None, size=None, max_items=20):
    """
    Scrape Depop for items matching the keyword, gender, and size.
    Returns a list of item dicts: {title, price, image_url, item_url}
    """
    base_url = "https://www.depop.com/search/"
    search_url = f"{base_url}?q={keyword}"

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--lang=en-US')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36')


    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(search_url)

    # Explicitly wait for item cards to load
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-testid="listing-card-link"]'))
        )
    except Exception as e:
        print(f"[DEBUG] Timeout waiting for item cards: {e}")

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, 'html.parser')
    items = []
    # New selector based on provided HTML
    listing_items = soup.select('li.styles_listItem__Uv9lb a.styles_unstyledLink__DsttP')
    print(f"[DEBUG] Found {len(listing_items)} listing items on the page.")
    if len(listing_items) == 0:
        print('[DEBUG] No listing items found. Printing HTML snippet:')
        print(html[:1000])

    for link in listing_items:
        item_url = 'https://www.depop.com' + link.get('href', '')
        # Find image
        image_tag = link.find('img', class_='_mainImage_e5j9l_11')
        image_url = image_tag['src'] if image_tag and 'src' in image_tag.attrs else None
        # Find price and size
        parent_li = link.find_parent('li', class_='styles_listItem__Uv9lb')
        price = None
        size = None
        brand = None
        title = None
        if parent_li:
            price_tag = parent_li.find('p', class_='styles_price__H8qdh')
            price = price_tag.text.strip() if price_tag else 'N/A'
            size_tag = parent_li.find('p', class_='styles_sizeAttributeText__r9QJj')
            size = size_tag.text.strip() if size_tag else None
            brand_tag = parent_li.find_all('p')
            if len(brand_tag) > 2:
                brand = brand_tag[2].text.strip()
            title = brand if brand else 'No title'
        # No filtering by gender or size
        items.append({
            'title': title,
            'price': price,
            'size': size,
            'image_url': image_url,
            'item_url': item_url
        })
        if len(items) >= max_items:
            break
    print(f"[DEBUG] Returning {len(items)} filtered items.")
    return items
