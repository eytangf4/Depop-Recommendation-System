
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_depop_items(keyword, gender=None, size=None, max_items=20, offset=0):
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

    count = 0
    for link in listing_items:
        if count < offset:
            count += 1
            continue
        item_url = 'https://www.depop.com' + link.get('href', '')
        # Get main image from main container
        main_img = link.select_one('._container_e5j9l_4 > img._mainImage_e5j9l_11')
        image_url = main_img['src'] if main_img and 'src' in main_img.attrs else None
        # Get hover image from hover overlay
        hover_img = link.select_one('.styles_hoverOverlay__6Zs_i img._mainImage_e5j9l_11')
        image_url2 = hover_img['src'] if hover_img and 'src' in hover_img.attrs else None
        # Find price and size
        parent_li = link.find_parent('li', class_='styles_listItem__Uv9lb')
        price = None
        price_original = None
        price_sale = None
        size = None
        brand = None
        title = None
        if parent_li:
            # Sale price (if any) using aria-labels
            price_full = parent_li.find('p', attrs={'aria-label': 'Full price'})
            price_discount = parent_li.find('p', attrs={'aria-label': 'Discounted price'})
            price_regular = parent_li.find('p', attrs={'aria-label': 'Price'})

            if price_full and price_discount:
                price_original = price_full.text.strip()
                price_sale = price_discount.text.strip()
                price = price_sale
            elif price_regular:
                price = price_regular.text.strip()
                price_original = None
                price_sale = None
            else:
                price = 'N/A'
                price_original = None
                price_sale = None
            size_tag = parent_li.find('p', class_='styles_sizeAttributeText__r9QJj')
            size = size_tag.text.strip() if size_tag else None
            brand_tag = parent_li.find_all('p')
            if len(brand_tag) > 2:
                brand = brand_tag[2].text.strip()
            title = brand if brand else 'No title'
        items.append({
            'title': title,
            'price': price,
            'price_original': price_original,
            'price_sale': price_sale,
            'size': size,
            'image_url': image_url,
            'image_url2': image_url2,
            'item_url': item_url
        })
        count += 1
        if len(items) >= max_items:
            break
    print(f"[DEBUG] Returning {len(items)} filtered items.")
    return items
