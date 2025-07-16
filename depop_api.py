import requests
import json

def fetch_depop_items(query, group=None, gender=None, brand=None, size=None, color=None, price_min=None, price_max=None, on_sale=None, max_items=24, offset=None):
    """
    Fetch Depop items using the Depop API.
    Returns a tuple: (items, next_cursor)
    """
    base_url = "https://webapi.depop.com/api/v3/search/products/"
    params = {
        "what": query,
        "country": "us",
        "currency": "USD",
        "items_per_page": str(max_items),
        "force_fee_calculation": "false",
        "from": "in_country_search"
    }
    if group:
        params["groups"] = group
    if gender:
        params["gender"] = gender
    if size:
        params["size"] = size
    if price_min:
        params["price_min"] = price_min
    if price_max:
        params["price_max"] = price_max
    if brand:
        params["brand"] = brand
    if color:
        params["colour"] = color  # UK spelling
    if on_sale:
        params["on_sale"] = "true"
    if offset:
        params["cursor"] = offset

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Referer": "https://www.depop.com/"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        products = data.get("products", [])
        # Get next_cursor from meta
        next_cursor = data.get("meta", {}).get("cursor")
        items = []
        for product in products:
            slug = product.get("slug", "")
            brand = product.get("brand_name", "")
            title = slug.replace('-', ' ').title()
            if brand and brand.lower() not in title.lower():
                title = f"{brand} {title}".strip()
            pricing = product.get("pricing", {})
            price_original = pricing.get("original_price", {}).get("total_price", None)
            price_sale = pricing.get("discounted_price", {}).get("total_price") if pricing.get("is_reduced") else None
            price = price_sale if price_sale else price_original
            preview = product.get("preview", {})
            image_url = preview.get("640") or preview.get("480") or preview.get("320") or ""
            pictures = product.get("pictures", [])
            image_url2 = ""
            if len(pictures) > 1:
                image_url2 = pictures[1].get("640") or pictures[1].get("480") or pictures[1].get("320") or ""
            sizes = product.get("sizes", [])
            size = sizes[0] if sizes else None
            item_url = f"https://www.depop.com/products/{slug}/"
            item = {
                "title": title,
                "price": price,
                "price_original": price_original,
                "price_sale": price_sale if price_sale else price_original,
                "size": size,
                "brand": brand,
                "image_url": image_url,
                "image_url2": image_url2,
                "item_url": item_url
            }
            items.append(item)
        return items, next_cursor
    except Exception as e:
        print(f"Error fetching Depop API: {e}")
        return [], None
