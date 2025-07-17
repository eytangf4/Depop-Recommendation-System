import requests
import json

def fetch_depop_items(query, gender=None, category=None, subcategory=None, brand=None, size=None, color=None, condition=None, price_min=None, price_max=None, on_sale=None, sort=None, max_items=24, offset=None, **kwargs):
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
    
    # Map new filter parameters
    if gender:
        params["gender"] = gender
    
    # Handle category/group mapping
    if category:
        # Map category to Depop's groups parameter
        category_mapping = {
            'tops': 'tops',
            'bottoms': 'bottoms', 
            'footwear': 'footwear',
            'shoes': 'footwear',  # Add shoes alias
            'accessories': 'accessories',
            'coats_and_jackets': 'outerwear',
            'outerwear': 'outerwear',
            'dresses': 'dresses',
            'jumpsuits_and_playsuits': 'jumpsuit-and-playsuit',
            'suits': 'suits',
            'nightwear': 'nightwear',
            'underwear': 'underwear',
            'swimwear': 'swimwear',
            'bags_and_luggage': 'bags',
            'jewellery': 'jewellery',
            'beauty': 'beauty',
            'electronics': 'electronics',
            'home': 'home',
            'pets': 'pets',
            'books_and_magazines': 'books',
            'music': 'music',
            'sports': 'sports'
        }
        mapped_category = category_mapping.get(category.lower(), category)
        params["groups"] = mapped_category
    
    # Use subcategory if available and more specific
    if subcategory:
        # Remove 'category-' prefix if it exists and map to productTypes
        clean_subcategory = subcategory.replace('category-', '')
        params["productTypes"] = clean_subcategory
        
    if price_min:
        params["price_min"] = price_min
    if price_max:
        params["price_max"] = price_max
    if brand:
        # Handle array of brands - join with comma for multiple brands
        # Note: We should send brand IDs, not brand names to the API
        if isinstance(brand, list) and len(brand) > 0:
            # If we receive brand names, we need to convert them to IDs
            # For now, assume we're getting brand IDs already
            params["brands"] = ",".join(str(b) for b in brand)
        elif not isinstance(brand, list):
            params["brands"] = str(brand)
    if color:
        # Handle array of colors - join with comma for multiple colors
        if isinstance(color, list) and len(color) > 0:
            # Convert to lowercase and join with comma
            color_values = [c.lower() for c in color]
            params["colours"] = ",".join(color_values)
        elif not isinstance(color, list):
            params["colours"] = color.lower()
    if condition:
        # Handle array of conditions - join with comma, map to Depop's condition values
        condition_mapping = {
            'new_with_tags': 'brand_new',
            'new_without_tags': 'brand_new', 
            'very_good': 'used_like_new',
            'good': 'used_good',
            'satisfactory': 'used_fair',
            'poor': 'used_fair'
        }
        if isinstance(condition, list) and len(condition) > 0:
            mapped_conditions = [condition_mapping.get(c.lower(), c.lower()) for c in condition]
            params["conditions"] = ",".join(mapped_conditions)
        elif not isinstance(condition, list):
            mapped_condition = condition_mapping.get(condition.lower(), condition.lower())
            params["conditions"] = mapped_condition
    if size:
        # Handle array of sizes - join with comma for multiple sizes
        if isinstance(size, list) and len(size) > 0:
            # For sizes, we can directly use the display values (S, M, L, etc.)
            params["size"] = ",".join(size)
        elif not isinstance(size, list):
            params["size"] = size
    if price_min:
        params["price_min"] = price_min
    if price_max:
        params["price_max"] = price_max
    if on_sale:
        params["isDiscounted"] = "true"
    if sort:
        # Map sort values to Depop's actual API format
        sort_mapping = {
            'relevance': 'relevance',
            'price_low': 'priceAscending', 
            'price_high': 'priceDescending',
            'newest': 'newlyListed',
            'oldest': 'oldest',
            'most_popular': 'mostPopular'
        }
        mapped_sort = sort_mapping.get(sort, 'relevance')
        params["sort"] = mapped_sort
    if offset:
        params["cursor"] = offset

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Referer": "https://www.depop.com/"
    }

    # Debug logging
    print(f"DEBUG - Depop API URL: {base_url}")
    print(f"DEBUG - Depop API params: {params}")

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
