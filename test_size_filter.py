#!/usr/bin/env python3

# Test script to verify size filtering behavior
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from depop_api import fetch_depop_items

def test_size_filtering():
    print("Testing Depop API size filtering...")
    
    # Test with size filter
    print("\n1. Testing with size='M' filter:")
    items, cursor = fetch_depop_items("t shirt", size="M", max_items=5)
    
    print(f"Returned {len(items)} items")
    for i, item in enumerate(items[:3]):
        print(f"Item {i+1}: '{item['title'][:30]}...', Size: {item['size']}, All sizes: {item.get('all_sizes', 'N/A')}")
    
    # Test without size filter for comparison
    print("\n2. Testing without size filter:")
    items_no_filter, cursor_no_filter = fetch_depop_items("t shirt", max_items=5)
    
    print(f"Returned {len(items_no_filter)} items")
    for i, item in enumerate(items_no_filter[:3]):
        print(f"Item {i+1}: '{item['title'][:30]}...', Size: {item['size']}, All sizes: {item.get('all_sizes', 'N/A')}")
    
    # Test client-side filtering logic
    print("\n3. Testing client-side filtering logic:")
    selected_sizes = ["M"]
    
    def client_side_filter(items, selected_sizes):
        filtered = []
        for item in items:
            item_sizes = item.get('all_sizes', [item.get('size')])
            if any(selected_size in item_sizes for selected_size in selected_sizes):
                filtered.append(item)
        return filtered
    
    filtered_items = client_side_filter(items_no_filter, selected_sizes)
    print(f"Client-side filtering: {len(items_no_filter)} -> {len(filtered_items)} items for size M")
    
    for i, item in enumerate(filtered_items[:3]):
        print(f"Filtered Item {i+1}: '{item['title'][:30]}...', Size: {item['size']}, All sizes: {item.get('all_sizes', 'N/A')}")

if __name__ == "__main__":
    test_size_filtering()
