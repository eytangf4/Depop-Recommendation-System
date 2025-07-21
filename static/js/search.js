class SearchManager {
    constructor() {
        this.currentFilters = {};
        this.nextCursor = null;
        this.isLoading = false;
        this.hasMoreResults = true;
        this.loadedItemUrls = new Set();
        // Create a data store for item data to avoid JSON in HTML attributes
        this.itemDataStore = new Map();
        this.setupSearchHandlers();
        this.setupInfiniteScroll();
        this.setupSpinner();
        this.setupMobileFilters();
    }

    // Simple hash function to create unique IDs from URLs
    simpleHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash).toString(36);
    }

    setupSearchHandlers() {
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.performSearch();
            });
        }
    }

    setupMobileFilters() {
        // Mobile filter button
        const mobileFilterBtn = document.getElementById('mobileFilterBtn');
        const filtersContainer = document.getElementById('filtersContainer');
        const closeFiltersBtn = document.getElementById('closeFiltersBtn');
        const viewItemsBtn = document.getElementById('viewItemsBtn');

        if (mobileFilterBtn && filtersContainer) {
            mobileFilterBtn.addEventListener('click', () => {
                filtersContainer.classList.remove('d-none');
                filtersContainer.classList.add('show');
                document.body.style.overflow = 'hidden';
            });
        }

        if (closeFiltersBtn && filtersContainer) {
            closeFiltersBtn.addEventListener('click', () => {
                this.closeMobileFilters();
            });
        }

        if (viewItemsBtn && filtersContainer) {
            viewItemsBtn.addEventListener('click', () => {
                this.closeMobileFilters();
                this.performSearch();
            });
        }

        // Close on backdrop click
        if (filtersContainer) {
            filtersContainer.addEventListener('click', (e) => {
                if (e.target === filtersContainer) {
                    this.closeMobileFilters();
                }
            });
        }

        // Apply price filter button
        const applyPriceBtn = document.getElementById('applyPriceBtn');
        if (applyPriceBtn) {
            applyPriceBtn.addEventListener('click', () => {
                this.updatePriceLabel();
                // Close the price dropdown
                const priceDropdown = bootstrap.Dropdown.getInstance(document.getElementById('priceDropdown'));
                if (priceDropdown) {
                    priceDropdown.hide();
                }
            });
        }

        // On sale toggle
        const onSaleToggle = document.getElementById('onSaleToggle');
        if (onSaleToggle) {
            onSaleToggle.addEventListener('change', () => {
                this.performSearch();
            });
        }
    }

    closeMobileFilters() {
        const filtersContainer = document.getElementById('filtersContainer');
        if (filtersContainer) {
            filtersContainer.classList.remove('show');
            document.body.style.overflow = '';
            setTimeout(() => {
                filtersContainer.classList.add('d-none');
            }, 300);
        }
    }

    updatePriceLabel() {
        const minPrice = document.getElementById('priceMinInput')?.value;
        const maxPrice = document.getElementById('priceMaxInput')?.value;
        const priceLabel = document.getElementById('priceLabel');
        
        if (priceLabel) {
            if (minPrice || maxPrice) {
                let labelText = 'Price: ';
                if (minPrice && maxPrice) {
                    labelText += `$${minPrice} - $${maxPrice}`;
                } else if (minPrice) {
                    labelText += `$${minPrice}+`;
                } else {
                    labelText += `Under $${maxPrice}`;
                }
                priceLabel.textContent = labelText;
                document.getElementById('priceDropdown').classList.add('active');
            } else {
                priceLabel.textContent = 'Price';
                document.getElementById('priceDropdown').classList.remove('active');
            }
        }
    }

    performSearch(loadMore = false) {
        if (this.isLoading) return;
        
        if (!loadMore) {
            // New search - reset everything
            this.nextCursor = null;
            this.hasMoreResults = true;
            this.loadedItemUrls.clear();
            // Clear the item data store to prevent memory leaks
            this.itemDataStore.clear();
            const resultsContainer = document.getElementById('resultsContainer');
            if (resultsContainer) {
                resultsContainer.innerHTML = '';
            }
        }

        this.isLoading = true;
        this.updateLoadingState(true);

        // Get search term
        const searchTerm = document.getElementById('queryInput')?.value?.trim() || '';
        
        // Get filter values from the advanced filter system
        const filters = window.depopFilters ? window.depopFilters.getFilterValues() : this.getBasicFilterValues();
        
        this.currentFilters = { query: searchTerm, ...filters };

        // Prepare the request data
        const requestData = { ...this.currentFilters };
        
        // Handle array values (like subcategories)
        if (Array.isArray(requestData.subcategory) && requestData.subcategory.length === 0) {
            delete requestData.subcategory;
        }

        if (this.nextCursor) {
            requestData.cursor = this.nextCursor;
        }

        const url = loadMore ? '/search_results_page' : '/search_results';

        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.items) {
                // Apply client-side size filtering if size filters are selected
                let filteredItems = data.items;
                if (window.depopFilters && window.depopFilters.selectedSizes.size > 0) {
                    const selectedSizes = Array.from(window.depopFilters.selectedSizes);
                    filteredItems = data.items.filter(item => {
                        // Check if item has any of the selected sizes
                        const itemSizes = item.all_sizes || [item.size];
                        return selectedSizes.some(selectedSize => 
                            itemSizes.includes(selectedSize)
                        );
                    });
                    console.log(`DEBUG - Size filtering: ${data.items.length} items -> ${filteredItems.length} items after filtering for sizes: ${selectedSizes.join(', ')}`);
                }

                if (loadMore) {
                    this.appendResults(filteredItems);
                } else {
                    this.displayResults(filteredItems);
                }
                this.nextCursor = data.next_cursor;
                this.hasMoreResults = !!data.next_cursor && data.items.length > 0;
                
                // Auto-load more if we have cursor and not enough content
                if (this.nextCursor && data.items.length > 0 && !loadMore) {
                    setTimeout(() => {
                        if (this.shouldAutoLoad()) {
                            this.performSearch(true);
                        }
                    }, 100);
                }
            } else {
                this.displayError(data.message || 'No results found');
            }
        })
        .catch(error => {
            console.error('Search error:', error);
            this.displayError('An error occurred while searching.');
        })
        .finally(() => {
            this.isLoading = false;
            this.updateLoadingState(false);
        });
    }

    getBasicFilterValues() {
        // Fallback for basic filter values if depopFilters isn't available
        return {
            gender: document.getElementById('genderInput')?.value || '',
            category: document.getElementById('categoryInput')?.value || '',
            subcategory: [document.getElementById('subcategoryInput')?.value || ''].filter(v => v),
            brand: document.getElementById('brandInput')?.value || '',
            size: document.getElementById('sizeInput')?.value || '',
            color: document.getElementById('colorInput')?.value || '',
            condition: document.getElementById('conditionInput')?.value || '',
            price_min: document.getElementById('priceMinInput')?.value || '',
            price_max: document.getElementById('priceMaxInput')?.value || '',
            on_sale: document.getElementById('onSaleToggle')?.checked || false,
            sort: 'relevance'
        };
    }

    displayResults(items) {
        const resultsContainer = document.getElementById('resultsContainer');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = '';
        // Clear data store when displaying new results
        this.itemDataStore.clear();

        if (!items || items.length === 0) {
            resultsContainer.innerHTML = '<p class="text-center">No items found. Try different search terms or filters.</p>';
            return;
        }

        this.appendResults(items);
    }

    appendResults(items) {
        if (!items || items.length === 0) {
            this.hasMoreResults = false;
            return;
        }

        const resultsContainer = document.getElementById('resultsContainer');
        if (!resultsContainer) return;

        let grid = resultsContainer.querySelector('.results-grid');
        if (!grid) {
            grid = document.createElement('div');
            grid.className = 'results-grid';
            resultsContainer.appendChild(grid);
        }

        items.forEach(item => {
            // Avoid duplicates
            if (this.loadedItemUrls.has(item.item_url)) {
                return;
            }
            this.loadedItemUrls.add(item.item_url);

            const itemElement = this.createItemElement(item);
            grid.appendChild(itemElement);
        });

        this.attachItemHandlers();
    }

    createItemElement(item) {
        const card = document.createElement('div');
        card.className = 'depop-card';
        card.setAttribute('data-url', item.item_url);
        
        // Generate a more robust unique ID to avoid collisions
        // Use a combination of URL hash and timestamp for uniqueness
        const urlHash = this.simpleHash(item.item_url);
        const uniqueId = 'item_' + urlHash + '_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
        console.log(`🔗 Creating item element: "${item.title}" -> ID: ${uniqueId} -> URL: ${item.item_url}`);
        this.itemDataStore.set(uniqueId, item);
        card.setAttribute('data-item-id', uniqueId);

        // Handle pricing display
        let priceHtml = '';
        if (item.price_original && item.price_sale && item.price_original !== item.price_sale) {
            priceHtml = `<span class="price-original" style="text-decoration:line-through;color:#888;margin-right:0.5em;">$${item.price_original}</span> <span class="price-sale" style="color:#222;font-weight:600;">$${item.price_sale}</span>`;
        } else {
            priceHtml = `<span class="price-sale" style="color:#222;font-weight:600;">$${item.price_sale || item.price}</span>`;
        }

        // Handle size display - show all available sizes if multiple, or just the main size
        let sizeHtml = '';
        if (item.all_sizes && item.all_sizes.length > 1) {
            sizeHtml = `Size: ${item.all_sizes.join(', ')}`;
        } else if (item.size) {
            sizeHtml = `Size: ${item.size}`;
        }

        card.innerHTML = `
            <div class="depop-img-wrap">
                <img src="${item.image_url}" alt="Item image" class="item-img" data-img1="${item.image_url}" data-img2="${item.image_url2 || ''}">
            </div>
            <div class="depop-info">
                <div class="depop-price">${priceHtml}</div>
                <div class="depop-size">${sizeHtml}</div>
            </div>
            <div class="depop-actions">
                <div class="feedback-buttons">
                    <button class="feedback-btn love-btn" data-feedback="love" data-item-id="${uniqueId}" title="Love it!">
                        <span class="double-thumbs">👍👍</span>
                    </button>
                    <button class="feedback-btn like-btn" data-feedback="like" data-item-id="${uniqueId}" title="Like">
                        <span class="single-thumb">👍</span>
                    </button>
                    <button class="feedback-btn dislike-btn" data-feedback="dislike" data-item-id="${uniqueId}" title="Not interested">
                        <span class="dislike-thumb">👎</span>
                    </button>
                </div>
            </div>
        `;

        return card;
    }

    attachItemHandlers() {
        // Card click handlers
        document.querySelectorAll('.depop-card').forEach(card => {
            card.removeEventListener('click', this.handleCardClick);
            card.addEventListener('click', this.handleCardClick);
        });

        // Feedback button handlers - use event delegation for dynamically loaded content
        document.removeEventListener('click', this.handleFeedbackDelegate);
        document.addEventListener('click', this.handleFeedbackDelegate.bind(this));

        // Image hover handlers
        document.querySelectorAll('.item-img').forEach(img => {
            const img1 = img.getAttribute('data-img1');
            const img2 = img.getAttribute('data-img2');
            if (img2 && img2 !== '') {
                img.removeEventListener('mouseenter', this.handleImageHover);
                img.removeEventListener('mouseleave', this.handleImageLeave);
                img.addEventListener('mouseenter', () => { img.src = img2; });
                img.addEventListener('mouseleave', () => { img.src = img1; });
            }
        });

        // Restore feedback states for currently displayed items
        this.restoreRatingStates();
    }

    async restoreRatingStates() {
        // Get all item URLs currently displayed
        const itemCards = document.querySelectorAll('.depop-card');
        const itemUrls = Array.from(itemCards).map(card => card.getAttribute('data-url')).filter(url => url);
        
        if (itemUrls.length === 0) return;

        try {
            const response = await fetch('/get_user_feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ item_urls: itemUrls })
            });

            if (!response.ok) return;

            const data = await response.json();
            const feedbackData = data.feedback || {};

            // Apply feedback states to buttons
            itemCards.forEach(card => {
                const itemUrl = card.getAttribute('data-url');
                const feedback = feedbackData[itemUrl];
                
                if (feedback && feedback !== 'none') {
                    // Clear all clicked states first
                    const allBtns = card.querySelectorAll('.feedback-btn');
                    allBtns.forEach(btn => btn.classList.remove('clicked'));
                    
                    // Apply the correct clicked state
                    const targetBtn = card.querySelector(`.${feedback}-btn`);
                    if (targetBtn) {
                        targetBtn.classList.add('clicked');
                    }
                }
            });

        } catch (error) {
            console.error('Error restoring rating states:', error);
        }
    }

    handleFeedbackDelegate(e) {
        // Check if the clicked element is a feedback button or within one
        const feedbackBtn = e.target.closest('.feedback-btn');
        if (feedbackBtn) {
            console.log('🎯 Feedback button clicked via delegation:', feedbackBtn);
            e.preventDefault();
            e.stopPropagation();
            this.handleFeedbackClick({
                currentTarget: feedbackBtn,
                preventDefault: () => {},
                stopPropagation: () => {}
            });
        }
    }

    handleFeedbackClick(e) {
        e.preventDefault();
        e.stopPropagation();
        
        const btn = e.currentTarget;
        console.log('🎯 Feedback button clicked:', btn);
        
        const feedback = btn.getAttribute('data-feedback');
        const itemId = btn.getAttribute('data-item-id');
        
        console.log('📝 Feedback type:', feedback);
        console.log('� Item ID:', itemId);
        
        if (!feedback) {
            console.error('❌ No feedback type found');
            return;
        }
        
        if (!itemId) {
            console.error('❌ No item ID found');
            return;
        }
        
        // Get item data from store instead of parsing JSON
        const itemData = this.itemDataStore.get(itemId);
        if (!itemData) {
            console.error('❌ Item data not found in store for ID:', itemId);
            console.error('📍 Available IDs in store:', Array.from(this.itemDataStore.keys()));
            return;
        }
        
        console.log('📦 Retrieved item data:', itemData);
        console.log('🔗 Item URL from data:', itemData.item_url);
        console.log('📝 Item title from data:', itemData.title);
        
        const itemCard = btn.closest('.depop-card');
        if (!itemCard) {
            console.error('❌ Could not find item card');
            return;
        }
        
        // Check if this button is already clicked (toggle functionality)
        const isCurrentlyClicked = btn.classList.contains('clicked');
        
        // Clear previous selections on this item
        const allFeedbackBtns = itemCard.querySelectorAll('.feedback-btn');
        allFeedbackBtns.forEach(b => b.classList.remove('clicked'));
        
        // If button was already clicked, remove the rating (toggle off)
        if (isCurrentlyClicked) {
            console.log('🔄 Toggling off rating');
            // Visual feedback for removal
            btn.style.transform = 'scale(0.8)';
            setTimeout(() => {
                btn.style.transform = 'scale(1)';
            }, 150);
            
            // Send removal feedback to server
            this.removeFeedback(itemData, btn);
            return;
        }
        
        // Mark this button as clicked
        btn.classList.add('clicked');
        
        // Visual feedback animation
        btn.style.transform = 'scale(1.2)';
        setTimeout(() => {
            btn.style.transform = 'scale(1)';
        }, 150);
        
        console.log('📤 Submitting feedback...');
        
        // Send feedback to server
        this.submitFeedback(feedback, itemData, btn);
    }

    submitFeedback(feedback, item, btn) {
        console.log('🚀 Starting feedback submission:', feedback, item);
        
        // Prevent duplicate submissions
        if (btn.dataset.submitting === 'true') {
            console.log('🚫 Already submitting feedback for this item, ignoring duplicate request');
            return;
        }
        btn.dataset.submitting = 'true';
        
        // Convert arrays to strings for database storage
        let itemSize = item.size || '';
        if (Array.isArray(itemSize)) {
            itemSize = itemSize.join(', ');
        }
        
        // Use all_sizes if available, otherwise use size
        if (item.all_sizes && Array.isArray(item.all_sizes)) {
            itemSize = item.all_sizes.join(', ');
        }

        // Handle category and subcategory arrays
        let itemCategory = this.currentFilters?.category || '';
        if (Array.isArray(itemCategory)) {
            itemCategory = itemCategory.join(', ');
        }

        let itemSubcategory = this.currentFilters?.subcategory || '';
        if (Array.isArray(itemSubcategory)) {
            itemSubcategory = itemSubcategory.join(', ');
        }

        const feedbackData = {
            feedback: feedback,
            item_url: item.item_url,
            item_title: item.title,
            item_brand: item.brand || '',
            item_sizes: itemSize,
            item_price: item.price || item.price_sale,
            item_image: item.image_url || item.image || '',
            item_category: itemCategory,
            item_subcategory: itemSubcategory,
            item_color: '', // Could extract from title/description
            item_condition: '', // Could extract from item data
            search_query: this.currentFilters?.query || '',
            search_filters: JSON.stringify(this.currentFilters || {})
        };

        console.log('📦 Feedback data being sent:', feedbackData);

        // Disable button during request
        btn.disabled = true;
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

        fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedbackData)
        })
        .then(response => {
            console.log('📡 Response status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Feedback response:', data);
            if (data.message) {
                console.log(`🧠 Model Learning: ${feedback} feedback recorded for "${item.title}"`);
                console.log(`📊 Training Progress: ${data.training_info || 'Neural network updated'}`);
                
                // Keep the button in clicked state to show it was successful
                btn.classList.add('clicked');
                this.showFeedbackSuccess(feedback);
            } else if (data.error) {
                console.error('❌ Server error:', data.error);
                // If there was an error, remove the clicked state
                btn.classList.remove('clicked');
                this.showFeedbackError(data.error);
            }
        })
        .catch(error => {
            console.error('❌ Error submitting feedback:', error);
            // Remove clicked state on error
            btn.classList.remove('clicked');
            this.showFeedbackError('Failed to submit feedback. Please try again.');
        })
        .finally(() => {
            // Re-enable button and clear submitting flag
            btn.disabled = false;
            btn.innerHTML = originalText;
            btn.dataset.submitting = 'false';
        });
    }

    removeFeedback(item, btn) {
        const feedbackData = {
            feedback: 'remove',
            item_url: item.item_url,
            item_title: item.title
        };

        fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedbackData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                console.log(`🧠 Model Learning: Rating removed for "${item.title}"`);
                console.log(`📊 Training Progress: ${data.training_info || 'Neural network updated'}`);
                this.showFeedbackSuccess('remove');
            } else {
                console.error('Error removing feedback');
            }
        })
        .catch(error => {
            console.error('Error removing feedback:', error);
        });
    }

    showFeedbackSuccess(feedback) {
        // Optional: Show a brief toast notification
        const message = feedback === 'love' ? 'Added to favorites! 💖' : 
                       feedback === 'like' ? 'Liked! 👍' : 
                       feedback === 'dislike' ? 'Not interested 👎' :
                       feedback === 'remove' ? 'Rating removed ↩️' : 'Updated!';
        
        // You could implement a toast notification here
        console.log(`✓ ${message}`);
        
        // Optional: Add a subtle visual indicator
        this.showBriefNotification(message);
    }

    showBriefNotification(message) {
        // Create a temporary notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 14px;
            z-index: 10000;
            transform: translateX(300px);
            transition: transform 0.3s ease;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);

        // Remove after 2 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(300px)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 2000);
    }

    showFeedbackError(message) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(220, 53, 69, 0.9);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 14px;
            z-index: 10000;
            transform: translateX(300px);
            transition: transform 0.3s ease;
            max-width: 300px;
        `;
        notification.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);

        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(300px)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    handleCardClick(e) {
        if (e.target.closest('button') || e.target.closest('form')) return;
        const url = this.getAttribute('data-url');
        if (url) window.open(url, '_blank');
    }

    displayError(message) {
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.innerHTML = `<p class="text-center text-danger">${message}</p>`;
        }
    }

    updateLoadingState(isLoading) {
        if (isLoading) {
            this.showSpinner();
        } else {
            this.hideSpinner();
        }
    }

    setupSpinner() {
        if (!document.getElementById('infiniteSpinner')) {
            const spinner = document.createElement('div');
            spinner.id = 'infiniteSpinner';
            spinner.style.display = 'none';
            spinner.style.justifyContent = 'center';
            spinner.style.alignItems = 'center';
            spinner.style.width = '100%';
            spinner.innerHTML = '<div style="margin:2rem auto; display:inline-block;"><span class="spinner" style="display:inline-block;width:32px;height:32px;border:4px solid #cbd5e1;border-top:4px solid #6366f1;border-radius:50%;animation:spin 1s linear infinite;"></span></div>';
            document.body.appendChild(spinner);
            
            // Add spinner animation CSS
            if (!document.getElementById('spinnerStyle')) {
                const style = document.createElement('style');
                style.id = 'spinnerStyle';
                style.innerHTML = `@keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }`;
                document.head.appendChild(style);
            }
        }
    }

    showSpinner() {
        const spinner = document.getElementById('infiniteSpinner');
        if (spinner) {
            spinner.style.display = 'flex';
            
            // Position spinner appropriately
            const grid = document.querySelector('.results-grid');
            const resultsContainer = document.getElementById('resultsContainer');
            
            if (grid && spinner.parentNode !== grid.parentNode) {
                grid.insertAdjacentElement('afterend', spinner);
            } else if (resultsContainer && !grid && spinner.parentNode !== resultsContainer) {
                resultsContainer.appendChild(spinner);
            }
        }
    }

    hideSpinner() {
        const spinner = document.getElementById('infiniteSpinner');
        if (spinner) {
            spinner.style.display = 'none';
        }
    }

    shouldAutoLoad() {
        const scrollPosition = window.scrollY + window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        return scrollPosition >= documentHeight - 300;
    }

    setupInfiniteScroll() {
        let ticking = false;
        
        const handleScroll = () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    if (!this.isLoading && this.hasMoreResults && this.nextCursor) {
                        const scrollPosition = window.scrollY + window.innerHeight;
                        const documentHeight = document.documentElement.scrollHeight;
                        
                        // Load more when user is within 300px of bottom
                        if (scrollPosition >= documentHeight - 300) {
                            this.performSearch(true);
                        }
                    }
                    
                    ticking = false;
                });
                
                ticking = true;
            }
        };

        window.addEventListener('scroll', handleScroll);
    }
}

// Initialize search manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.searchManager = new SearchManager();
});
