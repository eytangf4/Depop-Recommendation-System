class SearchManager {
    constructor() {
        this.currentFilters = {};
        this.nextCursor = null;
        this.isLoading = false;
        this.hasMoreResults = true;
        this.loadedItemUrls = new Set();
        this.setupSearchHandlers();
        this.setupInfiniteScroll();
        this.setupSpinner();
        this.setupMobileFilters();
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
                <form method="post" action="/feedback" class="feedback-form">
                    <input type="hidden" name="item_url" value="${item.item_url}">
                    <button type="submit" name="feedback" value="like" class="like-btn emoji-circle" title="Like">&#128077;</button>
                    <button type="submit" name="feedback" value="dislike" class="dislike-btn emoji-circle" title="Dislike">&#128078;</button>
                </form>
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
