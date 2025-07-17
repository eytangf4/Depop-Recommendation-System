// Dynamic Filters Manager
class DepopFilters {
    constructor() {
        this.data = {
            categories: null,
            brands: null,
            colors: null,
            sizes: null,
            conditions: null
        };
        this.selectedCategory = null; // Single category (men, women, kids, everything_else)
        this.selectedMidlevels = new Set(); // Track selected midlevels with main/midlevel format
        this.selectedSubcategories = new Set(); // Multi-select
        this.selectedBrands = new Set(); // Multi-select
        this.selectedSizes = new Set(); // Multi-select
        this.selectedColors = new Set(); // Multi-select
        this.selectedConditions = new Set(); // Multi-select
        this.expandedCategories = new Set(); // Track which main categories are expanded
        this.currentSort = 'relevance';
        this.onSale = false;
        this.init();
    }

    async init() {
        await this.loadFilterData();
        this.populateStaticFilters();
        this.setupEventListeners();
        this.setupActiveFilters();
        
        // Debug: Check if Bootstrap is loaded
        if (typeof bootstrap === 'undefined') {
            console.error('Bootstrap JavaScript is not loaded!');
        } else {
            console.log('Bootstrap loaded successfully');
        }
    }

    async loadFilterData() {
        try {
            // Load all filter data files
            const [categories, brands, colors, sizes, conditions] = await Promise.all([
                fetch('/static/json/depop_categories.json').then(r => r.json()),
                fetch('/static/json/depop_brands.json').then(r => r.json()),
                fetch('/static/json/depop_colors.json').then(r => r.json()),
                fetch('/static/json/depop_sizes.json').then(r => r.json()),
                fetch('/static/json/depop_conditions.json').then(r => r.json())
            ]);

            this.data = { categories, brands, colors, sizes, conditions };
        } catch (error) {
            console.error('Error loading filter data:', error);
        }
    }

    populateStaticFilters() {
        this.populateColors();
        this.populateConditions();
        this.populateBrands();
        this.populateSizes(); // Add sizes to initial load
    }

    populateColors() {
        const colorList = document.getElementById('colorList');
        if (!colorList || !this.data.colors) return;

        // Color mapping for visual circles
        const colorMap = {
            'black': '#000000',
            'white': '#FFFFFF',
            'grey': '#808080',
            'gray': '#808080',
            'red': '#DC143C',
            'blue': '#0066CC',
            'green': '#228B22',
            'yellow': '#FFD700',
            'pink': '#FF69B4',
            'purple': '#800080',
            'orange': '#FF8C00',
            'brown': '#8B4513',
            'beige': '#F5F5DC',
            'navy': '#000080',
            'burgundy': '#800020',
            'gold': '#FFD700',
            'silver': '#C0C0C0',
            'cream': '#FFFDD0',
            'olive': '#808000',
            'maroon': '#800000',
            'teal': '#008080',
            'turquoise': '#40E0D0',
            'lime': '#00FF00',
            'multicolor': 'linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet)'
        };

        colorList.innerHTML = '';
        this.data.colors.forEach(color => {
            const item = document.createElement('div');
            item.className = 'dropdown-checkbox';
            
            const colorKey = color.name.toLowerCase();
            const colorValue = colorMap[colorKey] || '#CCCCCC';
            
            // Special handling for white color (add border)
            const borderStyle = colorKey === 'white' ? 'border: 2px solid #ddd;' : '';
            const backgroundStyle = colorKey === 'multicolor' ? 
                `background: ${colorValue};` : 
                `background-color: ${colorValue};`;
            
            item.innerHTML = `
                <input type="checkbox" id="color-${color.name}" value="${color.name}">
                <span class="color-circle" style="${backgroundStyle}${borderStyle}"></span>
                <label for="color-${color.name}">${color.name}</label>
            `;
            colorList.appendChild(item);
        });

        this.attachMultiSelectHandlers('colorList', this.selectedColors, 'colorLabel', 'Color');
    }

    populateConditions() {
        const conditionList = document.getElementById('conditionList');
        if (!conditionList || !this.data.conditions) return;

        conditionList.innerHTML = '';
        this.data.conditions.forEach(condition => {
            const item = document.createElement('div');
            item.className = 'dropdown-checkbox mb-3';
            item.innerHTML = `
                <div class="d-flex align-items-start">
                    <input type="checkbox" id="condition-${condition.value}" value="${condition.value}" class="me-2 mt-1">
                    <div class="flex-grow-1">
                        <label for="condition-${condition.value}" class="mb-1 fw-medium">${condition.name}</label>
                        <div class="condition-description text-muted small">${condition.description}</div>
                    </div>
                </div>
            `;
            conditionList.appendChild(item);
        });

        this.attachMultiSelectHandlers('conditionList', this.selectedConditions, 'conditionLabel', 'Condition');
    }

    populateBrands() {
        const brandList = document.getElementById('brandList');
        if (!brandList || !this.data.brands) return;

        // Add search box at the top
        const searchBox = document.createElement('div');
        searchBox.className = 'p-2 border-bottom sticky-top bg-white';
        searchBox.innerHTML = `
            <input type="text" class="form-control form-control-sm" id="brandSearchInput" placeholder="Search brands...">
        `;
        
        brandList.innerHTML = '';
        brandList.appendChild(searchBox);

        // Container for brands
        const brandsContainer = document.createElement('div');
        brandsContainer.id = 'brandsContainer';
        brandsContainer.style.maxHeight = '300px';
        brandsContainer.style.overflowY = 'auto';
        
        // Initial load variables
        this.brandOffset = 0;
        this.brandBatchSize = 50;
        this.allBrands = this.data.brands.brands || [];
        this.filteredBrands = [...this.allBrands];
        this.isLoadingBrands = false;
        
        // Load initial brands
        this.loadMoreBrands(brandsContainer);
        
        // Add scroll listener for infinite scroll
        brandsContainer.addEventListener('scroll', () => {
            if (brandsContainer.scrollTop + brandsContainer.clientHeight >= brandsContainer.scrollHeight - 5) {
                this.loadMoreBrands(brandsContainer);
            }
        });
        
        // Add search functionality
        const searchInput = searchBox.querySelector('#brandSearchInput');
        searchInput.addEventListener('input', (e) => {
            this.searchBrands(e.target.value, brandsContainer);
        });

        brandList.appendChild(brandsContainer);
        this.attachMultiSelectHandlers('brand', this.selectedBrands, 'brandLabel');
    }

    loadMoreBrands(container) {
        if (this.isLoadingBrands) return;
        
        this.isLoadingBrands = true;
        const endIndex = Math.min(this.brandOffset + this.brandBatchSize, this.filteredBrands.length);
        const brandsToLoad = this.filteredBrands.slice(this.brandOffset, endIndex);
        
        brandsToLoad.forEach(brand => {
            const item = document.createElement('div');
            item.className = 'dropdown-checkbox';
            item.setAttribute('data-brand-name', brand.brand_name.toLowerCase());
            const safeId = brand.brand_id || brand.brand_name.replace(/[^a-zA-Z0-9]/g, '');
            item.innerHTML = `
                <input type="checkbox" id="brand-${safeId}" value="${brand.brand_id}">
                <label for="brand-${safeId}">${brand.brand_name}</label>
            `;
            container.appendChild(item);
        });
        
        this.brandOffset = endIndex;
        this.isLoadingBrands = false;
        
        // Add loading indicator if there are more brands
        if (endIndex < this.filteredBrands.length) {
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'text-center p-2 text-muted';
            loadingIndicator.innerHTML = '<small>Scroll for more...</small>';
            loadingIndicator.id = 'brandLoadingIndicator';
            container.appendChild(loadingIndicator);
        }
    }

    searchBrands(searchTerm, container) {
        // Clear existing brands
        container.innerHTML = '';
        
        // Filter brands based on search term
        if (searchTerm.trim() === '') {
            this.filteredBrands = [...this.allBrands];
        } else {
            const lowerSearchTerm = searchTerm.toLowerCase();
            this.filteredBrands = this.allBrands.filter(brand => 
                brand.brand_name.toLowerCase().includes(lowerSearchTerm)
            );
        }
        
        // Reset offset and load brands
        this.brandOffset = 0;
        this.loadMoreBrands(container);

        brandList.appendChild(brandsContainer);
        this.attachMultiSelectHandlers('brandList', this.selectedBrands, 'brandLabel', 'Brand');
    }

    populateSubcategories(category) {
        const subcategoryList = document.getElementById('subcategoryList');
        const subcategoryContainer = document.getElementById('subcategoryContainer');
        
        if (!subcategoryList || !this.data.categories || !category) {
            subcategoryContainer.classList.add('d-none');
            return;
        }

        // Clear existing subcategories
        this.selectedSubcategories.clear();
        subcategoryList.innerHTML = '';

        let subcategories = [];
        const categoryMap = {
            'men': 'Men',
            'women': 'Women', 
            'kids': 'Kids',
            'everything_else': 'Everything Else'
        };

        const categoryKey = categoryMap[category];
        if (this.data.categories[categoryKey]) {
            // Get all subcategories for this category
            Object.keys(this.data.categories[categoryKey]).forEach(mainCat => {
                const subCats = this.data.categories[categoryKey][mainCat];
                if (Array.isArray(subCats)) {
                    subCats.forEach(subcat => {
                        if (subcat.value) {
                            subcategories.push({
                                value: subcat.value,
                                name: subcat.value.replace(/^category-/, '').replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
                            });
                        }
                    });
                }
            });
        }

        // Remove duplicates
        const uniqueSubcategories = subcategories.filter((item, index, self) => 
            index === self.findIndex(t => t.value === item.value)
        );

        if (uniqueSubcategories.length > 0) {
            uniqueSubcategories.forEach(subcat => {
                const item = document.createElement('div');
                item.className = 'dropdown-checkbox';
                item.innerHTML = `
                    <input type="checkbox" id="subcat-${subcat.value}" value="${subcat.value}">
                    <label for="subcat-${subcat.value}">${subcat.name}</label>
                `;
                subcategoryList.appendChild(item);
            });

            this.attachMultiSelectHandlers('subcategoryList', this.selectedSubcategories, 'subcategoryLabel', 'Subcategory');
            subcategoryContainer.classList.remove('d-none');
        } else {
            subcategoryContainer.classList.add('d-none');
        }

        this.populateSizes(category);
    }

    populateSizes(category = null) {
        const sizeList = document.getElementById('sizeList');
        if (!sizeList) return;

        this.selectedSizes.clear();
        sizeList.innerHTML = '';

        // Use standard sizes that work with Depop API
        let sizes = [];
        
        // Determine size category based on selected category
        if (category === 'men' || category === 'women') {
            sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'One size'];
        } else if (category === 'kids') {
            sizes = ['2T', '3T', '4T', '5T', '6', '7', '8', '10', '12', '14', '16'];
        } else {
            // Default to common adult sizes (when no category or unknown category)
            sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'One size'];
        }

        sizes.forEach(size => {
            const item = document.createElement('div');
            item.className = 'dropdown-checkbox';
            item.innerHTML = `
                <input type="checkbox" id="size-${size}" value="${size}">
                <label for="size-${size}">${size}</label>
            `;
            sizeList.appendChild(item);
        });

        this.attachMultiSelectHandlers('sizeList', this.selectedSizes, 'sizeLabel', 'Size');
    }

    attachMultiSelectHandlers(listId, selectedSet, labelId, defaultLabel) {
        const list = document.getElementById(listId);
        if (!list) return;

        // For brand list, look in the brands container, for others look directly in the list
        const container = listId === 'brandList' ? 
            list.querySelector('#brandsContainer') || list : 
            list;

        const checkboxes = container.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                if (checkbox.checked) {
                    selectedSet.add(checkbox.value);
                } else {
                    selectedSet.delete(checkbox.value);
                }
                this.updateLabel(labelId, selectedSet, defaultLabel);
                this.updateActiveFilters();
                this.triggerSearch(); // Live update
            });
        });

        // Prevent dropdown from closing when clicking inside
        list.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    updateLabel(labelId, selectedSet, defaultLabel) {
        const label = document.getElementById(labelId);
        if (!label) return;

        if (selectedSet.size === 0) {
            label.textContent = defaultLabel;
        } else if (selectedSet.size === 1) {
            const value = Array.from(selectedSet)[0];
            const displayValue = value.replace(/^category-/, '').replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            label.textContent = displayValue;
        } else {
            label.textContent = `${selectedSet.size} selected`;
        }
    }

    setupEventListeners() {
        // Category selection (hierarchical)
        const categoryItems = document.querySelectorAll('#categoryList .dropdown-item');
        categoryItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation(); // Prevent dropdown from closing
                
                const category = e.target.getAttribute('data-category');
                const subcategory = e.target.getAttribute('data-subcategory');
                
                if (category) {
                    // Main category selection (Men, Women, etc.) - just toggle expansion
                    this.showCategoryHierarchy(category);
                    
                    // Update size filter based on category
                    this.populateSizes(category);
                } else if (subcategory) {
                    // Subcategory selection (T-shirts, Jeans, etc.)
                    if (this.selectedSubcategories.has(subcategory)) {
                        this.selectedSubcategories.delete(subcategory);
                        e.target.classList.remove('active');
                    } else {
                        this.selectedSubcategories.add(subcategory);
                        e.target.classList.add('active');
                    }
                    this.updateActiveFilters();
                    this.triggerSearch();
                }
            });
        });

        // On Sale toggle (now a checkbox)
        const onSaleToggle = document.getElementById('onSaleToggle');
        if (onSaleToggle) {
            onSaleToggle.addEventListener('change', () => {
                this.onSale = onSaleToggle.checked;
                this.updateActiveFilters();
                this.triggerSearch(); // Live update
            });
        }

        // Sort dropdown
        const sortItems = document.querySelectorAll('#sortList .dropdown-item');
        sortItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.currentSort = e.target.getAttribute('data-sort');
                const sortLabel = document.getElementById('sortLabel');
                if (sortLabel) {
                    // Update just the text, keeping the simple "Sort" format for now
                    // You can customize this to show the selected sort if desired
                    sortLabel.textContent = 'Sort';
                }
                
                // Update active state
                sortItems.forEach(si => si.classList.remove('active'));
                e.target.classList.add('active');
                
                this.updateActiveFilters();
                this.triggerSearch(); // Live update
            });
        });

        // Clear filters button
        const clearBtn = document.getElementById('clearFiltersBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearAllFilters();
            });
        }

        // Price inputs with live update
        ['priceMinInput', 'priceMaxInput'].forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                let timeout;
                input.addEventListener('input', () => {
                    clearTimeout(timeout);
                    timeout = setTimeout(() => {
                        this.updateActiveFilters();
                        this.triggerSearch(); // Live update
                    }, 500); // Debounce for 500ms
                });
            }
        });
    }

    showCategoryHierarchy(mainCategory) {
        const categoryList = document.getElementById('categoryList');
        if (!categoryList || !this.data.categories) return;

        // Get the midlevel categories for this main category
        const categoryMap = {
            'men': 'Men',
            'women': 'Women', 
            'kids': 'Kids',
            'everything_else': 'Everything Else'
        };

        const categoryKey = categoryMap[mainCategory];
        if (!this.data.categories[categoryKey]) return;

        // Check if this category is already expanded
        if (this.expandedCategories.has(mainCategory)) {
            // Collapse - remove midlevel items and collapse state
            this.expandedCategories.delete(mainCategory);
            this.removeMidlevelItems(mainCategory);
            return;
        }

        // Expand - first collapse any other expanded categories
        this.expandedCategories.forEach(expandedCat => {
            if (expandedCat !== mainCategory) {
                this.removeMidlevelItems(expandedCat);
                this.expandedCategories.delete(expandedCat);
            }
        });

        // Add this category to expanded set
        this.expandedCategories.add(mainCategory);

        // Find the main category item that was clicked
        const mainCategoryItem = categoryList.querySelector(`[data-category="${mainCategory}"]`);
        if (!mainCategoryItem) return;

        const mainCategoryLi = mainCategoryItem.closest('li');
        
        // Add midlevel categories right after the main category with checkboxes
        const midlevels = Object.keys(this.data.categories[categoryKey]);
        let insertAfter = mainCategoryLi;
        
        midlevels.forEach((midlevel) => {
            const li = document.createElement('li');
            li.className = 'midlevel-category-item';
            li.setAttribute('data-parent-category', mainCategory);
            
            const midlevelKey = `${mainCategory}/${midlevel.toLowerCase()}`;
            const isSelected = this.selectedMidlevels.has(midlevelKey);
            
            li.innerHTML = `
                <div class="dropdown-checkbox">
                    <input type="checkbox" id="mid-${mainCategory}-${midlevel.toLowerCase().replace(/\s+/g, '_')}" 
                           value="${midlevelKey}" ${isSelected ? 'checked' : ''}>
                    <label for="mid-${mainCategory}-${midlevel.toLowerCase().replace(/\s+/g, '_')}" class="ms-2">${midlevel}</label>
                </div>
            `;
            
            // Insert after the current position
            if (insertAfter.nextElementSibling) {
                categoryList.insertBefore(li, insertAfter.nextElementSibling);
            } else {
                categoryList.appendChild(li);
            }
            
            // Update insertAfter to this new element for the next iteration
            insertAfter = li;
            
            // Add event listener for the checkbox
            const checkbox = li.querySelector('input[type="checkbox"]');
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation(); // Prevent dropdown from closing
                
                const midlevelValue = checkbox.value;
                if (checkbox.checked) {
                    this.selectedMidlevels.add(midlevelValue);
                    // Show subcategories for this midlevel
                    this.showSubcategories(mainCategory, midlevel);
                } else {
                    this.selectedMidlevels.delete(midlevelValue);
                    // Hide subcategories if no midlevel selected
                    if (this.selectedMidlevels.size === 0) {
                        this.hideSubcategories();
                    }
                }
                
                this.updateCategoryLabel();
                this.updateActiveFilters();
                this.triggerSearch();
            });
        });
    }

    removeMidlevelItems(category) {
        const categoryList = document.getElementById('categoryList');
        const midlevelItems = categoryList.querySelectorAll(`[data-parent-category="${category}"]`);
        midlevelItems.forEach(item => item.remove());
    }

    updateCategoryLabel() {
        const categoryLabel = document.getElementById('categoryLabel');
        if (!categoryLabel) return;

        if (this.selectedMidlevels.size === 0) {
            categoryLabel.textContent = 'Category';
        } else if (this.selectedMidlevels.size === 1) {
            const selected = Array.from(this.selectedMidlevels)[0];
            const [mainCat, midCat] = selected.split('/');
            const displayMain = this.getCategoryDisplayName(mainCat);
            const displayMid = midCat.charAt(0).toUpperCase() + midCat.slice(1);
            categoryLabel.textContent = `${displayMain}/${displayMid}`;
        } else {
            categoryLabel.textContent = `${this.selectedMidlevels.size} selected`;
        }
    }

    hideSubcategories() {
        const subcategoryContainer = document.getElementById('subcategoryContainer');
        if (subcategoryContainer) {
            subcategoryContainer.classList.add('d-none');
        }
    }

    getCategoryDisplayName(category) {
        const displayMap = {
            'men': 'Men',
            'women': 'Women',
            'kids': 'Kids', 
            'everything_else': 'Everything Else'
        };
        return displayMap[category] || category;
    }

    showSubcategories(mainCategory, midlevel) {
        const subcategoryContainer = document.getElementById('subcategoryContainer');
        const subcategoryList = document.getElementById('subcategoryList');
        
        if (!subcategoryContainer || !subcategoryList || !this.data.categories) return;

        const categoryKey = this.getCategoryDisplayName(mainCategory);
        const categoryData = this.data.categories[categoryKey];
        
        if (!categoryData || !categoryData[midlevel]) return;

        // Show the subcategory container
        subcategoryContainer.classList.remove('d-none');
        
        // Clear existing subcategories
        subcategoryList.innerHTML = '';
        
        // Add subcategories as checkboxes
        const subcategories = categoryData[midlevel];
        subcategories.forEach(subcat => {
            const item = document.createElement('div');
            item.className = 'dropdown-checkbox mb-2';
            item.innerHTML = `
                <input type="checkbox" id="subcat-${subcat.value}" value="${subcat.value}">
                <label for="subcat-${subcat.value}" class="ms-2">${subcat.name}</label>
            `;
            subcategoryList.appendChild(item);
            
            // Add event listener directly to this checkbox
            const checkbox = item.querySelector('input[type="checkbox"]');
            checkbox.addEventListener('change', () => {
                if (checkbox.checked) {
                    this.selectedSubcategories.add(checkbox.value);
                } else {
                    this.selectedSubcategories.delete(checkbox.value);
                }
                this.updateLabel('subcategoryLabel', this.selectedSubcategories, 'Subcategory');
                this.updateActiveFilters();
                this.triggerSearch();
            });
        });

        // Prevent dropdown from closing when clicking inside
        subcategoryList.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    triggerSearch() {
        // Trigger search with current filters
        if (window.searchManager) {
            window.searchManager.performSearch();
        }
    }

    setupActiveFilters() {
        this.updateActiveFilters();
    }

    updateActiveFilters() {
        const activeFiltersRow = document.getElementById('activeFiltersRow');
        const activeFiltersPills = document.getElementById('activeFiltersPills');
        
        if (!activeFiltersPills) return;

        // Clear existing pills
        activeFiltersPills.innerHTML = '';

        let hasActiveFilters = false;

        // Add midlevel filter pills (showing as "Main/Midlevel" format)
        if (this.selectedMidlevels.size > 0) {
            hasActiveFilters = true;
            this.selectedMidlevels.forEach(midlevelKey => {
                const [main, mid] = midlevelKey.split('/');
                const displayMain = this.getCategoryDisplayName(main);
                const displayMid = mid.charAt(0).toUpperCase() + mid.slice(1);
                const displayValue = `${displayMain}/${displayMid}`;
                
                const pill = document.createElement('span');
                pill.className = 'filter-pill';
                pill.innerHTML = `
                    <span>${displayValue}</span>
                    <button class="remove-filter" data-filter="midlevel" data-value="${midlevelKey}">&times;</button>
                `;
                
                const removeBtn = pill.querySelector('.remove-filter');
                removeBtn.addEventListener('click', () => this.removeMidlevelFilter(midlevelKey));
                
                activeFiltersPills.appendChild(pill);
            });
        }

        // Add other filter pills
        const otherFilters = {
            subcategory: Array.from(this.selectedSubcategories),
            brand: Array.from(this.selectedBrands),
            size: Array.from(this.selectedSizes),
            color: Array.from(this.selectedColors),
            condition: Array.from(this.selectedConditions)
        };

        Object.entries(otherFilters).forEach(([key, value]) => {
            if (value && Array.isArray(value) && value.length > 0) {
                hasActiveFilters = true;
                const pill = this.createFilterPill(key, value);
                activeFiltersPills.appendChild(pill);
            }
        });

        // Add price filters
        const priceMin = document.getElementById('priceMinInput')?.value;
        const priceMax = document.getElementById('priceMaxInput')?.value;
        if (priceMin || priceMax) {
            hasActiveFilters = true;
            let priceText = 'Price: ';
            if (priceMin && priceMax) {
                priceText += `$${priceMin} - $${priceMax}`;
            } else if (priceMin) {
                priceText += `$${priceMin}+`;
            } else {
                priceText += `Under $${priceMax}`;
            }
            
            const pill = document.createElement('span');
            pill.className = 'filter-pill';
            pill.innerHTML = `
                <span>${priceText}</span>
                <button class="remove-filter" data-filter="price">&times;</button>
            `;
            
            const removeBtn = pill.querySelector('.remove-filter');
            removeBtn.addEventListener('click', () => this.removePriceFilter());
            
            activeFiltersPills.appendChild(pill);
        }

        // Add on-sale filter
        if (this.onSale) {
            hasActiveFilters = true;
            const pill = document.createElement('span');
            pill.className = 'filter-pill';
            pill.innerHTML = `
                <span>On sale</span>
                <button class="remove-filter" data-filter="on_sale">&times;</button>
            `;
            
            const removeBtn = pill.querySelector('.remove-filter');
            removeBtn.addEventListener('click', () => this.removeOnSaleFilter());
            
            activeFiltersPills.appendChild(pill);
        }

        // Show/hide active filters row
        if (activeFiltersRow) {
            activeFiltersRow.style.display = hasActiveFilters ? 'block' : 'none';
        }
    }

    removeMidlevelFilter(midlevelKey) {
        this.selectedMidlevels.delete(midlevelKey);
        
        // Uncheck the corresponding checkbox
        const [main, mid] = midlevelKey.split('/');
        const checkbox = document.getElementById(`mid-${main}-${mid.replace(/\s+/g, '_')}`);
        if (checkbox) {
            checkbox.checked = false;
        }
        
        this.updateCategoryLabel();
        this.updateActiveFilters();
        this.triggerSearch();
    }

    removePriceFilter() {
        document.getElementById('priceMinInput').value = '';
        document.getElementById('priceMaxInput').value = '';
        this.updateActiveFilters();
        this.triggerSearch();
    }

    removeOnSaleFilter() {
        this.onSale = false;
        const onSaleToggle = document.getElementById('onSaleToggle');
        if (onSaleToggle) {
            onSaleToggle.checked = false;
        }
        this.updateActiveFilters();
        this.triggerSearch();
    }

    getBrandNameFromId(brandId) {
        if (!this.allBrands || !brandId) return null;
        const brand = this.allBrands.find(b => b.brand_id === brandId);
        return brand ? brand.brand_name : null;
    }

    createFilterPill(key, value) {
        const pill = document.createElement('span');
        pill.className = 'filter-pill';
        
        let displayValue = value;
        if (Array.isArray(value)) {
            if (key === 'brand') {
                // Convert brand IDs to brand names for display
                const brandNames = value.map(brandId => this.getBrandNameFromId(brandId)).filter(name => name);
                displayValue = brandNames.length > 1 ? `${brandNames.length} selected` : brandNames[0] || value[0];
            } else {
                displayValue = value.length > 1 ? `${value.length} selected` : value[0];
            }
        } else if (key === 'brand') {
            // Handle single brand value
            displayValue = this.getBrandNameFromId(value) || value;
        }
        
        pill.innerHTML = `
            <span>${this.getFilterDisplayName(key)}: ${displayValue}</span>
            <button class="remove-filter" data-filter="${key}">&times;</button>
        `;
        
        const removeBtn = pill.querySelector('.remove-filter');
        removeBtn.addEventListener('click', () => this.removeFilter(key));
        
        return pill;
    }

    getFilterDisplayName(key) {
        const displayNames = {
            'category': 'Category',
            'subcategory': 'Subcategory',
            'brand': 'Brand',
            'size': 'Size',
            'color': 'Color',
            'condition': 'Condition',
            'price_min': 'Min Price',
            'price_max': 'Max Price',
            'on_sale': 'On Sale',
            'sort': 'Sort'
        };
        return displayNames[key] || key;
    }

    removeFilter(key) {
        if (key === 'category') {
            this.selectedCategory = null;
            document.getElementById('categoryLabel').textContent = 'Category';
            document.querySelectorAll('#categoryList .dropdown-item').forEach(item => {
                item.classList.remove('active');
            });
            document.getElementById('subcategoryContainer').classList.add('d-none');
        } else if (key === 'subcategory') {
            this.selectedSubcategories.clear();
            document.querySelectorAll('#subcategoryList input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            this.updateLabel('subcategoryLabel', this.selectedSubcategories, 'Subcategory');
        } else if (key === 'brand') {
            this.selectedBrands.clear();
            document.querySelectorAll('#brandList input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            this.updateLabel('brandLabel', this.selectedBrands, 'Brand');
        } else if (key === 'size') {
            this.selectedSizes.clear();
            document.querySelectorAll('#sizeList input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            this.updateLabel('sizeLabel', this.selectedSizes, 'Size');
        } else if (key === 'color') {
            this.selectedColors.clear();
            document.querySelectorAll('#colorList input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            this.updateLabel('colorLabel', this.selectedColors, 'Color');
        } else if (key === 'condition') {
            this.selectedConditions.clear();
            document.querySelectorAll('#conditionList input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            this.updateLabel('conditionLabel', this.selectedConditions, 'Condition');
        } else if (key === 'price_min') {
            document.getElementById('priceMinInput').value = '';
        } else if (key === 'price_max') {
            document.getElementById('priceMaxInput').value = '';
        } else if (key === 'on_sale') {
            this.onSale = false;
            document.getElementById('onSaleToggle').setAttribute('data-active', 'false');
        } else if (key === 'sort') {
            this.currentSort = 'relevance';
            document.getElementById('sortLabel').textContent = 'Sort: Relevance';
            document.querySelectorAll('#sortList .dropdown-item').forEach(item => {
                item.classList.remove('active');
                if (item.getAttribute('data-sort') === 'relevance') {
                    item.classList.add('active');
                }
            });
        }

        this.updateActiveFilters();
        this.triggerSearch(); // Live update
    }

    clearAllFilters() {
        // Clear all selections
        this.selectedCategory = null;
        this.selectedMidlevel = null;
        this.selectedSubcategories.clear();
        this.selectedBrands.clear();
        this.selectedSizes.clear();
        this.selectedColors.clear();
        this.selectedConditions.clear();
        this.onSale = false;
        this.currentSort = 'relevance';

        // Reset UI
        document.getElementById('categoryLabel').textContent = 'Category';
        document.getElementById('subcategoryLabel').textContent = 'Subcategory';
        document.getElementById('brandLabel').textContent = 'Brand';
        document.getElementById('sizeLabel').textContent = 'Size';
        document.getElementById('colorLabel').textContent = 'Colour'; // Use UK spelling like Depop
        document.getElementById('conditionLabel').textContent = 'Condition';
        document.getElementById('sortLabel').textContent = 'Sort';

        // Reset price inputs
        const priceMinInput = document.getElementById('priceMinInput');
        const priceMaxInput = document.getElementById('priceMaxInput');
        if (priceMinInput) priceMinInput.value = '';
        if (priceMaxInput) priceMaxInput.value = '';
        
        // Reset price label
        const priceLabel = document.getElementById('priceLabel');
        if (priceLabel) priceLabel.textContent = 'Price';

        // Reset on sale checkbox
        const onSaleElement = document.getElementById('onSaleToggle');
        if (onSaleElement) onSaleElement.checked = false;

        // Reset category dropdown to original state and remove subcategories
        const categoryList = document.getElementById('categoryList');
        if (categoryList) {
            // Remove all midlevel categories first
            const midlevelItems = categoryList.querySelectorAll('[data-midlevel]');
            midlevelItems.forEach(item => item.remove());
            
            // Reset main categories to original state
            categoryList.innerHTML = `
                <li><a class="dropdown-item" href="#" data-category="men">Men</a></li>
                <li><a class="dropdown-item" href="#" data-category="women">Women</a></li>
                <li><a class="dropdown-item" href="#" data-category="kids">Kids</a></li>
                <li><a class="dropdown-item" href="#" data-category="everything_else">Everything Else</a></li>
            `;
            
            // Re-attach event listeners to category items
            const categoryItems = categoryList.querySelectorAll('.dropdown-item');
            categoryItems.forEach(item => {
                item.addEventListener('click', (e) => {
                    e.preventDefault();
                    const category = e.target.getAttribute('data-category');
                    
                    if (category) {
                        // Main category selection
                        this.selectedCategory = category;
                        document.getElementById('categoryLabel').textContent = e.target.textContent;
                        
                        // Show subcategories in the same dropdown
                        this.showCategoryHierarchy(category);
                        
                        // Update size filter based on category
                        this.populateSizes(category);
                        
                        this.updateActiveFilters();
                        this.triggerSearch();
                    }
                });
            });
        }

        // Clear all checkboxes
        document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.checked = false;
        });

        // Clear price inputs
        document.getElementById('priceMinInput').value = '';
        document.getElementById('priceMaxInput').value = '';

        // Reset sort selection
        document.querySelectorAll('#sortList .dropdown-item').forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('data-sort') === 'relevance') {
                item.classList.add('active');
            }
        });

        // Hide subcategory container
        document.getElementById('subcategoryContainer').classList.add('d-none');

        // Clear brand search
        const brandSearchInput = document.getElementById('brandSearchInput');
        if (brandSearchInput) {
            brandSearchInput.value = '';
            // Show all brands
            const brandItems = document.querySelectorAll('#brandsContainer .dropdown-checkbox');
            brandItems.forEach(item => {
                item.style.display = 'flex';
            });
        }

        this.updateActiveFilters();
        this.triggerSearch(); // Live update
    }

    getFilterValues() {
        // Extract main category and midlevel from selectedMidlevels
        let mainCategory = null;
        let midlevelCategory = null;
        
        if (this.selectedMidlevels.size > 0) {
            // For now, take the first selected midlevel to determine main category
            const firstSelected = Array.from(this.selectedMidlevels)[0];
            const [main, mid] = firstSelected.split('/');
            mainCategory = main;
            midlevelCategory = mid;
        }
        
        return {
            category: mainCategory,
            midlevel: midlevelCategory,
            subcategory: Array.from(this.selectedSubcategories),
            brand: Array.from(this.selectedBrands),
            size: Array.from(this.selectedSizes),
            color: Array.from(this.selectedColors),
            condition: Array.from(this.selectedConditions),
            price_min: document.getElementById('priceMinInput')?.value || '',
            price_max: document.getElementById('priceMaxInput')?.value || '',
            on_sale: this.onSale,
            sort: this.currentSort
        };
    }
}

// Initialize filters when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.depopFilters = new DepopFilters();
});
