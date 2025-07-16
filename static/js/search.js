const form = document.getElementById('searchForm');
const resultsContainer = document.getElementById('resultsContainer');
form.addEventListener('submit', function(e) {
    e.preventDefault();
    const query = document.getElementById('queryInput').value;
    const group = document.getElementById('groupInput').value;
    const gender = document.getElementById('genderInput').value;
    const brand = document.getElementById('brandInput').value;
    const size = document.getElementById('sizeInput').value;
    const color = document.getElementById('colorInput').value;
    const price_min = document.getElementById('priceMinInput').value;
    const price_max = document.getElementById('priceMaxInput').value;
    const on_sale = document.getElementById('onSaleInput').checked;

    resultsContainer.innerHTML = '';
    loadedItems = 0;
    allLoaded = false;
    currentQuery = query;
    loadedItemUrls = new Set();
    // Store filters and reset cursor for pagination
    window.currentFilters = {
        query,
        group,
        gender,
        brand,
        size,
        color,
        price_min,
        price_max,
        on_sale
    };
    window.currentCursor = null;
    showSpinner();
    fetch('/search_results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(window.currentFilters)
    })
    .then(res => res.json())
    .then(data => {
        hideSpinner();
        if (data.items) {
            resultsContainer.innerHTML = '';
            loadedItemUrls = new Set();
            loadedItems = 0;
            allLoaded = false;
            appendResults(data.items);
            loadedItems += data.items.length;
            window.currentCursor = data.next_cursor || null;
            if (!window.currentCursor || !data.items.length) allLoaded = true;
            // Immediately try to load more if cursor is set and not allLoaded
            if (window.currentCursor && !allLoaded) {
                loadMoreItems();
            }
        }
    });
});

// renderResults is now unused; all rendering is handled in appendResults for infinite scroll

// --- Infinite Scroll & Spinner ---
let currentQuery = '';
let loadedItems = 0;
let isLoadingMore = false;
let allLoaded = false;
let loadedItemUrls = new Set();

// --- Spinner setup (run once) ---
(function setupSpinner() {
    if (!document.getElementById('infiniteSpinner')) {
        const spinner = document.createElement('div');
        spinner.id = 'infiniteSpinner';
        spinner.style.display = 'none';
        spinner.style.justifyContent = 'center';
        spinner.style.alignItems = 'center';
        spinner.style.width = '100%';
        spinner.innerHTML = '<div style="margin:2rem auto; display:inline-block;"><span class="spinner" style="display:inline-block;width:32px;height:32px;border:4px solid #cbd5e1;border-top:4px solid #6366f1;border-radius:50%;animation:spin 1s linear infinite;"></span></div>';
        document.body.appendChild(spinner);
        // Spinner animation CSS
        const style = document.createElement('style');
        style.innerHTML = `@keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }`;
        document.head.appendChild(style);
    }
})();

function showSpinner() {
    let spinner = document.getElementById('infiniteSpinner');
    if (!spinner) {
        spinner = document.createElement('div');
        spinner.id = 'infiniteSpinner';
        spinner.style.display = 'none';
        spinner.style.justifyContent = 'center';
        spinner.style.alignItems = 'center';
        spinner.style.width = '100%';
        spinner.innerHTML = '<div style="margin:2rem auto; display:inline-block;"><span class="spinner" style="display:inline-block;width:32px;height:32px;border:4px solid #cbd5e1;border-top:4px solid #6366f1;border-radius:50%;animation:spin 1s linear infinite;"></span></div>';
        document.body.appendChild(spinner);
        // Spinner animation CSS
        if (!document.getElementById('spinnerStyle')) {
            const style = document.createElement('style');
            style.id = 'spinnerStyle';
            style.innerHTML = `@keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }`;
            document.head.appendChild(style);
        }
    }
    spinner.style.display = 'flex';
    // Place spinner after the grid
    const grid = document.querySelector('.results-grid');
    if (grid && spinner) {
        grid.insertAdjacentElement('afterend', spinner);
    } else if (spinner && resultsContainer) {
        resultsContainer.appendChild(spinner);
    }
}
function hideSpinner() {
    const spinner = document.getElementById('infiniteSpinner');
    if (spinner) spinner.style.display = 'none';
}

function appendResults(items) {
    let grid = document.querySelector('.results-grid');
    if (!grid) {
        grid = document.createElement('div');
        grid.className = 'results-grid';
        resultsContainer.appendChild(grid);
    }
    for (const item of items) {
        let priceHtml = '';
        if (item.price_original && item.price_sale && item.price_original !== item.price_sale) {
            priceHtml = `<span class="price-original" style="text-decoration:line-through;color:#888;margin-right:0.5em;">$${item.price_original}</span> <span class="price-sale" style="color:#222;font-weight:600;">$${item.price_sale}</span>`;
        } else {
            priceHtml = `<span class="price-sale" style="color:#222;font-weight:600;">$${item.price_sale || item.price}</span>`;
        }
        const card = document.createElement('div');
        card.className = 'depop-card';
        card.setAttribute('data-url', item.item_url);
        card.innerHTML = `
            <div class="depop-img-wrap">
              <img src="${item.image_url}" alt="Item image" class="item-img" data-img1="${item.image_url}" data-img2="${item.image_url2 || ''}">
            </div>
            <div class="depop-info">
              <div class="depop-price">${priceHtml}</div>
              <div class="depop-size">${item.size ? 'Size: ' + item.size : ''}</div>
            </div>
            <div class="depop-actions">
              <form method="post" action="/feedback" class="feedback-form">
                <input type="hidden" name="item_url" value="${item.item_url}">
                <button type="submit" name="feedback" value="like" class="like-btn emoji-circle" title="Like">&#128077;</button>
                <button type="submit" name="feedback" value="dislike" class="dislike-btn emoji-circle" title="Dislike">&#128078;</button>
              </form>
            </div>
        `;
        grid.appendChild(card);
    }
    // Re-attach handlers
    document.querySelectorAll('.depop-card').forEach(card => {
        card.addEventListener('click', function(e) {
            if (e.target.closest('button') || e.target.closest('form')) return;
            const url = this.getAttribute('data-url');
            if (url) window.open(url, '_blank');
        });
    });
    document.querySelectorAll('.item-img').forEach(img => {
        const img1 = img.getAttribute('data-img1');
        const img2 = img.getAttribute('data-img2');
        if (img2 && img2 !== '') {
            img.addEventListener('mouseenter', () => { img.src = img2; });
            img.addEventListener('mouseleave', () => { img.src = img1; });
        }
    });
}

// Patch loadMoreItems to force repaint and use cleaned spinner logic
async function loadMoreItems() {
    if (isLoadingMore || allLoaded) {
        return;
    }
    isLoadingMore = true;
    showSpinner();
    try {
        const filters = window.currentFilters || {};
        const cursor = window.currentCursor;
        console.log('[DEBUG] loadMoreItems called. Cursor:', cursor);
        const res = await fetch('/search_results_page', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...filters,
                cursor: cursor,
                max_items: 24
            })
        });
        if (!res.ok) throw new Error('Network response was not ok');
        const data = await res.json();
        hideSpinner();
        if (data.items && data.items.length) {
            appendResults(data.items);
            loadedItems += data.items.length;
            window.currentCursor = data.next_cursor || null;
            console.log('[DEBUG] New cursor after load:', window.currentCursor);
            if (!window.currentCursor) allLoaded = true;
        } else {
            allLoaded = true;
        }
    } catch (err) {
        hideSpinner();
        allLoaded = true;
        // Show error in UI
        let grid = document.querySelector('.results-grid');
        if (!grid) grid = resultsContainer;
        const errDiv = document.createElement('div');
        errDiv.style.color = 'red';
        errDiv.style.textAlign = 'center';
        errDiv.textContent = 'Error loading more results.';
        grid.appendChild(errDiv);
    } finally {
        isLoadingMore = false;
    }
}

window.addEventListener('scroll', () => {
    if (allLoaded || isLoadingMore) {
        return;
    }
    const scrollY = window.scrollY || window.pageYOffset;
    const viewport = window.innerHeight;
    const fullHeight = document.body.offsetHeight;
    if (scrollY + viewport > fullHeight - 300) {
        showSpinner();
        loadMoreItems();
    }
});

// Patch form submit to reset scroll state
form.addEventListener('submit', function(e) {
    loadedItems = 0;
    allLoaded = false;
});
