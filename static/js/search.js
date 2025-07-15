const form = document.getElementById('searchForm');
const loadingBarContainer = document.getElementById('loadingBarContainer');
const loadingBar = document.getElementById('loadingBar');
const loadingText = document.getElementById('loadingText');
const resultsContainer = document.getElementById('resultsContainer');
form.addEventListener('submit', function(e) {
    e.preventDefault();
    const query = document.getElementById('queryInput').value;
    resultsContainer.innerHTML = '';
    loadingBarContainer.style.display = 'block';
    loadingBar.style.width = '0%';
    loadingText.innerText = 'Loading results...';
    fetch('/search_results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    })
    .then(res => res.json())
    .then(data => {
        const progressId = data.progress_id;
        const evtSource = new EventSource(`/search_progress/${progressId}`);
        evtSource.onmessage = function(event) {
            const update = JSON.parse(event.data);
            loadingBar.style.width = update.progress + '%';
            loadingText.innerText = update.status;
            if (update.progress >= 100) {
                evtSource.close();
                loadingBarContainer.style.display = 'none';
                if (update.items) {
                    renderResults(update.items);
                }
            }
        };
    });
});

function renderResults(items) {
    if (!items.length) {
        resultsContainer.innerHTML = '<div class="text-center">No results found.</div>';
        return;
    }
    resultsContainer.innerHTML = '';
    loadedItems = 0;
    allLoaded = false;
    currentQuery = document.getElementById('queryInput').value;
    appendResults(items);
    loadedItems += items.length;
}

// --- Infinite Scroll & Spinner ---
let currentQuery = '';
let loadedItems = 0;
let isLoadingMore = false;
let allLoaded = false;

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
    const spinner = document.getElementById('infiniteSpinner');
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
        if (item.price_original && item.price_original !== item.price_sale) {
            priceHtml = `<span class="price-original">${item.price_original}</span> <span class="price-sale">${item.price_sale}</span>`;
        } else {
            priceHtml = `<span class="price-sale">${item.price_sale || item.price}</span>`;
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
            if (e.target.closest('button')) return;
            window.open(this.getAttribute('data-url'), '_blank');
        });
    });
    document.querySelectorAll('.item-img').forEach(img => {
        const img1 = img.getAttribute('data-img1');
        const img2 = img.getAttribute('data-img2');
        if (img2) {
            img.addEventListener('mouseenter', () => { img.src = img2; });
            img.addEventListener('mouseleave', () => { img.src = img1; });
        }
    });
}

// Patch loadMoreItems to force repaint and use cleaned spinner logic
async function loadMoreItems() {
    if (isLoadingMore || allLoaded) return;
    isLoadingMore = true;
    showSpinner();
    await new Promise(requestAnimationFrame); // Force repaint
    const res = await fetch('/search_results_page', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentQuery, offset: loadedItems, max_items: 10 })
    });
    const data = await res.json();
    hideSpinner();
    if (data.items && data.items.length) {
        appendResults(data.items);
        loadedItems += data.items.length;
        if (data.items.length < 10) allLoaded = true;
    } else {
        allLoaded = true;
    }
    isLoadingMore = false;
}

window.addEventListener('scroll', () => {
    if (allLoaded || isLoadingMore) return;
    const scrollY = window.scrollY || window.pageYOffset;
    const viewport = window.innerHeight;
    const fullHeight = document.body.offsetHeight;
    if (scrollY + viewport > fullHeight - 300) {
        loadMoreItems();
    }
});

// Patch form submit to reset scroll state
form.addEventListener('submit', function(e) {
    loadedItems = 0;
    allLoaded = false;
});
