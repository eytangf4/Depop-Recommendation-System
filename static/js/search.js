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
    let html = '<div class="row">';
    for (const item of items) {
        // Price display logic
        let priceHtml = '';
        if (item.price_original && item.price_original !== item.price_sale) {
            priceHtml = `<span class="price-original">${item.price_original}</span> <span class="price-sale">${item.price_sale}</span>`;
        } else {
            priceHtml = `<span class="price-sale">${item.price_sale || item.price}</span>`;
        }
        html += `
        <div class="col-md-4 mb-4">
            <div class="depop-card" data-url="${item.item_url}">
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
              </div>
        </div>
        `;
    }
    html += '</div>';
    resultsContainer.innerHTML = html;

    // Card click handler
    document.querySelectorAll('.depop-card').forEach(card => {
        card.addEventListener('click', function(e) {
            // Prevent click if like/dislike button is pressed
            if (e.target.closest('button')) return;
            window.open(this.getAttribute('data-url'), '_blank');
        });
    });
    // Image hover handler
    document.querySelectorAll('.item-img').forEach(img => {
        const img1 = img.getAttribute('data-img1');
        const img2 = img.getAttribute('data-img2');
        if (img2) {
            img.addEventListener('mouseenter', () => { img.src = img2; });
            img.addEventListener('mouseleave', () => { img.src = img1; });
        }
    });
}
