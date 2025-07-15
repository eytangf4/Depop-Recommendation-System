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
        html += `
        <div class="col-md-6 mb-4">
            <div class="card p-3 d-flex flex-row align-items-center">
                <img src="${item.image_url}" alt="Item image" class="item-img me-3">
                <div class="flex-grow-1">
                    <div><strong>${item.title}</strong></div>
                    <div>Price: ${item.price}</div>
                    <div>Size: ${item.size}</div>
                    <a href="${item.item_url}" target="_blank" class="btn btn-link p-0">View on Depop</a>
                </div>
                <form method="post" action="/feedback" class="ms-3">
                    <input type="hidden" name="item_url" value="${item.item_url}">
                    <button type="submit" name="feedback" value="like" class="like-btn" title="Like">&#128077;</button>
                    <button type="submit" name="feedback" value="dislike" class="dislike-btn" title="Dislike">&#128078;</button>
                </form>
            </div>
        </div>
        `;
    }
    html += '</div>';
    resultsContainer.innerHTML = html;
}
