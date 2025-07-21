// My Ratings page functionality - works exactly like search results
console.log('🔍 My Ratings JS loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('🔍 DOMContentLoaded fired, calling loadUserRatings');
    loadUserRatings();
});

let ratingsData = [];

async function loadUserRatings() {
    console.log('🔍 loadUserRatings called');
    try {
        console.log('🔍 Making fetch request to /get_user_ratings');
        const response = await fetch('/get_user_ratings', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        console.log('🔍 Response received:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('🔍 Data received:', data);
        ratingsData = data.ratings || [];
        
        updateRatingStats(data.stats || {});
        displayRatings(ratingsData);
        
    } catch (error) {
        console.error('❌ Error loading ratings:', error);
        showError('Failed to load ratings. Please try again.');
    }
}

function updateRatingStats(stats) {
    const likedCount = stats.liked || 0;
    const dislikedCount = stats.disliked || 0;
    
    document.getElementById('likedCount').textContent = `${likedCount} Liked`;
    document.getElementById('dislikedCount').textContent = `${dislikedCount} Disliked`;
}

function displayRatings(ratings) {
    const container = document.getElementById('ratingsContainer');
    const emptyState = document.getElementById('emptyState');
    
    if (!ratings || ratings.length === 0) {
        emptyState.style.display = 'block';
        return;
    }
    
    emptyState.style.display = 'none';
    
    // Create ratings grid exactly like search results
    container.innerHTML = `
        <div class="row g-3" id="ratingsGrid">
            ${ratings.map((item, index) => createRatingCard(item, index)).join('')}
        </div>
    `;
    
    // Get initial feedback states
    loadFeedbackStates();
}

function createRatingCard(item, index) {
    const isLiked = item.rating_info?.feedback_type === 'like';
    const isDisliked = item.rating_info?.feedback_type === 'dislike';
    const isLoved = item.rating_info?.feedback_type === 'love';
    
    const activeClass = isLiked ? 'like' : (isDisliked ? 'dislike' : (isLoved ? 'love' : ''));
    
    return `
        <div class="col-lg-3 col-md-4 col-sm-6">
            <div class="search-result-item rating-item" data-item-url="${item.item_url}" data-rating-id="${item.rating_info?.rating_id}">
                <div class="delete-overlay" id="deleteOverlay${index}">
                    <button class="delete-btn" onclick="deleteRating(${item.rating_info?.rating_id}, ${index})">
                        Delete Rating
                    </button>
                </div>
                <div class="item-image-container" onclick="openDepopPage('${item.item_url}')">
                    <img src="${item.item_image || item.image || 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik03NS4wMDAzIDEyNUwxMDAgOTlMMTI1IDEyNUwxNTAgOTlWODBIMTYwVjE2MEg0MFY4MEg1MFY5OUw3NS4wMDAzIDEyNVoiIGZpbGw9IiM5Q0EzQUYiLz4KPC9zdmc+Cg=='}" 
                         alt="${item.item_title || item.title}" 
                         class="item-image" 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik03NS4wMDAzIDEyNUwxMDAgOTlMMTI1IDEyNUwxNTAgOTlWODBIMTYwVjE2MEg0MFY4MEg1MFY5OUw3NS4wMDAzIDEyNVoiIGZpbGw9IiM5Q0EzQUYiLz4KPC9zdmc+Cg=='">
                    <div class="item-price">$${item.item_price || item.price}</div>
                </div>
                <div class="item-details">
                    <div class="item-title" onclick="openDepopPage('${item.item_url}')">${item.item_title || item.title}</div>
                    <div class="item-brand">${item.item_brand || item.brand || 'Unknown Brand'}</div>
                    <div class="item-size">${formatSizes(item.sizes)}</div>
                    <div class="rating-info">
                        <small class="text-muted">Rated ${formatDate(item.rating_info?.timestamp)}</small>
                    </div>
                    <div class="feedback-buttons mt-2" data-item-url="${item.item_url}">
                        <button class="feedback-btn like-btn ${activeClass === 'like' ? 'active' : ''}" 
                                data-feedback="like" 
                                onclick="handleFeedback(event, '${item.item_url}', 'like')">
                            <i class="fas fa-thumbs-up"></i>
                        </button>
                        <button class="feedback-btn dislike-btn ${activeClass === 'dislike' ? 'active' : ''}" 
                                data-feedback="dislike" 
                                onclick="handleFeedback(event, '${item.item_url}', 'dislike')">
                            <i class="fas fa-thumbs-down"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

async function loadFeedbackStates() {
    // Get all item URLs currently displayed
    const itemUrls = ratingsData.map(item => item.item_url).filter(url => url);
    
    if (itemUrls.length === 0) return;
    
    try {
        const response = await fetch('/get_user_feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                item_urls: itemUrls
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const feedback = data.feedback || {};
        
        // Update button states
        updateFeedbackButtons(feedback);
        
    } catch (error) {
        console.error('Error loading feedback states:', error);
    }
}

function updateFeedbackButtons(feedbackData) {
    Object.entries(feedbackData).forEach(([itemUrl, feedbackType]) => {
        const feedbackContainer = document.querySelector(`[data-item-url="${itemUrl}"] .feedback-buttons`);
        if (!feedbackContainer) return;
        
        const likeBtn = feedbackContainer.querySelector('.like-btn');
        const dislikeBtn = feedbackContainer.querySelector('.dislike-btn');
        
        // Reset buttons
        likeBtn.classList.remove('active');
        dislikeBtn.classList.remove('active');
        
        // Set active button
        if (feedbackType === 'like' || feedbackType === 'love') {
            likeBtn.classList.add('active');
        } else if (feedbackType === 'dislike') {
            dislikeBtn.classList.add('active');
        }
    });
}

async function handleFeedback(event, itemUrl, feedbackType) {
    event.preventDefault();
    event.stopPropagation();
    
    const button = event.currentTarget;
    const feedbackContainer = button.parentElement;
    const likeBtn = feedbackContainer.querySelector('.like-btn');
    const dislikeBtn = feedbackContainer.querySelector('.dislike-btn');
    
    const currentlyActive = button.classList.contains('active');
    const currentFeedbackType = likeBtn.classList.contains('active') ? 'like' : 
                              (dislikeBtn.classList.contains('active') ? 'dislike' : null);
    
    // If clicking the same active button, show delete overlay
    if (currentlyActive && currentFeedbackType === feedbackType) {
        const ratingItem = button.closest('.rating-item');
        const ratingId = ratingItem.getAttribute('data-rating-id');
        const index = Array.from(ratingItem.parentElement.children).indexOf(ratingItem.parentElement);
        showDeleteOverlay(index);
        return;
    }
    
    // Provide immediate visual feedback
    button.disabled = true;
    const originalContent = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    
    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                item_url: itemUrl,
                feedback: feedbackType,
                item_title: getItemTitle(itemUrl),
                item_brand: getItemBrand(itemUrl),
                item_price: getItemPrice(itemUrl),
                item_sizes: getItemSizes(itemUrl),
                item_image: getItemImage(itemUrl)
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        // Update button states
        likeBtn.classList.remove('active');
        dislikeBtn.classList.remove('active');
        button.classList.add('active');
        
        // Update the rating in our local data
        const ratingItem = ratingsData.find(item => item.item_url === itemUrl);
        if (ratingItem && ratingItem.rating_info) {
            ratingItem.rating_info.feedback_type = feedbackType;
        }
        
        showSuccess(`Item ${feedbackType}d successfully!`);
        
        // Reload stats
        setTimeout(() => {
            loadUserRatings();
        }, 500);

    } catch (error) {
        console.error('Error submitting feedback:', error);
        showError(`Failed to ${feedbackType} item. Please try again.`);
    } finally {
        // Restore button
        button.disabled = false;
        button.innerHTML = originalContent;
    }
}

// Helper functions to get item data from the DOM
function getItemTitle(itemUrl) {
    const item = document.querySelector(`[data-item-url="${itemUrl}"] .item-title`);
    return item ? item.textContent : '';
}

function getItemBrand(itemUrl) {
    const item = document.querySelector(`[data-item-url="${itemUrl}"] .item-brand`);
    return item ? item.textContent : '';
}

function getItemPrice(itemUrl) {
    const item = document.querySelector(`[data-item-url="${itemUrl}"] .item-price`);
    return item ? parseFloat(item.textContent.replace('$', '')) : 0;
}

function getItemSizes(itemUrl) {
    const item = document.querySelector(`[data-item-url="${itemUrl}"] .item-size`);
    return item ? item.textContent : '';
}

function getItemImage(itemUrl) {
    const item = document.querySelector(`[data-item-url="${itemUrl}"] .item-image`);
    return item ? item.src : '';
}

function showDeleteOverlay(index) {
    const overlay = document.getElementById(`deleteOverlay${index}`);
    const ratingItem = overlay.parentElement;
    
    ratingItem.classList.add('deleting');
    overlay.classList.add('show');
    
    // Hide overlay after 3 seconds if not clicked
    setTimeout(() => {
        hideDeleteOverlay(index);
    }, 3000);
}

function hideDeleteOverlay(index) {
    const overlay = document.getElementById(`deleteOverlay${index}`);
    const ratingItem = overlay ? overlay.parentElement : null;
    
    if (overlay) overlay.classList.remove('show');
    if (ratingItem) ratingItem.classList.remove('deleting');
}

async function deleteRating(ratingId, index) {
    try {
        const response = await fetch('/delete_rating', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                rating_id: ratingId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showSuccess('Rating deleted successfully!');
            
            // Reload the entire ratings display
            setTimeout(() => {
                loadUserRatings();
            }, 300);
        } else {
            throw new Error(result.error || 'Failed to delete rating');
        }
        
    } catch (error) {
        console.error('Error deleting rating:', error);
        showError('Failed to delete rating. Please try again.');
    }
}

function openDepopPage(url) {
    window.open(url, '_blank');
}

function formatSizes(sizes) {
    if (!sizes) return 'Size not specified';
    if (typeof sizes === 'string') {
        try {
            sizes = JSON.parse(sizes);
        } catch (e) {
            return sizes;
        }
    }
    if (Array.isArray(sizes)) {
        return sizes.join(', ');
    }
    return sizes;
}

function formatDate(timestamp) {
    if (!timestamp) return 'Unknown date';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) {
        return 'today';
    } else if (diffDays <= 7) {
        return `${diffDays} days ago`;
    } else if (diffDays <= 30) {
        return `${Math.ceil(diffDays / 7)} weeks ago`;
    } else {
        return date.toLocaleDateString();
    }
}

function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'alert alert-success position-fixed';
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
    toast.innerHTML = `
        <i class="fas fa-check-circle me-2"></i>
        ${message}
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'alert alert-danger position-fixed';
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
    toast.innerHTML = `
        <i class="fas fa-exclamation-circle me-2"></i>
        ${message}
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

async function handleRatingChange(ratingId, index, newFeedback) {
    const currentRating = ratingsData[index];
    const currentFeedback = currentRating.feedback_type;
    
    // If clicking the same button, show delete overlay
    if (currentFeedback === newFeedback) {
        showDeleteOverlay(index);
        return;
    }
    
    // Change rating
    try {
        const response = await fetch('/update_rating', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                rating_id: ratingId,
                feedback_type: newFeedback
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Update local data
            ratingsData[index].feedback_type = newFeedback;
            
            // Update UI
            updateRatingButtons(index, newFeedback);
            
            // Update stats
            updateRatingStats(result.stats);
            
            showSuccess('Rating updated successfully!');
        } else {
            throw new Error(result.error || 'Failed to update rating');
        }
        
    } catch (error) {
        console.error('Error updating rating:', error);
        showError('Failed to update rating. Please try again.');
    }
}

function updateRatingButtons(index, feedback) {
    const ratingItem = document.querySelector(`[data-index="${index}"]`);
    const likeBtn = ratingItem.querySelector('.like-btn');
    const dislikeBtn = ratingItem.querySelector('.dislike-btn');
    
    // Reset both buttons
    likeBtn.classList.remove('active');
    dislikeBtn.classList.remove('active');
    
    // Set active button
    if (feedback === 'like') {
        likeBtn.classList.add('active');
    } else if (feedback === 'dislike') {
        dislikeBtn.classList.add('active');
    }
}

function showDeleteOverlay(index) {
    const overlay = document.getElementById(`deleteOverlay${index}`);
    const ratingItem = document.querySelector(`[data-index="${index}"] .rating-item`);
    
    ratingItem.classList.add('deleting');
    overlay.classList.add('show');
    
    // Hide overlay after 3 seconds if not clicked
    setTimeout(() => {
        hideDeleteOverlay(index);
    }, 3000);
}

function hideDeleteOverlay(index) {
    const overlay = document.getElementById(`deleteOverlay${index}`);
    const ratingItem = document.querySelector(`[data-index="${index}"] .rating-item`);
    
    if (overlay) overlay.classList.remove('show');
    if (ratingItem) ratingItem.classList.remove('deleting');
}

async function deleteRating(ratingId, index) {
    try {
        const response = await fetch('/delete_rating', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                rating_id: ratingId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Remove from local data
            ratingsData.splice(index, 1);
            
            // Remove from UI with animation
            const ratingElement = document.querySelector(`[data-rating-id="${ratingId}"]`);
            ratingElement.style.transition = 'all 0.3s ease';
            ratingElement.style.opacity = '0';
            ratingElement.style.transform = 'scale(0.8)';
            
            setTimeout(() => {
                ratingElement.remove();
                reindexRatings();
                updateRatingStats(result.stats);
                
                // Show empty state if no ratings left
                if (ratingsData.length === 0) {
                    document.getElementById('emptyState').style.display = 'block';
                }
            }, 300);
            
            showSuccess('Rating deleted successfully!');
        } else {
            throw new Error(result.error || 'Failed to delete rating');
        }
        
    } catch (error) {
        console.error('Error deleting rating:', error);
        showError('Failed to delete rating. Please try again.');
    }
}

function reindexRatings() {
    const ratingElements = document.querySelectorAll('[data-index]');
    ratingElements.forEach((element, newIndex) => {
        element.setAttribute('data-index', newIndex);
        
        // Update delete overlay ID
        const overlay = element.querySelector('.delete-overlay');
        if (overlay) {
            overlay.id = `deleteOverlay${newIndex}`;
        }
        
        // Update onclick handlers
        const deleteBtn = element.querySelector('.delete-btn');
        if (deleteBtn) {
            const ratingId = element.getAttribute('data-rating-id');
            deleteBtn.setAttribute('onclick', `deleteRating(${ratingId}, ${newIndex})`);
        }
        
        const likeBtn = element.querySelector('.like-btn');
        const dislikeBtn = element.querySelector('.dislike-btn');
        const ratingId = element.getAttribute('data-rating-id');
        
        if (likeBtn) {
            likeBtn.setAttribute('onclick', `handleRatingChange(${ratingId}, ${newIndex}, 'like')`);
        }
        if (dislikeBtn) {
            dislikeBtn.setAttribute('onclick', `handleRatingChange(${ratingId}, ${newIndex}, 'dislike')`);
        }
    });
}

function getItemImageUrl(item) {
    // If we have an image URL stored, use it
    if (item.item_image) {
        return item.item_image;
    }
    
    // Try to construct Depop image URL from item URL
    if (item.item_url) {
        // Extract product ID from Depop URL
        const match = item.item_url.match(/\/products\/(\d+)/);
        if (match) {
            const productId = match[1];
            // Depop's image URL pattern (this might need adjustment)
            return `https://d2h1pu99sxkfvn.cloudfront.net/b0/19/${productId}/m_0.jpg`;
        }
    }
    
    // Fallback to placeholder
    return 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik03NS4wMDAzIDEyNUwxMDAgOTlMMTI1IDEyNUwxNTAgOTlWODBIMTYwVjE2MEg0MFY4MEg1MFY5OUw3NS4wMDAzIDEyNVoiIGZpbGw9IiM5Q0EzQUYiLz4KPC9zdmc+Cg==';
}

function openDepopPage(url) {
    window.open(url, '_blank');
}

function formatSizes(sizes) {
    if (!sizes) return 'Size not specified';
    if (typeof sizes === 'string') {
        try {
            sizes = JSON.parse(sizes);
        } catch (e) {
            return sizes;
        }
    }
    if (Array.isArray(sizes)) {
        return sizes.join(', ');
    }
    return sizes;
}

function formatDate(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) {
        return 'today';
    } else if (diffDays <= 7) {
        return `${diffDays} days ago`;
    } else if (diffDays <= 30) {
        return `${Math.ceil(diffDays / 7)} weeks ago`;
    } else {
        return date.toLocaleDateString();
    }
}
