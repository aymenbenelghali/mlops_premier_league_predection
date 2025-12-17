/**
 * Premier League Predictions Frontend
 * Loads upcoming fixtures and recent results from API
 * Supports both XGBoost and LSTM models
 */

let selectedModel = "xgboost";

async function loadUpcomingMatches() {
  try {
    const response = await fetch('/api/upcoming');
    const data = await response.json();
    const matches = data.matches || [];
    
    let html = '<table><thead><tr><th>Date</th><th>Home Team</th><th>Away Team</th><th>Action</th></tr></thead><tbody>';
    
    if (matches.length === 0) {
      html += '<tr><td colspan="4" style="text-align:center;color:var(--text-secondary);">No upcoming matches</td></tr>';
    } else {
      for (const match of matches) {
        html += `<tr>
          <td>${match.date}</td>
          <td><strong>${match.home_team}</strong></td>
          <td><strong>${match.away_team}</strong></td>
          <td><button class="btn btn-primary" style="padding:6px 12px;font-size:12px;" onclick="predictMatch('${match.home_team}', '${match.away_team}', '${match.date}')"><i class="fas fa-magic"></i> Predict</button></td>
        </tr>`;
      }
    }
    
    html += '</tbody></table>';
    document.getElementById('fixtures').innerHTML = html;
  } catch (error) {
    console.error('Error loading upcoming matches:', error);
    document.getElementById('fixtures').innerHTML = '<p style="color:red;">Error loading matches</p>';
  }
}

async function loadRecentResults() {
  try {
    const response = await fetch('/api/recent_results');
    const data = await response.json();
    const results = data.results || [];
    
    let html = '<table><thead><tr><th>Date</th><th>Home</th><th>Score</th><th>Away</th></tr></thead><tbody>';
    
    if (results.length === 0) {
      html += '<tr><td colspan="4" style="text-align:center;color:var(--text-secondary);">No recent results</td></tr>';
    } else {
      for (const result of results) {
        html += `<tr>
          <td>${result.date}</td>
          <td><strong>${result.home_team}</strong></td>
          <td><span style="color:var(--accent-blue);font-weight:bold;">${result.home_goals}-${result.away_goals}</span></td>
          <td><strong>${result.away_team}</strong></td>
        </tr>`;
      }
    }
    
    html += '</tbody></table>';
    document.getElementById('recent').innerHTML = html;
  } catch (error) {
    console.error('Error loading recent results:', error);
    document.getElementById('recent').innerHTML = '<p style="color:red;">Error loading results</p>';
  }
}

async function predictMatch(homeTeam, awayTeam, date) {
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        home_team: homeTeam,
        away_team: awayTeam,
        date: date,
        model: selectedModel
      })
    });
    
    const pred = await response.json();
    const modelEmoji = selectedModel === 'xgboost' ? '🌳' : '🧠';
    const modelName = selectedModel === 'xgboost' ? 'XGBoost' : 'LSTM';
    
    alert(`${modelEmoji} ${modelName} Prediction: ${homeTeam} vs ${awayTeam}\n\n📊 Total Goals: ${pred.total_goals.toFixed(2)}\n🟨 Total Cards: ${pred.total_cards.toFixed(2)}\n🔲 Total Corners: ${pred.total_corners.toFixed(2)}\n📈 Confidence: ${(pred.confidence * 100).toFixed(1)}%`);
  } catch (error) {
    console.error('Prediction error:', error);
    alert('Error making prediction');
  }
}

// Load data on page load
document.addEventListener('DOMContentLoaded', () => {
  loadUpcomingMatches();
  loadRecentResults();
  
  const fetchBtn = document.getElementById('btnFetch');
  const recentBtn = document.getElementById('btnRecent');
  const modelSelect = document.getElementById('model-select');
  
  if (fetchBtn) fetchBtn.addEventListener('click', loadUpcomingMatches);
  if (recentBtn) recentBtn.addEventListener('click', loadRecentResults);
  
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      selectedModel = e.target.value;
      const modelName = selectedModel === 'xgboost' ? 'XGBoost' : 'LSTM';
      document.getElementById('model-name').textContent = `Model: ${modelName}`;
      console.log(`Switched to ${modelName} model`);
    });
  }
});

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnFetch').addEventListener('click', fetchUpcoming);
  document.getElementById('btnRecent').addEventListener('click', fetchRecent);
  // initial load
  fetchUpcoming();
  fetchRecent();
});


