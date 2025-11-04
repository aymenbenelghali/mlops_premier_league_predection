function renderTable(rows) {
  if (!rows || rows.length === 0) {
    return '<p class="badge">No fixtures found for the selected window.</p>';
  }
  const head = `
    <thead>
      <tr>
        <th>Date</th>
        <th>Fixture</th>
        <th>Goals</th>
        <th>Cards</th>
        <th>Corners</th>
        <th>Offsides</th>
      </tr>
    </thead>`;
  const body = rows.map(r => `
      <tr>
        <td>${r.date}</td>
        <td>${r.home_team} vs ${r.away_team}</td>
        <td>${r.total_goals.toFixed(2)}</td>
        <td>${r.total_cards.toFixed(2)}</td>
        <td>${r.total_corners.toFixed(2)}</td>
        <td>${r.total_offsides.toFixed(2)}</td>
      </tr>`).join('');
  return `<table>${head}<tbody>${body}</tbody></table>`;
}

async function fetchUpcoming() {
  const el = document.getElementById('fixtures');
  el.innerHTML = '<p>Loading...</p>';
  try {
    const res = await fetch('/predict_upcoming_week?days=14');
    const data = await res.json();
    el.innerHTML = renderTable(data.fixtures || []);
  } catch (e) {
    el.innerHTML = `<p style="color:#b91c1c">Error: ${e}</p>`;
  }
}

async function fetchRecent() {
  const el = document.getElementById('recent');
  el.innerHTML = '<p>Loading...</p>';
  try {
    const res = await fetch('/recent_results?days=30');
    const data = await res.json();
    el.innerHTML = renderTable(data.results || []);
  } catch (e) {
    el.innerHTML = `<p style="color:#b91c1c">Error: ${e}</p>`;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnFetch').addEventListener('click', fetchUpcoming);
  document.getElementById('btnRecent').addEventListener('click', fetchRecent);
  // initial load
  fetchUpcoming();
  fetchRecent();
});


