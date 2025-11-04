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

async function predictManual() {
  const home = document.getElementById('home').value.trim();
  const away = document.getElementById('away').value.trim();
  const date = document.getElementById('date').value;
  const out = document.getElementById('manualOut');
  out.textContent = 'Predicting...';
  try {
    const res = await fetch('/predict_match', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: home, away_team: away, date })
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = `Error: ${e}`;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnFetch').addEventListener('click', fetchUpcoming);
  document.getElementById('btnRecent').addEventListener('click', fetchRecent);
  document.getElementById('btnPredict').addEventListener('click', predictManual);
  // initial load
  fetchUpcoming();
  fetchRecent();
});


